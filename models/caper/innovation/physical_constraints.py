from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicalData:
    occupancy_count: np.ndarray
    duration_hours: np.ndarray
    volume_kwh: np.ndarray
    volume_11kw_kwh: np.ndarray
    capacity: np.ndarray
    time: pd.DatetimeIndex
    zone_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_data_provenance(data_dir: Path) -> dict[str, object] | None:
    """Fail closed when the frozen UrbanEV files differ from their lock.

    External adapters may use this loader without an UrbanEV lock; the check is
    activated automatically only for a reproduction tree containing
    innovation/DATA_PROVENANCE_LOCK.json.
    """

    root = data_dir.resolve().parents[1]
    lock_path = root / "innovation" / "DATA_PROVENANCE_LOCK.json"
    if not lock_path.exists():
        return None
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, expected in payload["files"].items():
        path = data_dir / name
        if not path.exists():
            raise RuntimeError(f"locked UrbanEV file missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != int(expected["bytes"]) or actual_hash != str(expected["sha256"]).upper():
            raise RuntimeError(
                f"UrbanEV provenance mismatch for {name}: "
                f"bytes={actual_size}, sha256={actual_hash}"
            )
    return payload


def _read_zone_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.columns = frame.columns.astype(str)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains non-finite values")
    if np.any(values < 0):
        raise ValueError(f"{path.name} contains negative values")
    return frame


def load_physical_data(data_dir: Path) -> PhysicalData:
    """Load and strictly align the four UrbanEV charging targets.

    Occupancy is an unavailable/busy pile count. Duration is the sum of
    actively-delivering pile-hours within an hour, and volume is the energy
    constructed from active duration and rated pile power. These meanings are
    intentionally kept separate here.
    """

    provenance = verify_data_provenance(data_dir)

    names = ("occupancy.csv", "duration.csv", "volume.csv", "volume-11kW.csv")
    frames = {name: _read_zone_frame(data_dir / name) for name in names}
    reference = frames["occupancy.csv"]
    for name, frame in frames.items():
        if not frame.index.equals(reference.index):
            raise ValueError(f"timestamp mismatch: occupancy.csv vs {name}")
        if not frame.columns.equals(reference.columns):
            raise ValueError(f"zone order mismatch: occupancy.csv vs {name}")

    info = pd.read_csv(data_dir / "inf.csv")
    info["TAZID"] = info["TAZID"].astype(str)
    capacity_by_zone = info.groupby("TAZID", sort=False)["charge_count"].sum()
    missing = [zone for zone in reference.columns if zone not in capacity_by_zone.index]
    if missing:
        raise ValueError(f"missing capacity for zones: {missing[:5]}")
    capacity = capacity_by_zone.loc[reference.columns].to_numpy(dtype=np.float64)
    if np.any(capacity <= 0):
        raise ValueError("capacity must be positive")

    result = PhysicalData(
        occupancy_count=frames["occupancy.csv"].to_numpy(dtype=np.float64),
        duration_hours=frames["duration.csv"].to_numpy(dtype=np.float64),
        volume_kwh=frames["volume.csv"].to_numpy(dtype=np.float64),
        volume_11kw_kwh=frames["volume-11kW.csv"].to_numpy(dtype=np.float64),
        capacity=capacity,
        time=pd.DatetimeIndex(reference.index),
        zone_ids=tuple(reference.columns),
    )
    if provenance is not None:
        expected_steps = int(provenance["expected_time_steps"])
        expected_nodes = int(provenance["expected_nodes"])
        if result.occupancy_count.shape != (expected_steps, expected_nodes):
            raise RuntimeError(
                "locked UrbanEV shape mismatch: "
                f"{result.occupancy_count.shape} != {(expected_steps, expected_nodes)}"
            )
    return result


def effective_power_kw(
    volume_kwh: np.ndarray, duration_hours: np.ndarray, *, tolerance: float = 1e-12
) -> np.ndarray:
    """Return V/D where charging duration is positive, NaN otherwise."""

    if volume_kwh.shape != duration_hours.shape:
        raise ValueError("volume and duration shapes must match")
    result = np.full(duration_hours.shape, np.nan, dtype=np.float64)
    mask = duration_hours > tolerance
    result[mask] = volume_kwh[mask] / duration_hours[mask]
    return result


def decode_duration_power(
    raw_duration: torch.Tensor,
    raw_power: torch.Tensor,
    capacity: torch.Tensor,
    power_low: torch.Tensor | float,
    power_high: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-constraint decoder for aggregate duration, power, and volume.

    The decoder guarantees 0 <= duration <= capacity, a bounded positive
    effective power, and volume == duration * effective_power by construction.
    Bounds must be estimated from the training part of each fold.
    """

    if raw_duration.shape != raw_power.shape:
        raise ValueError("duration and power logits must have identical shapes")
    capacity = torch.as_tensor(capacity, dtype=raw_duration.dtype, device=raw_duration.device)
    power_low = torch.as_tensor(power_low, dtype=raw_duration.dtype, device=raw_duration.device)
    power_high = torch.as_tensor(power_high, dtype=raw_duration.dtype, device=raw_duration.device)
    if torch.any(capacity <= 0):
        raise ValueError("capacity must be positive")
    if torch.any(power_low < 0) or torch.any(power_high <= power_low):
        raise ValueError("power bounds must satisfy 0 <= low < high")
    duration = capacity * torch.sigmoid(raw_duration)
    power = power_low + (power_high - power_low) * torch.sigmoid(raw_power)
    volume = duration * power
    return duration, power, volume


def decode_duration_positive_power(
    raw_duration: torch.Tensor,
    raw_power: torch.Tensor,
    capacity: torch.Tensor,
    power_reference_kw: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Main PACE-EV decoder without inventing a universal upper power bound.

    ``power_reference_kw`` is a positive scale fitted on the training segment;
    it does not constrain the support.  The construction guarantees a
    capacity-feasible duration, non-negative effective power, and the exact
    aggregate energy identity ``volume == duration * power``.
    """

    if raw_duration.shape != raw_power.shape:
        raise ValueError("duration and power logits must have identical shapes")
    capacity = torch.as_tensor(capacity, dtype=raw_duration.dtype, device=raw_duration.device)
    power_reference = torch.as_tensor(
        power_reference_kw, dtype=raw_duration.dtype, device=raw_duration.device
    )
    if torch.any(capacity <= 0):
        raise ValueError("capacity must be positive")
    if torch.any(power_reference <= 0):
        raise ValueError("power_reference_kw must be positive")
    duration = capacity * torch.sigmoid(raw_duration)
    power = power_reference * F.softplus(raw_power)
    volume = duration * power
    return duration, power, volume


def decode_duration_empirical_power_bounds(
    raw_duration: torch.Tensor,
    raw_power: torch.Tensor,
    capacity: torch.Tensor,
    power_low: torch.Tensor,
    power_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode using train-only per-zone effective-power support.

    Equal lower/upper bounds are allowed: some regions expose only one
    effective rated-power level in an early fold.  These bounds are an
    empirical device-support assumption and must never be fitted on validation
    or test observations.
    """

    if raw_duration.shape != raw_power.shape:
        raise ValueError("duration and power logits must have identical shapes")
    tensors = [
        torch.as_tensor(value, dtype=raw_duration.dtype, device=raw_duration.device)
        for value in (capacity, power_low, power_high)
    ]
    capacity, power_low, power_high = tensors
    if torch.any(capacity <= 0):
        raise ValueError("capacity must be positive")
    if torch.any(power_low < 0) or torch.any(power_high < power_low):
        raise ValueError("empirical power bounds must satisfy 0 <= low <= high")
    duration = capacity * torch.sigmoid(raw_duration)
    power = power_low + (power_high - power_low) * torch.sigmoid(raw_power)
    volume = duration * power
    return duration, power, volume


def physical_violation_summary(data: PhysicalData, *, tolerance: float = 1e-12) -> dict:
    capacity = data.capacity[np.newaxis, :]
    occupancy = data.occupancy_count
    duration = data.duration_hours
    volume = data.volume_kwh
    volume_11kw = data.volume_11kw_kwh
    power = effective_power_kw(volume, duration, tolerance=tolerance)
    power_11kw = effective_power_kw(volume_11kw, duration, tolerance=tolerance)
    finite_power = power[np.isfinite(power)]
    finite_power_11kw = power_11kw[np.isfinite(power_11kw)]

    quantile_levels = np.asarray([0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])

    def quantiles(values: np.ndarray) -> dict[str, float]:
        result = np.quantile(values, quantile_levels)
        return {f"q{level:g}": float(value) for level, value in zip(quantile_levels, result)}

    rounded_distance = np.abs(occupancy - np.round(occupancy))
    duration_grid_error = np.abs(duration * 12.0 - np.round(duration * 12.0))
    return {
        "hours": int(len(data.time)),
        "zones": int(len(data.zone_ids)),
        "cells": int(occupancy.size),
        "time_start": data.time[0].isoformat(),
        "time_end": data.time[-1].isoformat(),
        "capacity_min": float(data.capacity.min()),
        "capacity_max": float(data.capacity.max()),
        "occupancy_below_zero": int(np.count_nonzero(occupancy < -tolerance)),
        "occupancy_above_capacity": int(np.count_nonzero(occupancy - capacity > tolerance)),
        "occupancy_half_integer_cells": int(np.count_nonzero(np.isclose(rounded_distance, 0.5, atol=1e-8))),
        "duration_below_zero": int(np.count_nonzero(duration < -tolerance)),
        "duration_above_capacity": int(np.count_nonzero(duration - capacity > tolerance)),
        "duration_5min_grid_max_error": float(duration_grid_error.max()),
        "main_volume_duration_zero_mismatch": int(
            np.count_nonzero((duration <= tolerance) != (volume <= tolerance))
        ),
        "volume_11kw_duration_zero_mismatch": int(
            np.count_nonzero((duration <= tolerance) != (volume_11kw <= tolerance))
        ),
        "duration_gt_raw_occupancy_tolerant": int(
            np.count_nonzero(duration - occupancy > tolerance)
        ),
        "duration_gt_raw_occupancy_strict": int(np.count_nonzero(duration > occupancy)),
        "raw_occupancy_zero_duration_positive": int(
            np.count_nonzero((occupancy <= tolerance) & (duration > tolerance))
        ),
        "effective_power_positive_cells": int(finite_power.size),
        "effective_power_kw_quantiles": quantiles(finite_power),
        "volume_11kw_effective_power_kw_quantiles": quantiles(finite_power_11kw),
        "volume_11kw_max_abs_error_from_11kw": float(
            np.max(np.abs(finite_power_11kw - 11.0))
        ),
    }
