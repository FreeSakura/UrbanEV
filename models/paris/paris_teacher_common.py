"""Shared, fail-closed contracts for Paris development-only teacher qualification."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

FOLDS = ("2020-08", "2020-09", "2020-10", "2020-11")
HORIZONS = (3, 6, 9, 12)
MODELS = ("Paris_CAPER_phase_only", "TimeXer_local_audited_compact_L168")
STATE_NAMES = ("available", "active", "unavailable")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class DevelopmentBundle:
    time: pd.DatetimeIndex
    stations: np.ndarray
    rate_observed: np.ndarray
    observed_mask: np.ndarray
    capacity_observed: np.ndarray
    state_observed: np.ndarray
    paths: dict[str, Path]


def assert_sealed_boundaries(root: Path) -> None:
    guard = json.loads((root / "experiments/08_paris_new_evidence/PROTECTED_ACCESS_GUARD.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "experiments/08_paris_new_evidence/DATA_BOUNDARY_LOCK.json").read_text(encoding="utf-8"))
    if not (
        guard.get("status") == "SEALED"
        and guard.get("access_count") == 0
        and guard.get("target_materialized") is False
        and lock.get("formal_target_access") is False
        and lock.get("protected_target_access") is False
        and lock.get("protected_open_count") == 0
    ):
        raise RuntimeError("formal/protected boundary guard drift")


def load_bundle(root: Path) -> DevelopmentBundle:
    assert_sealed_boundaries(root)
    dev = (root / "experiments/08_paris_new_evidence/prepared_v1/development").resolve()
    rate_path = dev / "non_available_port_rate_observed.csv"
    mask_path = dev / "observation_mask.csv"
    cap_path = dev / "station_capacity.csv"
    state_path = dev / "three_state_rates_observed.npz"
    for path in (rate_path, mask_path, cap_path, state_path):
        if not path.is_relative_to(dev) or not path.exists():
            raise FileNotFoundError(f"development-only artifact missing: {path}")
    rate = pd.read_csv(rate_path, index_col=0, parse_dates=True)
    mask = pd.read_csv(mask_path, index_col=0, parse_dates=True).astype(bool)
    state = np.load(state_path, allow_pickle=False)
    stations = np.asarray(rate.columns).astype("U")
    if rate.shape != (3624, 50) or not rate.index.equals(mask.index) or list(rate.columns) != list(mask.columns):
        raise AssertionError("development target identity drift")
    if not np.array_equal(state["time_ns"].astype("<i8"), rate.index.view("i8")):
        raise AssertionError("three-state time identity drift")
    if not np.array_equal(state["station_ids"].astype("U"), stations):
        raise AssertionError("three-state station identity drift")
    state_values = state["state_rate"].astype(np.float32)
    capacity_observed = state["capacity_count"].astype(np.float32)
    state_mask = state["observation_mask"].astype(bool)
    if (
        state_values.shape != (3624, 50, 3)
        or capacity_observed.shape != (3624, 50)
        or not np.array_equal(state_mask, mask.to_numpy(bool))
        or not np.array_equal(np.isfinite(capacity_observed), state_mask)
    ):
        raise AssertionError("three-state mask/shape drift")
    total = state_values[..., 1] + state_values[..., 2]
    observed = mask.to_numpy(bool)
    if not np.allclose(total[observed], rate.to_numpy(np.float32)[observed], atol=1e-6, rtol=0.0):
        raise AssertionError("three-state target reconstruction drift")
    return DevelopmentBundle(
        time=rate.index,
        stations=stations,
        rate_observed=rate.to_numpy(np.float32),
        observed_mask=observed,
        capacity_observed=capacity_observed,
        state_observed=state_values,
        paths={"rate": rate_path, "mask": mask_path, "capacity": cap_path, "state": state_path},
    )


def _fallback_fill(values: np.ndarray, mask: np.ndarray, time: pd.DatetimeIndex, fit_end: int) -> np.ndarray:
    """Causal ffill(4), then train-only station-hour-of-week and station medians."""
    frame = pd.DataFrame(values, index=time)
    causal = frame.ffill(limit=4).to_numpy(np.float64)
    train = values[: fit_end + 1].astype(np.float64)
    train_mask = mask[: fit_end + 1]
    train_nan = np.where(train_mask, train, np.nan)
    station_median = np.nanmedian(train_nan, axis=0)
    global_median = float(np.nanmedian(train_nan))
    station_median = np.where(np.isfinite(station_median), station_median, global_median)
    how = time.dayofweek.to_numpy() * 24 + time.hour.to_numpy()
    how_median = np.full((168, values.shape[1]), np.nan, dtype=np.float64)
    for hour in range(168):
        chosen = np.flatnonzero((how[: fit_end + 1] == hour))
        if chosen.size:
            how_median[hour] = np.nanmedian(train_nan[chosen], axis=0)
    for index in range(len(time)):
        missing = ~np.isfinite(causal[index])
        if missing.any():
            fallback = np.where(np.isfinite(how_median[how[index]]), how_median[how[index]], station_median)
            causal[index, missing] = fallback[missing]
        missing = ~np.isfinite(causal[index])
        causal[index, missing] = station_median[missing]
    if not np.isfinite(causal).all():
        raise AssertionError("causal fill produced non-finite values")
    return causal.astype(np.float32)


def fill_and_scale(bundle: DevelopmentBundle, validation_start: pd.Timestamp) -> dict[str, np.ndarray | str | int]:
    fit_end = int(bundle.time.get_loc(validation_start - pd.Timedelta(hours=1)))
    rate_filled = _fallback_fill(bundle.rate_observed, bundle.observed_mask, bundle.time, fit_end)
    states = np.empty_like(bundle.state_observed, dtype=np.float32)
    for component in range(3):
        states[..., component] = _fallback_fill(
            bundle.state_observed[..., component], bundle.observed_mask, bundle.time, fit_end
        )
    states = np.clip(states, 0.0, 1.0)
    state_sum = states.sum(axis=-1, keepdims=True)
    states = states / np.maximum(state_sum, 1e-8)
    train_nan = np.where(bundle.observed_mask[: fit_end + 1], bundle.rate_observed[: fit_end + 1], np.nan)
    mean = np.nanmean(train_nan, axis=0).astype(np.float32)
    scale = np.nanstd(train_nan, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    scaled = ((rate_filled - mean[None, :]) / scale[None, :]).astype(np.float32)
    fold_capacity = np.empty(len(bundle.stations), dtype=np.float32)
    for station in range(len(bundle.stations)):
        values = bundle.capacity_observed[: fit_end + 1, station]
        values = values[np.isfinite(values)]
        if not values.size:
            raise AssertionError("fold train-only capacity is empty")
        unique, counts = np.unique(values, return_counts=True)
        fold_capacity[station] = unique[np.flatnonzero(counts == counts.max())[0]]
    return {
        "fit_end_index": fit_end,
        "rate_filled": rate_filled,
        "rate_scaled": scaled,
        "state_filled": states.astype(np.float32),
        "scaler_mean": mean,
        "scaler_scale": scale,
        "fold_capacity": fold_capacity,
        "fill_hash": array_hash(rate_filled),
        "state_fill_hash": array_hash(states.astype("<f4")),
        "scaler_hash": array_hash(np.stack([mean, scale]).astype("<f4")),
        "fold_capacity_hash": array_hash(fold_capacity.astype("<f4")),
    }


def index_range(time: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.flatnonzero((time >= start) & (time <= end)).astype(np.int64)


def cell_contract(bundle: DevelopmentBundle, manifest_row: pd.Series, lookback: int = 168) -> dict[str, Any]:
    horizon = int(manifest_row.horizon)
    validation_start = pd.Timestamp(manifest_row.validation_start)
    validation_end = pd.Timestamp(manifest_row.validation_end)
    evaluation_start = pd.Timestamp(manifest_row.eval_start)
    evaluation_end = pd.Timestamp(manifest_row.eval_end)
    shared = fill_and_scale(bundle, validation_start)
    train_min = lookback - 1 + horizon
    train_max = int(bundle.time.get_loc(validation_start - pd.Timedelta(hours=1)))
    train_targets = np.arange(train_min, train_max + 1, dtype=np.int64)
    valid_targets = index_range(
        bundle.time, validation_start + pd.Timedelta(hours=horizon), validation_end
    )
    eval_targets = index_range(bundle.time, evaluation_start, evaluation_end)
    eval_origins = eval_targets - horizon
    if str(bundle.time[train_targets[0]]) != str(pd.Timestamp(manifest_row.train_start)):
        raise AssertionError("actual train target start/manifest drift")
    if str(bundle.time[train_targets[-1]]) != str(pd.Timestamp(manifest_row.train_end)):
        raise AssertionError("actual train target end/manifest drift")
    if str(bundle.time[valid_targets[0]]) != str(pd.Timestamp(manifest_row.validation_target_start)):
        raise AssertionError("actual validation target start/manifest drift")
    if eval_targets.size != int(manifest_row.eval_timestamps):
        raise AssertionError("evaluation target count drift")
    expected = {
        "forecast_origin_hash": array_hash(bundle.time[eval_origins].view("i8")),
        "target_timestamp_hash": array_hash(bundle.time[eval_targets].view("i8")),
        "station_order_hash": array_hash(bundle.stations),
        "mask_hash": array_hash(bundle.observed_mask.astype(np.uint8)),
    }
    for key, actual in expected.items():
        if str(manifest_row[key]) != actual:
            raise AssertionError(f"manifest {key} drift")
    identity = {
        **expected,
        "eval_mask_hash": array_hash(bundle.observed_mask[eval_targets].astype(np.uint8)),
        "eval_target_value_hash": array_hash(
            np.nan_to_num(bundle.rate_observed[eval_targets], nan=-9.0).astype("<f4")
        ),
        "fill_hash": shared["fill_hash"],
        "state_fill_hash": shared["state_fill_hash"],
        "scaler_hash": shared["scaler_hash"],
        "fold_capacity_hash": shared["fold_capacity_hash"],
    }
    identity["shared_contract_hash"] = json_hash(identity)
    for key in (
        "eval_target_value_hash",
        "eval_mask_hash",
        "fill_hash",
        "state_fill_hash",
        "scaler_hash",
        "fold_capacity_hash",
        "shared_contract_hash",
    ):
        if key in manifest_row.index and pd.notna(manifest_row[key]) and str(manifest_row[key]) != identity[key]:
            raise AssertionError(f"manifest {key} drift")
    return {
        **shared,
        "train_targets": train_targets,
        "validation_targets": valid_targets,
        "evaluation_targets": eval_targets,
        "evaluation_origins": eval_origins,
        "identity": identity,
    }


def load_model_config(root: Path) -> dict[str, Any]:
    path = root / "experiments/09_distill_v2/PARIS_TEACHER_MODEL_CONFIG.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("formal_target_access") is not False or payload.get("protected_target_access") is not False:
        raise RuntimeError("model config boundary drift")
    return payload


def implementation_hashes(root: Path) -> dict[str, str]:
    experiment = root / "experiments/09_distill_v2"
    audited = root.parent / "work/UrbanEV-reproduction/audited/code-transformer"
    paths = {
        "PARIS_TEACHER_MODEL_CONFIG.json": experiment / "PARIS_TEACHER_MODEL_CONFIG.json",
        "run_paris_teacher_cell.py": experiment / "scripts/run_paris_teacher_cell.py",
        "paris_teacher_common.py": experiment / "scripts/paris_teacher_common.py",
        "run_paris_teacher_queue.py": experiment / "scripts/run_paris_teacher_queue.py",
        "aggregate_paris_teacher.py": experiment / "scripts/aggregate_paris_teacher.py",
        "TimeXer.py": audited / "models/TimeXer.py",
        "Embed.py": audited / "layers/Embed.py",
        "SelfAttention_Family.py": audited / "layers/SelfAttention_Family.py",
        "masking.py": audited / "utils/masking.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def implementation_bundle_hash(root: Path) -> str:
    return json_hash(implementation_hashes(root))


def load_manifests(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = root / "experiments/09_distill_v2/outputs/teacher_qualification"
    folds = pd.read_csv(out / "DEVELOPMENT_FOLD_MANIFEST.csv")
    runs = pd.read_csv(out / "PARIS_TEACHER_RUN_MANIFEST.csv")
    if len(folds) != 16 or len(runs) != 32 or runs.fingerprint.nunique() != 32:
        raise AssertionError("frozen manifest cardinality drift")
    return folds, runs


def masked_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = np.asarray(prediction, np.float64)[mask]
    y = np.asarray(target, np.float64)[mask]
    if not p.size or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("empty/non-finite metric inputs")
    error = p - y
    absolute = np.abs(error)
    denominator = np.abs(p) + np.abs(y)
    smask = denominator > 1e-8
    target_sum = float(np.abs(y).sum())
    rae_denominator = float(np.abs(y - y.mean()).sum())
    return {
        "RMSE": float(np.sqrt(np.mean(error * error))),
        "MAE": float(absolute.mean()),
        "WAPE": float(absolute.sum() / target_sum) if target_sum else float("nan"),
        "sMAPE": float(np.mean(2.0 * absolute[smask] / denominator[smask])) if smask.any() else float("nan"),
        "RAE": float(absolute.sum() / rae_denominator) if rae_denominator else float("nan"),
    }
