"""Leakage-free real five-minute state pretraining for UrbanEV.

This module deliberately keeps the auxiliary five-minute task separate from
the formal hourly occupancy forecaster.  The auxiliary decoder is discarded
after pretraining, so every P0--P4 candidate has the same 12-hour input and the
same inference-time capacity-relaxation model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from innovation.physical_constraints import PhysicalData
from innovation.physical_multitask import (
    PhysicalMultiTaskMLP,
    PhysicalScales,
    PhysicalSequenceDataset,
)
from innovation.three_state_ctmc import (
    generator_from_parameters,
    reconstruct_iau_state,
)


PretrainingVariant = Literal[
    "random_init",
    "free_simplex",
    "three_state_ctmc",
    "misaligned_three_state_ctmc",
    "two_state_ctmc",
]


@dataclass(frozen=True)
class FiveMinuteStateData:
    state_rate: np.ndarray
    consistent_mask: np.ndarray
    time: pd.DatetimeIndex
    zone_ids: tuple[str, ...]
    provenance: dict[str, object]


@dataclass(frozen=True)
class PretrainingAnchorSplit:
    all_anchors: np.ndarray
    train_anchors: np.ndarray
    valid_anchors: np.ndarray
    train_end: int
    split_position: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.columns = frame.columns.astype(str)
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite five-minute values in {path}")
    if np.any(values < 0):
        raise ValueError(f"negative five-minute values in {path}")
    return frame


def _locked_sources(
    project_root: Path, source_root: Path
) -> tuple[dict[str, object], dict[str, Path]]:
    lock_path = project_root / "innovation" / "THREE_STATE_DATA_PROVENANCE.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files: dict[str, Path] = {}
    for name, record in lock["five_minute_files"].items():
        path = source_root / str(record["relative_path"])
        if not path.is_file():
            raise RuntimeError(f"locked real UrbanEV source is missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != int(record["bytes"]) or actual_hash != str(record["sha256"]).upper():
            raise RuntimeError(
                f"five-minute provenance mismatch for {path}; "
                f"bytes={actual_size}, sha256={actual_hash}"
            )
        files[name] = path
    return lock, files


def build_five_minute_state_cache(
    project_root: Path,
    source_root: Path,
    hourly_data: PhysicalData,
    cache_path: Path,
    *,
    tolerance: float = 1e-9,
) -> FiveMinuteStateData:
    """Build a derived cache only after verifying the immutable real sources."""

    lock, files = _locked_sources(project_root, source_root)
    occupancy = _read_matrix(files["occupancy.csv"])
    duration = _read_matrix(files["duration.csv"])
    expected_shape = tuple(int(value) for value in lock["expected_shape"])
    if occupancy.shape != expected_shape or duration.shape != expected_shape:
        raise RuntimeError(
            f"five-minute shape mismatch: {occupancy.shape}, {duration.shape}, "
            f"expected {expected_shape}"
        )
    if not occupancy.index.equals(duration.index):
        raise RuntimeError("five-minute occupancy and duration timestamps differ")
    if not occupancy.columns.equals(duration.columns):
        raise RuntimeError("five-minute occupancy and duration node order differs")
    if not occupancy.index[::12].equals(hourly_data.time):
        raise RuntimeError("five-minute timestamps do not align to formal hourly timestamps")
    if tuple(occupancy.columns) != hourly_data.zone_ids:
        raise RuntimeError("five-minute nodes do not match formal hourly node order")

    reconstruction = reconstruct_iau_state(
        occupancy.to_numpy(copy=False),
        duration.to_numpy(copy=False),
        hourly_data.capacity,
        tolerance=tolerance,
    )
    state = reconstruction.state_rate.astype(np.float32)
    consistent = reconstruction.consistent_mask.astype(np.bool_)
    provenance = {
        "source_kind": "real_UrbanEV_release_from_user_workspace",
        "source_root": str(source_root.resolve()),
        "lock_path": str(
            (project_root / "innovation" / "THREE_STATE_DATA_PROVENANCE.json").resolve()
        ),
        "source_sha256": {
            name: str(lock["five_minute_files"][name]["sha256"]).upper()
            for name in ("occupancy.csv", "duration.csv", "volume.csv")
        },
        "shape": [int(value) for value in state.shape],
        "consistent_cells": int(np.count_nonzero(consistent)),
        "inconsistent_cells": int(np.count_nonzero(~consistent)),
        "inconsistent_fraction": float(np.mean(~consistent)),
        "max_projection_count": float(reconstruction.projection_magnitude.max()),
        "state_order": ["idle_available", "active_electricity", "unavailable_non_active"],
        "active_count_definition": "12 * released_duration_5min_hours",
        "cache_is_derived_not_simulated": True,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        state_rate=state,
        consistent_mask=consistent.astype(np.uint8),
        time_ns=occupancy.index.asi8,
        zone_ids=np.asarray(occupancy.columns, dtype=str),
        provenance_json=np.asarray(json.dumps(provenance, ensure_ascii=False)),
    )
    (cache_path.with_suffix(cache_path.suffix + ".provenance.json")).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return FiveMinuteStateData(
        state_rate=state,
        consistent_mask=consistent,
        time=occupancy.index,
        zone_ids=tuple(occupancy.columns),
        provenance={**provenance, "cache_sha256": _sha256(cache_path)},
    )


def load_five_minute_states(
    project_root: Path,
    source_root: Path,
    hourly_data: PhysicalData,
    *,
    cache_path: Path | None = None,
) -> FiveMinuteStateData:
    """Load a verified derived cache or build it from the real release."""

    if cache_path is None:
        cache_path = project_root / "innovation" / "cache" / "five_minute_iau_v1.npz"
    lock, _ = _locked_sources(project_root, source_root)
    expected_hashes = {
        name: str(lock["five_minute_files"][name]["sha256"]).upper()
        for name in ("occupancy.csv", "duration.csv", "volume.csv")
    }
    if not cache_path.exists():
        return build_five_minute_state_cache(
            project_root, source_root, hourly_data, cache_path
        )

    with np.load(cache_path, allow_pickle=False) as payload:
        state = payload["state_rate"].astype(np.float32, copy=False)
        consistent = payload["consistent_mask"].astype(np.bool_, copy=False)
        time = pd.DatetimeIndex(pd.to_datetime(payload["time_ns"].astype(np.int64)))
        zone_ids = tuple(payload["zone_ids"].astype(str).tolist())
        provenance = json.loads(str(payload["provenance_json"].item()))
    expected_shape = (*tuple(int(value) for value in lock["expected_shape"]), 3)
    if state.shape != expected_shape or consistent.shape != expected_shape[:2]:
        raise RuntimeError("derived five-minute cache has an invalid shape")
    if provenance.get("source_sha256") != expected_hashes:
        raise RuntimeError("derived cache does not refer to the currently locked sources")
    if not time[::12].equals(hourly_data.time) or zone_ids != hourly_data.zone_ids:
        raise RuntimeError("derived cache does not align to the formal hourly dataset")
    simplex_error = float(np.max(np.abs(state.sum(axis=-1) - 1.0)))
    if simplex_error > 2e-6 or np.min(state) < -1e-7:
        raise RuntimeError(f"invalid cached I/A/U simplex; max error={simplex_error}")
    return FiveMinuteStateData(
        state_rate=state,
        consistent_mask=consistent,
        time=time,
        zone_ids=zone_ids,
        provenance={**provenance, "cache_sha256": _sha256(cache_path)},
    )


def make_pretraining_anchor_split(
    history_length: int,
    train_end: int,
    *,
    valid_fraction: float = 0.1,
) -> PretrainingAnchorSplit:
    """Construct train-only anchors whose full next-hour path stays in train."""

    if history_length <= 0 or train_end <= history_length + 1:
        raise ValueError("training fold is too short for five-minute pretraining")
    if not 0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must lie in (0, 0.5)")
    # Anchor a uses hourly [a-L+1, ..., a].  The 12 fine targets are
    # 12*a+1 ... 12*a+12, so a must be <= train_end-2.
    anchors = np.arange(history_length - 1, train_end - 1, dtype=np.int64)
    split = int(math.floor((1.0 - valid_fraction) * len(anchors)))
    split = min(max(split, 1), len(anchors) - 1)
    train = anchors[:split]
    valid = anchors[split:]
    if int(anchors[-1]) + 1 >= train_end:
        raise AssertionError("five-minute pretraining target crosses the train boundary")
    if np.intersect1d(train, valid).size:
        raise AssertionError("pretraining train and internal validation anchors overlap")
    return PretrainingAnchorSplit(
        all_anchors=anchors,
        train_anchors=train,
        valid_anchors=valid,
        train_end=int(train_end),
        split_position=split,
    )


class MultiResolutionStateDataset(Dataset):
    """Hourly O/D/V histories paired with real 5-minute next-state paths."""

    def __init__(
        self,
        hourly_data: PhysicalData,
        scales: PhysicalScales,
        fine_data: FiveMinuteStateData,
        anchors: np.ndarray,
        *,
        history_length: int = 12,
        misalignment_steps: int = 0,
    ) -> None:
        self.anchors = np.asarray(anchors, dtype=np.int64)
        if self.anchors.ndim != 1 or self.anchors.size == 0:
            raise ValueError("anchors must be a non-empty one-dimensional array")
        if np.any(np.diff(self.anchors) <= 0):
            raise ValueError("anchors must be strictly increasing")
        if misalignment_steps < 0 or misalignment_steps >= 12:
            raise ValueError("misalignment_steps must lie in 0..11")
        self.misalignment_steps = int(misalignment_steps)
        self.base = PhysicalSequenceDataset(
            hourly_data,
            scales,
            targets=self.anchors + 1,
            horizon=1,
            history_length=history_length,
            include_auxiliary_history=True,
        )
        self.fine = fine_data
        self.history_length = int(history_length)
        if not fine_data.time[::12].equals(hourly_data.time):
            raise RuntimeError("fine and hourly time axes are not synchronized")

    def __len__(self) -> int:
        return int(self.anchors.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        base = self.base[item]
        anchor = int(self.anchors[item])
        five_origin = 12 * anchor
        five_indices = np.arange(five_origin + 1, five_origin + 13, dtype=np.int64)
        if int(five_indices[-1]) != 12 * (anchor + 1):
            raise AssertionError("five-minute target path does not end at anchor+1 hour")
        trajectory = np.ascontiguousarray(
            self.fine.state_rate[five_indices].transpose(1, 0, 2)
        )
        mask = np.ascontiguousarray(
            self.fine.consistent_mask[five_indices].transpose(1, 0)
        )
        if self.misalignment_steps:
            trajectory = np.roll(
                trajectory, shift=-self.misalignment_steps, axis=1
            ).copy()
            mask = np.roll(mask, shift=-self.misalignment_steps, axis=1).copy()
        initial_consistent = np.ascontiguousarray(
            self.fine.consistent_mask[five_origin]
        )
        mask = mask & initial_consistent[:, np.newaxis]
        base.update(
            {
                "initial_state": torch.from_numpy(
                    np.ascontiguousarray(self.fine.state_rate[five_origin])
                ),
                "fine_trajectory": torch.from_numpy(trajectory),
                "fine_mask": torch.from_numpy(mask),
                "anchor_index": torch.tensor(anchor, dtype=torch.long),
                "five_target_start": torch.tensor(int(five_indices[0]), dtype=torch.long),
                "five_target_end": torch.tensor(int(five_indices[-1]), dtype=torch.long),
            }
        )
        return base


def encode_capacity_context(
    model: PhysicalMultiTaskMLP,
    history: torch.Tensor,
    target_calendar: torch.Tensor,
) -> torch.Tensor:
    """Return the exact shared latent used by the capacity-relaxation head."""

    if model.variant != "capacity_relaxation":
        raise ValueError("state pretraining requires the capacity_relaxation backbone")
    batch, n_nodes, _, _ = history.shape
    flattened = history.flatten(start_dim=2)
    node = model.node_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
    capacity = model.capacity_context.view(1, n_nodes, 1).expand(batch, -1, -1)
    calendar = target_calendar.unsqueeze(1).expand(-1, n_nodes, -1)
    return model.encoder(torch.cat([flattened, node, capacity, calendar], dim=-1))


class FineStatePretrainer(nn.Module):
    """Equal-access auxiliary transition head attached to the shared encoder."""

    def __init__(
        self,
        backbone: PhysicalMultiTaskMLP,
        variant: PretrainingVariant,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if variant == "random_init":
            raise ValueError("random_init has no auxiliary pretrainer")
        self.backbone = backbone
        self.variant = variant
        output_dim = 2 if variant == "two_state_ctmc" else 6
        self.transition_head = nn.Linear(hidden_dim, output_dim)
        if variant in {"three_state_ctmc", "misaligned_three_state_ctmc"}:
            rate_bias = math.log(math.expm1(0.1))
            with torch.no_grad():
                self.transition_head.bias[:4].fill_(rate_bias)
                self.transition_head.bias[4:].zero_()
        elif variant == "free_simplex":
            nn.init.constant_(self.transition_head.bias, -2.5)
        elif variant == "two_state_ctmc":
            rate_bias = math.log(math.expm1(0.1))
            nn.init.constant_(self.transition_head.bias, rate_bias)

    @property
    def auxiliary_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.transition_head.parameters())

    def _free_transition(self, raw: torch.Tensor) -> torch.Tensor:
        pair = raw.unflatten(-1, (3, 2))
        zero = torch.zeros_like(pair[..., 0, 0])
        row_i = torch.stack([zero, pair[..., 0, 0], pair[..., 0, 1]], dim=-1)
        row_a = torch.stack([pair[..., 1, 0], zero, pair[..., 1, 1]], dim=-1)
        row_u = torch.stack([pair[..., 2, 0], pair[..., 2, 1], zero], dim=-1)
        return torch.softmax(torch.stack([row_i, row_a, row_u], dim=-2), dim=-1)

    def _three_state_transition(self, raw: torch.Tensor) -> torch.Tensor:
        rates = F.softplus(raw[..., :4]) + 1e-6
        split = torch.sigmoid(raw[..., 4:5])
        clock = 0.5 + torch.sigmoid(raw[..., 5:6])
        parameters = torch.cat([rates * clock, split], dim=-1)
        generator = generator_from_parameters(parameters)
        return torch.matrix_exp(generator / 12.0)

    def _roll_three_state(
        self, initial: torch.Tensor, transition: torch.Tensor
    ) -> torch.Tensor:
        current = initial
        steps: list[torch.Tensor] = []
        for _ in range(12):
            current = torch.matmul(current.unsqueeze(-2), transition).squeeze(-2)
            steps.append(current)
        return torch.stack(steps, dim=-2)

    def _roll_two_state(
        self, initial: torch.Tensor, raw: torch.Tensor
    ) -> torch.Tensor:
        arrival = F.softplus(raw[..., 0]) + 1e-6
        departure = F.softplus(raw[..., 1]) + 1e-6
        total = arrival + departure
        equilibrium = arrival / total
        retention = torch.exp(-total / 12.0)
        occupied = initial[..., 1] + initial[..., 2]
        steps: list[torch.Tensor] = []
        for _ in range(12):
            occupied = retention * occupied + (1.0 - retention) * equilibrium
            steps.append(torch.stack([1.0 - occupied, occupied], dim=-1))
        return torch.stack(steps, dim=-2)

    def forward(
        self, history: torch.Tensor, target_calendar: torch.Tensor, initial: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        latent = encode_capacity_context(self.backbone, history, target_calendar)
        raw = self.transition_head(latent)
        if self.variant == "free_simplex":
            transition = self._free_transition(raw)
            trajectory = self._roll_three_state(initial, transition)
            return {"trajectory": trajectory, "transition": transition, "raw": raw}
        if self.variant in {"three_state_ctmc", "misaligned_three_state_ctmc"}:
            transition = self._three_state_transition(raw)
            trajectory = self._roll_three_state(initial, transition)
            return {"trajectory": trajectory, "transition": transition, "raw": raw}
        if self.variant == "two_state_ctmc":
            trajectory = self._roll_two_state(initial, raw)
            return {"trajectory_two_state": trajectory, "raw": raw}
        raise ValueError(f"unknown pretraining variant: {self.variant}")


def fine_state_objective(
    output: dict[str, torch.Tensor],
    target_three_state: torch.Tensor,
    mask: torch.Tensor,
    variant: PretrainingVariant,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked Brier loss plus common occupancy error for all variants."""

    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not torch.any(mask):
        raise RuntimeError("five-minute batch has no consistent supervision cells")
    target_occupancy = target_three_state[..., 1] + target_three_state[..., 2]
    if variant == "two_state_ctmc":
        prediction_two = output["trajectory_two_state"]
        predicted_occupancy = prediction_two[..., 1]
        state_loss = torch.mean(
            torch.sum(
                (prediction_two - torch.stack([1.0 - target_occupancy, target_occupancy], dim=-1))
                ** 2,
                dim=-1,
            )[mask]
        )
    else:
        prediction = output["trajectory"]
        predicted_occupancy = prediction[..., 1] + prediction[..., 2]
        state_loss = torch.mean(torch.sum((prediction - target_three_state) ** 2, dim=-1)[mask])
    occupancy_mse = torch.mean(((predicted_occupancy - target_occupancy) ** 2)[mask])
    return state_loss, {
        "state_brier": state_loss,
        "occupancy_mse": occupancy_mse,
        "supervised_fraction": mask.float().mean(),
    }


def aligned_fine_state_metrics(
    output: dict[str, torch.Tensor],
    target_three_state: torch.Tensor,
    mask: torch.Tensor,
    variant: PretrainingVariant,
) -> dict[str, float]:
    """Return directly comparable aligned reconstruction metrics."""

    mask = mask.bool()
    target_occ = target_three_state[..., 1] + target_three_state[..., 2]
    if variant == "two_state_ctmc":
        prediction_occ = output["trajectory_two_state"][..., 1]
        state_brier = float("nan")
        active_rmse = float("nan")
    else:
        prediction = output["trajectory"]
        prediction_occ = prediction[..., 1] + prediction[..., 2]
        state_brier = float(
            torch.mean(torch.sum((prediction - target_three_state) ** 2, dim=-1)[mask])
        )
        active_rmse = float(
            torch.sqrt(torch.mean(((prediction[..., 1] - target_three_state[..., 1]) ** 2)[mask]))
        )
    occupancy_rmse = float(
        torch.sqrt(torch.mean(((prediction_occ - target_occ) ** 2)[mask]))
    )
    return {
        "state_brier": state_brier,
        "occupancy_rmse": occupancy_rmse,
        "active_rmse": active_rmse,
        "supervised_cells": int(torch.count_nonzero(mask)),
    }

