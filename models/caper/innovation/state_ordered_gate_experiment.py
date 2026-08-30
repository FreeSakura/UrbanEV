"""Capacity model with a train-fitted, state-ordered phase expert.

The neural backbone sees only the recent 12-hour O/D/V window.  A second,
fully interpretable expert uses real 5-minute-derived active/unavailable states,
daily/weekly phase equilibria, and train-only retention coefficients.  The
ordered variant encodes the empirically stable rule that active charging is
released faster than unavailable-non-active occupancy at 3--12 hour horizons.
Matched reversed, tied, total-only, off-phase, trend-continuation, and padded
controls isolate the rule from extra parameters and sparse observations.
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from innovation.canonical import canonical_boundaries, canonical_target_indices
from innovation.deep_baselines import target_index_hash
from innovation.hybrid_expert_experiment import (
    Config as HybridConfig,
    HybridDataset,
    evaluate,
    train,
)
from innovation.multires_state import load_five_minute_states
from innovation.phase_mean_reversion_experiment import fit_phase_beta
from innovation.physical_constraints import load_physical_data
from innovation.physical_multitask import (
    PhysicalMultiTaskMLP,
    fit_physical_scales,
    set_deterministic,
)
from innovation.state_phase_relaxation_probe import fit_retention
from repro.metrics import audited_metrics, official_metrics


Variant = Literal[
    "ordered_state_gate",
    "free_fitted_state_gate",
    "reversed_state_gate",
    "tied_state_gate",
    "total_only_gate",
    "offphase_state_gate",
    "trend_state_gate",
    "padded_state_gate",
]

VARIANTS: tuple[Variant, ...] = (
    "ordered_state_gate",
    "free_fitted_state_gate",
    "reversed_state_gate",
    "tied_state_gate",
    "total_only_gate",
    "offphase_state_gate",
    "trend_state_gate",
    "padded_state_gate",
)


@dataclass(frozen=True)
class Config:
    fold: int
    horizon: int
    variant: Variant
    seed: int = 42
    history_length: int = 12
    common_history_budget: int = 168
    hidden1: int = 96
    hidden2: int = 48
    node_embedding_dim: int = 8
    gate_hidden: int = 15
    epochs: int = 100
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_delta: float = 1e-6
    gradient_clip: float = 1.0
    device: str = "cuda"
    causal_auxiliary_history: bool = False
    run_id: str | None = None
    attempt_id: int = 1

    def hybrid_config(self) -> HybridConfig:
        return HybridConfig(
            fold=self.fold,
            horizon=self.horizon,
            variant="linear_hybrid_aligned",
            seed=self.seed,
            history_length=self.history_length,
            common_history_budget=self.common_history_budget,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            node_embedding_dim=self.node_embedding_dim,
            gate_hidden=16,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            min_delta=self.min_delta,
            gradient_clip=self.gradient_clip,
            device=self.device,
            causal_auxiliary_history=self.causal_auxiliary_history,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )


class StatePhaseDataset(HybridDataset):
    def __init__(self, data, scales, state_hour, targets, config: Config) -> None:
        super().__init__(
            data, scales, state_hour, targets, config.hybrid_config()
        )
        self.state_variant = config.variant

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(item)
        target = int(self.targets[item])
        lags = (23, 167) if self.state_variant == "offphase_state_gate" else (24, 168)
        active_day = self.state[target - lags[0], :, 1]
        active_week = self.state[target - lags[1], :, 1]
        unavailable_day = self.state[target - lags[0], :, 2]
        unavailable_week = self.state[target - lags[1], :, 2]
        external = np.stack(
            [active_day, active_week, unavailable_day, unavailable_week], axis=-1
        )
        sample["external_experts"] = torch.from_numpy(
            np.ascontiguousarray(np.clip(external, 0.0, 1.0).astype(np.float32))
        )
        return sample


def phase_value(
    day: torch.Tensor,
    week: torch.Tensor,
    beta: float,
    trend: bool,
) -> torch.Tensor:
    if trend:
        return torch.clamp(day + beta * (day - week), 0.0, 1.0)
    return (1.0 - beta) * day + beta * week


def ordered_retention_pair(active: float, unavailable: float) -> tuple[float, float]:
    """Project two fitted retentions onto active <= unavailable."""

    return min(active, unavailable), max(active, unavailable)


class StateOrderedGateModel(nn.Module):
    CONTEXT_DIM = 14
    TARGET_PARAMETERS = 11_998

    def __init__(self, data, scales, config: Config, coefficients: dict) -> None:
        super().__init__()
        self.variant = config.variant
        self.coefficients = coefficients
        self.base = PhysicalMultiTaskMLP(
            variant="capacity_relaxation",
            history_length=config.history_length,
            input_channels=3,
            capacity=data.capacity,
            scales=scales,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            node_embedding_dim=config.node_embedding_dim,
            forecast_horizon=config.horizon,
        )
        self.gate = nn.Sequential(
            nn.Linear(self.CONTEXT_DIM, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        current = sum(parameter.numel() for parameter in self.parameters())
        if current > self.TARGET_PARAMETERS:
            raise AssertionError((current, self.TARGET_PARAMETERS))
        self.matched_padding = nn.Parameter(
            torch.zeros(self.TARGET_PARAMETERS - current)
        )

    def physical_expert(
        self,
        external: torch.Tensor,
        origin_active: torch.Tensor,
        origin_unavailable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_day, active_week = external[..., 0], external[..., 1]
        unavailable_day, unavailable_week = external[..., 2], external[..., 3]
        trend = self.variant == "trend_state_gate"
        phase_active = phase_value(
            active_day, active_week, self.coefficients["beta_active"], trend
        )
        phase_unavailable = phase_value(
            unavailable_day,
            unavailable_week,
            self.coefficients["beta_unavailable"],
            trend,
        )
        retention_active = self.coefficients["retention_active"]
        retention_unavailable = self.coefficients["retention_unavailable"]
        if self.variant == "free_fitted_state_gate":
            retention_active = self.coefficients["raw_retention_active"]
            retention_unavailable = self.coefficients["raw_retention_unavailable"]
        if self.variant == "reversed_state_gate":
            retention_active, retention_unavailable = (
                retention_unavailable, retention_active
            )
        elif self.variant == "tied_state_gate":
            tied = 0.5 * (retention_active + retention_unavailable)
            retention_active = retention_unavailable = tied
        if self.variant == "total_only_gate":
            total_day = active_day + unavailable_day
            total_week = active_week + unavailable_week
            phase_total = phase_value(
                total_day, total_week, self.coefficients["beta_total"], False
            )
            origin_total = origin_active + origin_unavailable
            physical = (
                self.coefficients["retention_total"] * origin_total
                + (1.0 - self.coefficients["retention_total"]) * phase_total
            )
            phase_active = phase_total
            phase_unavailable = torch.zeros_like(phase_total)
        else:
            active = (
                retention_active * origin_active
                + (1.0 - retention_active) * phase_active
            )
            unavailable = (
                retention_unavailable * origin_unavailable
                + (1.0 - retention_unavailable) * phase_unavailable
            )
            physical = active + unavailable
        return physical.clamp(0.0, 1.0), phase_active, phase_unavailable

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
        external_experts: torch.Tensor,
        origin_active: torch.Tensor,
        origin_unavailable: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = self.base(history, target_calendar)["occupancy_rate"]
        physical, phase_active, phase_unavailable = self.physical_expert(
            external_experts, origin_active, origin_unavailable
        )
        active_disagreement = torch.abs(
            external_experts[..., 0] - external_experts[..., 1]
        )
        unavailable_disagreement = torch.abs(
            external_experts[..., 2] - external_experts[..., 3]
        )
        if self.variant == "total_only_gate":
            origin_active_context = origin_active + origin_unavailable
            origin_unavailable_context = torch.zeros_like(origin_active)
            active_disagreement = torch.abs(
                external_experts[..., 0] + external_experts[..., 2]
                - external_experts[..., 1] - external_experts[..., 3]
            )
            unavailable_disagreement = torch.zeros_like(active_disagreement)
        else:
            origin_active_context = origin_active
            origin_unavailable_context = origin_unavailable
        calendar = target_calendar.unsqueeze(1).expand(-1, base.shape[1], -1)
        capacity = self.base.capacity_context.view(1, -1, 1).expand(
            base.shape[0], -1, -1
        )
        context = torch.cat(
            [
                base.unsqueeze(-1),
                physical.unsqueeze(-1),
                origin_active_context.unsqueeze(-1),
                origin_unavailable_context.unsqueeze(-1),
                phase_active.unsqueeze(-1),
                phase_unavailable.unsqueeze(-1),
                active_disagreement.unsqueeze(-1),
                unavailable_disagreement.unsqueeze(-1),
                (base - physical).unsqueeze(-1),
                calendar,
                capacity,
            ],
            dim=-1,
        )
        if context.shape[-1] != self.CONTEXT_DIM:
            raise AssertionError(context.shape)
        weight = torch.sigmoid(self.gate(context).squeeze(-1))
        if self.variant == "padded_state_gate":
            prediction = base + 0.0 * weight + 0.0 * self.matched_padding.sum()
        else:
            prediction = (
                (1.0 - weight) * base + weight * physical
                + 0.0 * self.matched_padding.sum()
            )
        experts = torch.stack([base, physical], dim=-1)
        weights = torch.stack([1.0 - weight, weight], dim=-1)
        return {
            "occupancy_rate": prediction,
            "base_rate": base,
            "experts": experts,
            "expert_weights": weights,
        }


def fit_coefficients(
    state: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    lags: tuple[int, int],
) -> dict[str, float]:
    active = state[..., 1]
    unavailable = state[..., 2]
    total = active + unavailable
    beta_active, raw_retention_active = fit_retention(
        active, targets, horizon, lags
    )
    beta_unavailable, raw_retention_unavailable = fit_retention(
        unavailable, targets, horizon, lags
    )
    retention_active, retention_unavailable = ordered_retention_pair(
        raw_retention_active, raw_retention_unavailable
    )
    beta_total, retention_total = fit_retention(total, targets, horizon, lags)
    return {
        "beta_active": beta_active,
        "beta_unavailable": beta_unavailable,
        "beta_total": beta_total,
        "retention_active": retention_active,
        "retention_unavailable": retention_unavailable,
        "raw_retention_active": raw_retention_active,
        "raw_retention_unavailable": raw_retention_unavailable,
        "retention_total": retention_total,
        "raw_active_below_unavailable": (
            raw_retention_active < raw_retention_unavailable
        ),
        "ordered_active_below_unavailable": retention_active <= retention_unavailable,
    }


def run(root: Path, source_root: Path, config: Config) -> Path:
    if config.variant not in VARIANTS:
        raise ValueError(config.variant)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_deterministic(config.seed)
    torch.set_num_threads(1)
    data = load_physical_data(root / "audited" / "data")
    bounds = canonical_boundaries(data.time, config.fold)
    scales = fit_physical_scales(data, bounds.train_end)
    fine = load_five_minute_states(root, source_root, data)
    state_hour = fine.state_rate[::12].astype(np.float32)
    targets = {
        split: canonical_target_indices(
            bounds, config.horizon, split, config.common_history_budget
        )
        for split in ("train", "valid")
    }
    lags = (23, 167) if config.variant == "offphase_state_gate" else (24, 168)
    coefficients = fit_coefficients(
        state_hour.astype(np.float64), targets["train"], config.horizon, lags
    )
    datasets = {
        split: StatePhaseDataset(data, scales, state_hour, target_index, config)
        for split, target_index in targets.items()
    }
    model = StateOrderedGateModel(data, scales, config, coefficients)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model, history, best_epoch, runtime = train(model, datasets, config)
    valid = evaluate(model, datasets["valid"], config)
    run_id = config.run_id or (
        f"state_ordered_gate_v1_{config.variant}_f{config.fold}_h{config.horizon}_"
        f"s{config.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_dir = root / "innovation" / "state_ordered_gate_runs" / run_id / (
        f"attempt_{config.attempt_id:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    mean_physical_weight = float(np.mean(valid["expert_weights"][..., 1]))
    for semantics, function in (("audited", audited_metrics), ("official", official_metrics)):
        metric_rows.append(
            {
                "run_id": run_id,
                "variant": config.variant,
                "fold": config.fold,
                "horizon": config.horizon,
                "seed": config.seed,
                "split": "validation",
                "metric_semantics": semantics,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "runtime_seconds": runtime,
                "mean_physical_weight": mean_physical_weight,
                **coefficients,
                **function(valid["prediction"], valid["target"]),
            }
        )
    history.to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(
        run_dir / "validation_predictions.npz", **valid,
        zone_ids=np.asarray(data.zone_ids),
    )
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "data_source": "real_UrbanEV_release_from_user_workspace",
                "source_root": str(source_root),
                "phase_lags": list(lags),
                "train_only_coefficients": coefficients,
                "formal_test_constructed": False,
                "formal_test_arrays_loaded": False,
                "dense_168_history_used": False,
                "sparse_state_anchor_count": 4,
                "validation_target_hash": target_index_hash(targets["valid"]),
                "state_cache_provenance": fine.provenance,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "success", "run_id": run_id}, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root", type=Path, default=Path("data/UrbanEV")
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id", type=int, default=1)
    args = parser.parse_args()
    config = Config(
        fold=args.fold,
        horizon=args.horizon,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
    )
    try:
        run(args.root.resolve(), args.source_root.resolve(), config)
    except Exception:
        run_id = args.run_id or "state_ordered_gate_failed"
        failed = args.root.resolve() / "innovation" / "state_ordered_gate_runs" / run_id / (
            f"attempt_{args.attempt_id:02d}"
        )
        failed.mkdir(parents=True, exist_ok=True)
        (failed / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        (failed / "status.json").write_text(
            json.dumps({"status": "failed", "run_id": run_id}, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
