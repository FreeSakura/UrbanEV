"""Interpretable stock-flow error-correction screen for UrbanEV.

The model keeps the twelve-hour O/D/V backbone and its bounded capacity
relaxation decoder.  A train-fitted daily/weekly equilibrium may only pull the
decoder equilibrium toward the observed phase equilibrium.  A second expert
uses the real five-minute-derived active/unavailable composition and the
empirical ordering that active charging relaxes no slower than unavailable
non-active occupancy.  A small linear gate combines the two forecasts.

All variants have the same parameter count.  Formal test targets are neither
constructed nor loaded during this validation-only screen.
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
from innovation.hybrid_expert_experiment import evaluate, train
from innovation.multires_state import load_five_minute_states
from innovation.physical_constraints import load_physical_data
from innovation.physical_multitask import (
    PhysicalMultiTaskMLP,
    fit_physical_scales,
    set_deterministic,
)
from innovation.state_ordered_gate_experiment import (
    Config as StateDatasetConfig,
    StatePhaseDataset,
    fit_coefficients,
    phase_value,
)
from repro.metrics import audited_metrics, official_metrics


Variant = Literal[
    "capacity_only",
    "phase_only",
    "ordered_stock_flow",
    "free_signed_stock_flow",
    "free_retention_stock_flow",
    "reversed_stock_flow",
    "tied_stock_flow",
    "total_stock_flow",
    "offphase_stock_flow",
    "trend_stock_flow",
    "free_signed_phase_only",
    "offphase_phase_only",
    "trend_phase_only",
    "matched_plain",
]

VARIANTS: tuple[Variant, ...] = (
    "capacity_only",
    "phase_only",
    "ordered_stock_flow",
    "free_signed_stock_flow",
    "free_retention_stock_flow",
    "reversed_stock_flow",
    "tied_stock_flow",
    "total_stock_flow",
    "offphase_stock_flow",
    "trend_stock_flow",
    "free_signed_phase_only",
    "offphase_phase_only",
    "trend_phase_only",
    "matched_plain",
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
    phase_hidden: int = 20
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

    def dataset_config(self) -> StateDatasetConfig:
        state_variant = (
            "offphase_state_gate"
            if self.variant in ("offphase_stock_flow", "offphase_phase_only")
            else "ordered_state_gate"
        )
        return StateDatasetConfig(
            fold=self.fold,
            horizon=self.horizon,
            variant=state_variant,
            seed=self.seed,
            history_length=self.history_length,
            common_history_budget=self.common_history_budget,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            node_embedding_dim=self.node_embedding_dim,
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


def ordered_state_projection(
    external: torch.Tensor,
    origin_active: torch.Tensor,
    origin_unavailable: torch.Tensor,
    coefficients: dict[str, float],
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a bounded state-flow forecast and its phase components."""

    active_day, active_week = external[..., 0], external[..., 1]
    unavailable_day, unavailable_week = external[..., 2], external[..., 3]
    trend = variant in ("trend_stock_flow", "trend_phase_only")
    phase_active = phase_value(
        active_day, active_week, coefficients["beta_active"], trend
    )
    phase_unavailable = phase_value(
        unavailable_day,
        unavailable_week,
        coefficients["beta_unavailable"],
        trend,
    )
    retention_active = coefficients["retention_active"]
    retention_unavailable = coefficients["retention_unavailable"]
    if variant == "free_retention_stock_flow":
        retention_active = coefficients["raw_retention_active"]
        retention_unavailable = coefficients["raw_retention_unavailable"]
    elif variant == "reversed_stock_flow":
        retention_active, retention_unavailable = (
            retention_unavailable,
            retention_active,
        )
    elif variant == "tied_stock_flow":
        tied = 0.5 * (retention_active + retention_unavailable)
        retention_active = retention_unavailable = tied
    if variant == "total_stock_flow":
        phase_total = phase_value(
            active_day + unavailable_day,
            active_week + unavailable_week,
            coefficients["beta_total"],
            False,
        )
        origin_total = origin_active + origin_unavailable
        prediction = (
            coefficients["retention_total"] * origin_total
            + (1.0 - coefficients["retention_total"]) * phase_total
        )
        return prediction.clamp(0.0, 1.0), phase_total, torch.zeros_like(phase_total)
    active = (
        retention_active * origin_active
        + (1.0 - retention_active) * phase_active
    )
    unavailable = (
        retention_unavailable * origin_unavailable
        + (1.0 - retention_unavailable) * phase_unavailable
    )
    return (
        (active + unavailable).clamp(0.0, 1.0),
        phase_active,
        phase_unavailable,
    )


class StockFlowErrorCorrectionModel(nn.Module):
    """Capacity partial adjustment plus an ordered state-flow expert."""

    PHASE_CONTEXT_DIM = 8
    GATE_CONTEXT_DIM = 8
    TARGET_PARAMETERS = 11_998

    def __init__(self, data, scales, config: Config, coefficients: dict) -> None:
        super().__init__()
        self.variant = config.variant
        self.horizon = float(config.horizon)
        self.coefficients = coefficients
        self.base = PhysicalMultiTaskMLP(
            variant=(
                "multivariate_occupancy"
                if config.variant == "matched_plain"
                else "capacity_relaxation"
            ),
            history_length=config.history_length,
            input_channels=3,
            capacity=data.capacity,
            scales=scales,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            node_embedding_dim=config.node_embedding_dim,
            forecast_horizon=config.horizon,
        )
        self.phase_head = nn.Sequential(
            nn.Linear(self.PHASE_CONTEXT_DIM, config.phase_hidden),
            nn.GELU(),
            nn.Linear(config.phase_hidden, 2),
        )
        self.gate = nn.Linear(self.GATE_CONTEXT_DIM, 1)
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        current = sum(parameter.numel() for parameter in self.parameters())
        if current > self.TARGET_PARAMETERS:
            raise AssertionError((current, self.TARGET_PARAMETERS))
        self.matched_padding = nn.Parameter(
            torch.zeros(self.TARGET_PARAMETERS - current)
        )

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
        external_experts: torch.Tensor,
        origin_active: torch.Tensor,
        origin_unavailable: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_output = self.base(history, target_calendar)
        base = base_output["occupancy_rate"]
        if self.variant == "matched_plain":
            prediction = base + 0.0 * self.phase_head[-1].weight.sum()
            prediction = prediction + 0.0 * self.gate.weight.sum()
            prediction = prediction + 0.0 * self.matched_padding.sum()
            return {
                "occupancy_rate": prediction,
                "base_rate": base,
                "experts": torch.stack([base, base], dim=-1),
                "expert_weights": torch.stack(
                    [torch.ones_like(base), torch.zeros_like(base)], dim=-1
                ),
                "equilibrium_weight": torch.zeros_like(base),
                "phase_active": torch.zeros_like(base),
                "phase_unavailable": torch.zeros_like(base),
            }
        equilibrium = base_output["equilibrium_rate"]
        decay = base_output["decay_rate"]
        origin_total = history[:, :, -1, 0].clamp(0.0, 1.0)
        total_day = external_experts[..., 0] + external_experts[..., 2]
        total_week = external_experts[..., 1] + external_experts[..., 3]
        beta_total = self.coefficients["beta_total"]
        trend_all = self.variant in ("trend_stock_flow", "trend_phase_only")
        phase_total = phase_value(total_day, total_week, beta_total, trend_all)
        capacity = self.base.capacity_context.view(1, -1).expand_as(base)
        phase_context = torch.stack(
            [
                base,
                equilibrium,
                torch.log1p(decay),
                origin_total,
                phase_total,
                torch.abs(total_day - total_week),
                origin_total - phase_total,
                capacity,
            ],
            dim=-1,
        )
        phase_delta = self.phase_head(phase_context)
        if self.variant in ("free_signed_stock_flow", "free_signed_phase_only"):
            corrected_equilibrium = torch.sigmoid(
                torch.logit(equilibrium.clamp(1e-6, 1.0 - 1e-6))
                + phase_delta[..., 0]
            )
            equilibrium_weight = torch.sigmoid(phase_delta[..., 0])
        else:
            equilibrium_weight = torch.sigmoid(phase_delta[..., 0] - 2.0)
            corrected_equilibrium = (
                (1.0 - equilibrium_weight) * equilibrium
                + equilibrium_weight * phase_total
            )
        corrected_decay = decay * torch.exp(phase_delta[..., 1].clamp(-3.0, 3.0))
        retention = torch.exp(-corrected_decay * self.horizon)
        phase_prediction = (
            retention * origin_total
            + (1.0 - retention) * corrected_equilibrium
        ).clamp(0.0, 1.0)
        physical, phase_active, phase_unavailable = ordered_state_projection(
            external_experts,
            origin_active,
            origin_unavailable,
            self.coefficients,
            self.variant,
        )
        gate_context = torch.stack(
            [
                phase_prediction,
                physical,
                torch.abs(phase_prediction - physical),
                origin_active,
                origin_unavailable,
                torch.abs(external_experts[..., 0] - external_experts[..., 1]),
                torch.abs(external_experts[..., 2] - external_experts[..., 3]),
                capacity,
            ],
            dim=-1,
        )
        gate_weight = torch.sigmoid(self.gate(gate_context).squeeze(-1))
        if self.variant == "capacity_only":
            prediction = base
        elif self.variant in (
            "phase_only",
            "free_signed_phase_only",
            "offphase_phase_only",
            "trend_phase_only",
        ):
            prediction = phase_prediction
        else:
            prediction = (
                (1.0 - gate_weight) * phase_prediction
                + gate_weight * physical
            )
        prediction = prediction + 0.0 * self.matched_padding.sum()
        return {
            "occupancy_rate": prediction,
            "base_rate": base,
            "experts": torch.stack([phase_prediction, physical], dim=-1),
            "expert_weights": torch.stack(
                [1.0 - gate_weight, gate_weight], dim=-1
            ),
            "equilibrium_weight": equilibrium_weight,
            "phase_active": phase_active,
            "phase_unavailable": phase_unavailable,
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
    lags = (23, 167) if config.variant == "offphase_stock_flow" else (24, 168)
    coefficients = fit_coefficients(
        state_hour.astype(np.float64), targets["train"], config.horizon, lags
    )
    dataset_config = config.dataset_config()
    datasets = {
        split: StatePhaseDataset(
            data, scales, state_hour, target_index, dataset_config
        )
        for split, target_index in targets.items()
    }
    model = StockFlowErrorCorrectionModel(data, scales, config, coefficients)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model, history, best_epoch, runtime = train(model, datasets, config)
    valid = evaluate(model, datasets["valid"], config)
    run_id = config.run_id or (
        f"stock_flow_error_correction_v1_{config.variant}_f{config.fold}_"
        f"h{config.horizon}_s{config.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_dir = root / "innovation" / "stock_flow_runs" / run_id / (
        f"attempt_{config.attempt_id:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    gate_weight = float(np.mean(valid["expert_weights"][..., 1]))
    for semantics, function in (
        ("audited", audited_metrics),
        ("official", official_metrics),
    ):
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
                "mean_state_expert_weight": gate_weight,
                **coefficients,
                **function(valid["prediction"], valid["target"]),
            }
        )
    history.to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(
        run_dir / "validation_predictions.npz",
        **valid,
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
        "--source-root", type=Path, default=Path(r"D:\UrbanEV\UrbanEV-main")
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--causal-auxiliary-history", action="store_true")
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
        causal_auxiliary_history=args.causal_auxiliary_history,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
    )
    try:
        run(args.root.resolve(), args.source_root.resolve(), config)
    except Exception:
        run_id = args.run_id or "stock_flow_error_correction_failed"
        failed = args.root.resolve() / "innovation" / "stock_flow_runs" / run_id / (
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
