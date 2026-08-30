"""Capacity dynamics with a train-identified convex phase equilibrium.

The rule is deliberately small and falsifiable.  Training observations fit a
single beta in [0, 1] such that the recent daily phase is shrunk toward the
same-weekday phase.  A tiny gate may then move the neural equilibrium toward
that convex phase equilibrium, never beyond it.  Opposite, off-phase, free-q,
and padded variants isolate the claimed direction from extra inputs/parameters.
"""

from __future__ import annotations

import argparse
import json
import platform
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
from innovation.physical_constraints import PhysicalData, load_physical_data
from innovation.physical_multitask import (
    PhysicalMultiTaskMLP,
    fit_physical_scales,
    set_deterministic,
)
from repro.metrics import audited_metrics, official_metrics


Variant = Literal[
    "phase_reversion_padded",
    "phase_reversion_convex",
    "phase_reversion_opposite",
    "phase_reversion_offphase",
    "phase_reversion_free_q",
]

VARIANTS: tuple[Variant, ...] = (
    "phase_reversion_padded",
    "phase_reversion_convex",
    "phase_reversion_opposite",
    "phase_reversion_offphase",
    "phase_reversion_free_q",
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
    gate_hidden: int = 8
    epochs: int = 100
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_delta: float = 1e-6
    gradient_clip: float = 1.0
    device: str = "cuda"
    run_id: str | None = None
    attempt_id: int = 1

    def hybrid_config(self) -> HybridConfig:
        hybrid_variant = (
            "nonlinear_hybrid_offphase"
            if self.variant == "phase_reversion_offphase"
            else "linear_hybrid_aligned"
        )
        return HybridConfig(
            fold=self.fold,
            horizon=self.horizon,
            variant=hybrid_variant,
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
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )


def fit_phase_beta(
    rate: np.ndarray, targets: np.ndarray, lags: tuple[int, int]
) -> tuple[float, float]:
    """Fit beta=-alpha on formal training labels only."""

    day = rate[targets - lags[0]].astype(np.float64)
    week = rate[targets - lags[1]].astype(np.float64)
    target = rate[targets].astype(np.float64)
    difference = (day - week).reshape(-1)
    residual = (target - day).reshape(-1)
    denominator = float(difference @ difference)
    alpha = float(difference @ residual / denominator) if denominator > 0 else 0.0
    return float(np.clip(-alpha, 0.0, 1.0)), alpha


class PhaseMeanReversionModel(nn.Module):
    CONTEXT_DIM = 8

    def __init__(
        self,
        data: PhysicalData,
        scales,
        config: Config,
        phase_beta: float,
    ) -> None:
        super().__init__()
        self.variant = config.variant
        self.horizon = float(config.horizon)
        self.phase_beta = float(phase_beta)
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
        nn.init.constant_(
            self.gate[-1].bias,
            0.0 if config.variant == "phase_reversion_free_q" else -2.0,
        )

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
        external_experts: torch.Tensor,
        origin_active: torch.Tensor,
        origin_unavailable: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del origin_active, origin_unavailable
        base_output = self.base(history, target_calendar)
        base = base_output["occupancy_rate"]
        equilibrium = base_output["equilibrium_rate"]
        decay = base_output["decay_rate"]
        origin = history[:, :, -1, 0].clamp(0.0, 1.0)
        day = external_experts[..., 0]
        week = external_experts[..., 1]
        phase = (1.0 - self.phase_beta) * day + self.phase_beta * week
        if self.variant == "phase_reversion_opposite":
            phase = torch.clamp(day + self.phase_beta * (day - week), 0.0, 1.0)
        context = torch.stack(
            [
                origin,
                base,
                equilibrium,
                day,
                week,
                torch.abs(day - week),
                origin - phase,
                self.base.capacity_context.view(1, -1).expand_as(origin),
            ],
            dim=-1,
        )
        raw = self.gate(context).squeeze(-1)
        if self.variant == "phase_reversion_padded":
            prediction = base + 0.0 * raw
        else:
            if self.variant == "phase_reversion_free_q":
                corrected_equilibrium = torch.sigmoid(
                    torch.logit(equilibrium.clamp(1e-6, 1.0 - 1e-6)) + raw
                )
            else:
                eta = torch.sigmoid(raw)
                corrected_equilibrium = (1.0 - eta) * equilibrium + eta * phase
            retention = torch.exp(-decay * self.horizon)
            prediction = retention * origin + (1.0 - retention) * corrected_equilibrium
        experts = torch.stack([base, day, week, phase], dim=-1)
        return {
            "occupancy_rate": prediction,
            "base_rate": base,
            "experts": experts,
            "expert_weights": torch.full_like(experts, float("nan")),
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
    lags = (23, 167) if config.variant == "phase_reversion_offphase" else (24, 168)
    rate = data.occupancy_count / data.capacity[None, :]
    phase_beta, phase_alpha_raw = fit_phase_beta(rate, targets["train"], lags)
    datasets = {
        split: HybridDataset(
            data, scales, state_hour, target_index, config.hybrid_config()
        )
        for split, target_index in targets.items()
    }
    model = PhaseMeanReversionModel(data, scales, config, phase_beta)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model, history, best_epoch, runtime = train(model, datasets, config)
    valid = evaluate(model, datasets["valid"], config)
    run_id = config.run_id or (
        f"phase_mean_reversion_v1_{config.variant}_f{config.fold}_h{config.horizon}_"
        f"s{config.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_dir = root / "innovation" / "phase_mean_reversion_runs" / run_id / (
        f"attempt_{config.attempt_id:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metric_rows = []
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
                "phase_beta_train_only": phase_beta,
                "phase_alpha_raw_train_only": phase_alpha_raw,
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
                "phase_beta_train_only": phase_beta,
                "phase_alpha_raw_train_only": phase_alpha_raw,
                "phase_beta_uses_formal_validation": False,
                "test_dataset_constructed": False,
                "test_arrays_loaded": False,
                "dense_168_history_used": False,
                "sparse_anchor_count": 2,
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
    parser.add_argument("--source-root", type=Path, default=Path("data/UrbanEV"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default=None)
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
    run(args.root.resolve(), args.source_root.resolve(), config)


if __name__ == "__main__":
    main()
