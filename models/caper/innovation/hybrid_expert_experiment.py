"""Validation-only hybrid expert gate for interpretable UrbanEV correction.

The gate combines five bounded experts: the 12-hour capacity-dynamics model,
two total-occupancy periodic anchors, and two state-selective anchors that keep
origin unavailable occupancy while updating active charging by exact phase.
All evaluation values are real released UrbanEV observations; no test split is
constructed during this screen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from innovation.canonical import canonical_boundaries, canonical_target_indices
from innovation.deep_baselines import target_index_hash
from innovation.multires_state import load_five_minute_states
from innovation.physical_constraints import PhysicalData, load_physical_data
from innovation.physical_multitask import (
    PhysicalMultiTaskMLP,
    PhysicalSequenceDataset,
    fit_physical_scales,
    set_deterministic,
)
from repro.metrics import audited_metrics, official_metrics


Variant = Literal[
    "linear_hybrid_aligned",
    "nonlinear_hybrid_aligned",
    "nonlinear_total_with_state_context",
    "nonlinear_hybrid_wrong_state",
    "nonlinear_hybrid_offphase",
    "nonlinear_hybrid_free_residual",
    "linear_hybrid_padded",
    "reliability_hybrid_aligned",
    "reliability_hybrid_inverse",
    "reliability_hybrid_offphase",
    "reliability_hybrid_wrong_state",
    "reliability_total_only",
]

VARIANTS: tuple[Variant, ...] = (
    "linear_hybrid_aligned",
    "nonlinear_hybrid_aligned",
    "nonlinear_total_with_state_context",
    "nonlinear_hybrid_wrong_state",
    "nonlinear_hybrid_offphase",
    "nonlinear_hybrid_free_residual",
    "linear_hybrid_padded",
    "reliability_hybrid_aligned",
    "reliability_hybrid_inverse",
    "reliability_hybrid_offphase",
    "reliability_hybrid_wrong_state",
    "reliability_total_only",
)

RELIABILITY_VARIANTS: tuple[Variant, ...] = (
    "linear_hybrid_padded",
    "reliability_hybrid_aligned",
    "reliability_hybrid_inverse",
    "reliability_hybrid_offphase",
    "reliability_hybrid_wrong_state",
    "reliability_total_only",
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
    gate_hidden: int = 16
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


class HybridDataset(Dataset):
    def __init__(
        self,
        data: PhysicalData,
        scales,
        state_hour: np.ndarray,
        targets: np.ndarray,
        config: Config,
    ) -> None:
        self.base = PhysicalSequenceDataset(
            data,
            scales,
            targets,
            config.horizon,
            config.history_length,
            include_auxiliary_history=True,
            causal_auxiliary_history=config.causal_auxiliary_history,
        )
        self.rate = (data.occupancy_count / data.capacity[None, :]).astype(np.float32)
        self.state = state_hour.astype(np.float32, copy=False)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.horizon = config.horizon
        self.variant = config.variant
        if self.state.shape[:2] != self.rate.shape:
            raise ValueError("Hourly I/A/U states do not align to hourly occupancy")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.base[item]
        target = int(self.targets[item])
        origin = target - self.horizon
        lags = (23, 167) if self.variant in (
            "nonlinear_hybrid_offphase", "reliability_hybrid_offphase"
        ) else (24, 168)
        if target - max(lags) < 0 or target - min(lags) > origin:
            raise AssertionError("Hybrid expert crosses the forecast origin")
        total = [self.rate[target - lag] for lag in lags]
        active_origin = self.state[origin, :, 1]
        unavailable_origin = self.state[origin, :, 2]
        if self.variant in (
            "nonlinear_hybrid_wrong_state", "reliability_hybrid_wrong_state"
        ):
            state_experts = [
                active_origin + self.state[target - lag, :, 2] for lag in lags
            ]
        elif self.variant in (
            "nonlinear_total_with_state_context", "reliability_total_only"
        ):
            state_experts = total
        else:
            state_experts = [
                unavailable_origin + self.state[target - lag, :, 1] for lag in lags
            ]
        external = np.stack(
            [total[0], total[1], state_experts[0], state_experts[1]], axis=-1
        )
        sample["external_experts"] = torch.from_numpy(
            np.ascontiguousarray(np.clip(external, 0.0, 1.0).astype(np.float32))
        )
        sample["origin_active"] = torch.from_numpy(
            np.ascontiguousarray(active_origin.astype(np.float32))
        )
        sample["origin_unavailable"] = torch.from_numpy(
            np.ascontiguousarray(unavailable_origin.astype(np.float32))
        )
        return sample


class HybridExpertModel(nn.Module):
    CONTEXT_DIM = 17

    def __init__(self, data: PhysicalData, scales, config: Config) -> None:
        super().__init__()
        self.variant = config.variant
        self.forecast_horizon = config.horizon
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
        if config.variant in (
            "linear_hybrid_aligned", "linear_hybrid_padded",
            "reliability_hybrid_aligned", "reliability_hybrid_inverse",
            "reliability_hybrid_offphase", "reliability_hybrid_wrong_state",
            "reliability_total_only",
        ):
            self.gate = nn.Linear(self.CONTEXT_DIM, 5)
            final = self.gate
        else:
            self.gate = nn.Sequential(
                nn.Linear(self.CONTEXT_DIM, config.gate_hidden),
                nn.GELU(),
                nn.Linear(config.gate_hidden, 5),
            )
            final = self.gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        nn.init.constant_(final.bias[0], 2.0)
        if config.variant in RELIABILITY_VARIANTS:
            # Two matched parameters control how total/state anchor disagreement
            # changes external-expert trust.  Softplus makes the intended
            # reliability penalty monotone and non-negative.
            self.reliability_raw = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
        external_experts: torch.Tensor,
        origin_active: torch.Tensor,
        origin_unavailable: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = self.base(history, target_calendar)["occupancy_rate"]
        experts = torch.cat([base.unsqueeze(-1), external_experts], dim=-1)
        origin = history[:, :, -1, 0].clamp(0.0, 1.0)
        delta1 = origin - history[:, :, -2, 0]
        delta3 = origin - history[:, :, -4, 0]
        total_disagreement = torch.abs(external_experts[..., 0] - external_experts[..., 1])
        state_disagreement = torch.abs(external_experts[..., 2] - external_experts[..., 3])
        base_minus_total = base - 0.5 * (
            external_experts[..., 0] + external_experts[..., 1]
        )
        calendar = target_calendar.unsqueeze(1).expand(-1, origin.shape[1], -1)
        capacity = self.base.capacity_context.view(1, -1, 1).expand(
            origin.shape[0], -1, -1
        )
        context = torch.cat(
            [
                experts,
                origin_active.unsqueeze(-1),
                origin_unavailable.unsqueeze(-1),
                delta1.unsqueeze(-1),
                delta3.unsqueeze(-1),
                total_disagreement.unsqueeze(-1),
                state_disagreement.unsqueeze(-1),
                base_minus_total.unsqueeze(-1),
                calendar,
                capacity,
            ],
            dim=-1,
        )
        if context.shape[-1] != self.CONTEXT_DIM:
            raise AssertionError("unexpected hybrid-gate context width")
        raw = self.gate(context)
        if self.variant in RELIABILITY_VARIANTS and self.variant != "linear_hybrid_padded":
            strength = torch.nn.functional.softplus(self.reliability_raw)
            total_penalty = strength[0] * total_disagreement
            state_penalty = strength[1] * state_disagreement
            sign = 1.0 if self.variant == "reliability_hybrid_inverse" else -1.0
            raw = raw.clone()
            raw[..., 1:3] = raw[..., 1:3] + sign * total_penalty.unsqueeze(-1)
            raw[..., 3:5] = raw[..., 3:5] + sign * state_penalty.unsqueeze(-1)
        elif self.variant == "linear_hybrid_padded":
            # Same parameter count as the reliability model, but the two
            # reliability parameters cannot affect its prediction.
            raw = raw + 0.0 * self.reliability_raw.sum()
        if self.variant == "nonlinear_hybrid_free_residual":
            residual = raw.sum(dim=-1) / math.sqrt(5.0)
            prediction = torch.sigmoid(
                torch.logit(base.clamp(1e-6, 1 - 1e-6)) + residual
            )
            weights = torch.full_like(experts, float("nan"))
        else:
            weights = torch.softmax(raw, dim=-1)
            prediction = torch.sum(weights * experts, dim=-1)
        return {
            "occupancy_rate": prediction,
            "base_rate": base,
            "experts": experts,
            "expert_weights": weights,
        }


def _loader(dataset: HybridDataset, config: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: HybridExpertModel, dataset: HybridDataset, config: Config) -> dict:
    model.eval()
    device = torch.device(config.device)
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "prediction", "base_prediction", "target", "target_index",
            "experts", "expert_weights",
        )
    }
    for cpu_batch in _loader(dataset, config, False):
        batch = _move(cpu_batch, device)
        output = model(
            batch["history"], batch["target_calendar"], batch["external_experts"],
            batch["origin_active"], batch["origin_unavailable"],
        )
        parts["prediction"].append(output["occupancy_rate"].float().cpu().numpy())
        parts["base_prediction"].append(output["base_rate"].float().cpu().numpy())
        parts["target"].append(batch["occupancy_rate"].float().cpu().numpy())
        parts["target_index"].append(batch["target_index"].cpu().numpy())
        parts["experts"].append(output["experts"].float().cpu().numpy())
        parts["expert_weights"].append(output["expert_weights"].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def train(model: HybridExpertModel, datasets: dict[str, HybridDataset], config: Config):
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    rows = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        sse = 0.0
        cells = 0
        for cpu_batch in _loader(datasets["train"], config, True):
            batch = _move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history"], batch["target_calendar"], batch["external_experts"],
                batch["origin_active"], batch["origin_unavailable"],
            )
            error = output["occupancy_rate"] - batch["occupancy_rate"]
            loss = error.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            sse += float(error.detach().square().sum())
            cells += int(error.numel())
        valid = evaluate(model, datasets["valid"], config)
        rmse = audited_metrics(valid["prediction"], valid["target"])["RMSE"]
        rows.append({"epoch": epoch, "train_RMSE": float(np.sqrt(sse / cells)), "valid_RMSE": rmse})
        print(json.dumps(rows[-1]), flush=True)
        if rmse < best_rmse - config.min_delta:
            best_rmse = rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch, time.perf_counter() - started


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
    datasets = {
        split: HybridDataset(data, scales, state_hour, indices, config)
        for split, indices in targets.items()
    }
    model = HybridExpertModel(data, scales, config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model, history, best_epoch, runtime = train(model, datasets, config)
    valid = evaluate(model, datasets["valid"], config)
    run_id = config.run_id or (
        f"hybrid_expert_v1_{config.variant}_f{config.fold}_h{config.horizon}_"
        f"s{config.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_dir = root / "innovation" / "hybrid_expert_runs" / run_id / (
        f"attempt_{config.attempt_id:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = []
    for semantics, function in (("audited", audited_metrics), ("official", official_metrics)):
        metrics.append({
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
            **function(valid["prediction"], valid["target"]),
        })
    history.to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "validation_predictions.npz", **valid, zone_ids=np.asarray(data.zone_ids))
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    (run_dir / "config.json").write_text(
        json.dumps({
            **asdict(config),
            "run_id": run_id,
            "source_root": str(source_root),
            "data_source": "real_UrbanEV_release_from_user_workspace",
            "state_cache_provenance": fine.provenance,
            "test_dataset_constructed": False,
            "test_arrays_loaded": False,
            "dense_168_history_used": False,
            "sparse_anchor_count": 2,
            "validation_target_hash": target_index_hash(targets["valid"]),
            "environment": {
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "success", "run_id": run_id}, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\UrbanEV\UrbanEV-main"))
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
        fold=args.fold, horizon=args.horizon, variant=args.variant, seed=args.seed,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        device=args.device, run_id=args.run_id, attempt_id=args.attempt_id,
    )
    try:
        run(args.root.resolve(), args.source_root.resolve(), config)
    except Exception:
        run_id = args.run_id or "hybrid_expert_failed"
        failed = args.root.resolve() / "innovation" / "hybrid_expert_runs" / run_id / (
            f"attempt_{args.attempt_id:02d}"
        )
        failed.mkdir(parents=True, exist_ok=True)
        (failed / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        (failed / "status.json").write_text(
            json.dumps({"status": "failed", "run_id": run_id}, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
