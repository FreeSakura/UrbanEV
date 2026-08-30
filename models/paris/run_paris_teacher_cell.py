"""Run one frozen Paris development teacher cell; never loads formal/protected data."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from paris_teacher_common import (
    DevelopmentBundle,
    array_hash,
    cell_contract,
    json_hash,
    implementation_bundle_hash,
    implementation_hashes,
    load_bundle,
    load_manifests,
    load_model_config,
    masked_metrics,
    project_root,
    set_deterministic,
    sha256_file,
    write_json_atomic,
)

TIMEXER_NAME = "TimeXer_local_audited_compact_L168"


class ParisSamples(Dataset):
    def __init__(self, bundle: DevelopmentBundle, contract: dict[str, Any], targets: np.ndarray, horizon: int):
        self.bundle = bundle
        self.contract = contract
        self.targets = np.asarray(targets, np.int64)
        self.horizon = int(horizon)
        self.lookback = 168
        if not self.targets.size or int(self.targets.min()) - horizon - self.lookback + 1 < 0:
            raise ValueError("empty/invalid sample split")

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_index = int(self.targets[item])
        origin = target_index - self.horizon
        history_index = np.arange(origin - self.lookback + 1, origin + 1, dtype=np.int64)
        state = self.contract["state_filled"]
        raw_target = self.bundle.rate_observed[target_index]
        target_mask = self.bundle.observed_mask[target_index]
        scaled_target = (raw_target - self.contract["scaler_mean"]) / self.contract["scaler_scale"]
        raw_target_safe = np.where(target_mask, raw_target, 0.0).astype(np.float32)
        scaled_target_safe = np.where(target_mask, scaled_target, 0.0).astype(np.float32)
        stamp = self.bundle.time[target_index]
        calendar = np.asarray(
            [
                np.sin(2 * np.pi * stamp.hour / 24.0),
                np.cos(2 * np.pi * stamp.hour / 24.0),
                np.sin(2 * np.pi * stamp.dayofweek / 7.0),
                np.cos(2 * np.pi * stamp.dayofweek / 7.0),
            ],
            dtype=np.float32,
        )
        phase = np.stack(
            [
                state[origin, :, 1],
                state[origin, :, 2],
                state[target_index - 24, :, 1],
                state[target_index - 168, :, 1],
                state[target_index - 24, :, 2],
                state[target_index - 168, :, 2],
            ],
            axis=-1,
        ).astype(np.float32)
        return {
            "history": torch.from_numpy(np.ascontiguousarray(self.contract["rate_scaled"][history_index])),
            "target_raw": torch.from_numpy(raw_target_safe),
            "target_scaled": torch.from_numpy(scaled_target_safe),
            "mask": torch.from_numpy(target_mask.astype(np.bool_)),
            "phase": torch.from_numpy(np.ascontiguousarray(phase)),
            "calendar": torch.from_numpy(calendar),
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }


def _fit_component(
    filled: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    day = filled[targets - 24].astype(np.float64)
    week = filled[targets - 168].astype(np.float64)
    origin = filled[targets - horizon].astype(np.float64)
    target = observed[targets].astype(np.float64)
    valid = mask[targets] & np.isfinite(target)
    difference = (day - week)[valid]
    residual = (target - day)[valid]
    denominator = float(difference @ difference)
    alpha = float(difference @ residual / denominator) if denominator > 0 else 0.0
    beta = float(np.clip(-alpha, 0.0, 1.0))
    phase = (1.0 - beta) * day + beta * week
    departure = (origin - phase)[valid]
    response = (target - phase)[valid]
    denominator = float(departure @ departure)
    retention = float(np.clip(departure @ response / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
    return beta, retention


def fit_caper_coefficients(bundle: DevelopmentBundle, contract: dict[str, Any], horizon: int) -> dict[str, float | bool]:
    targets = contract["train_targets"]
    state_filled = contract["state_filled"]
    beta_active, raw_active = _fit_component(
        state_filled[..., 1], bundle.state_observed[..., 1], bundle.observed_mask, targets, horizon
    )
    beta_unavailable, raw_unavailable = _fit_component(
        state_filled[..., 2], bundle.state_observed[..., 2], bundle.observed_mask, targets, horizon
    )
    if raw_active <= raw_unavailable:
        retention_active, retention_unavailable = raw_active, raw_unavailable
    else:
        pooled = 0.5 * (raw_active + raw_unavailable)
        retention_active = retention_unavailable = pooled
    return {
        "beta_active": beta_active,
        "beta_unavailable": beta_unavailable,
        "raw_retention_active": raw_active,
        "raw_retention_unavailable": raw_unavailable,
        "retention_active": retention_active,
        "retention_unavailable": retention_unavailable,
        "ordered_active_le_unavailable": retention_active <= retention_unavailable,
        "retention_projection": "euclidean_two_point_isotonic_active_le_unavailable",
    }


class ParisCAPERPhaseOnly(nn.Module):
    """Paris-specific phase model; random initialization and train-only coefficients."""

    def __init__(self, station_count: int, capacity: np.ndarray, coefficients: dict[str, Any], config: dict[str, Any]):
        super().__init__()
        hidden = int(config["history_encoder_hidden"])
        embedding = int(config["station_embedding_dim"])
        phase_hidden = int(config["phase_hidden"])
        dropout = float(config["dropout"])
        self.history_encoder = nn.Sequential(nn.Linear(168, hidden), nn.GELU(), nn.Dropout(dropout))
        self.station_embedding = nn.Embedding(station_count, embedding)
        context_dim = hidden + embedding + 6 + 4 + 1
        self.base_head = nn.Sequential(nn.Linear(context_dim, phase_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(phase_hidden, 1))
        self.gate_head = nn.Sequential(nn.Linear(context_dim + 1, phase_hidden), nn.GELU(), nn.Linear(phase_hidden, 1))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.0)
        self.register_buffer("capacity", torch.from_numpy((capacity / max(float(capacity.max()), 1.0)).astype(np.float32)))
        for key in ("beta_active", "beta_unavailable", "retention_active", "retention_unavailable"):
            self.register_buffer(key, torch.tensor(float(coefficients[key]), dtype=torch.float32))

    def forward(self, history: torch.Tensor, phase: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        batch, _, stations = history.shape
        encoded = self.history_encoder(history.permute(0, 2, 1))
        station_ids = torch.arange(stations, device=history.device)
        station_embedding = self.station_embedding(station_ids).unsqueeze(0).expand(batch, -1, -1)
        calendar_expanded = calendar.unsqueeze(1).expand(-1, stations, -1)
        capacity = self.capacity.view(1, stations, 1).expand(batch, -1, -1)
        context = torch.cat([encoded, station_embedding, phase, calendar_expanded, capacity], dim=-1)
        base = torch.sigmoid(self.base_head(context).squeeze(-1))
        origin_active, origin_unavailable = phase[..., 0], phase[..., 1]
        active_phase = (1.0 - self.beta_active) * phase[..., 2] + self.beta_active * phase[..., 3]
        unavailable_phase = (1.0 - self.beta_unavailable) * phase[..., 4] + self.beta_unavailable * phase[..., 5]
        physical = (
            self.retention_active * origin_active
            + (1.0 - self.retention_active) * active_phase
            + self.retention_unavailable * origin_unavailable
            + (1.0 - self.retention_unavailable) * unavailable_phase
        ).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.gate_head(torch.cat([context, physical.unsqueeze(-1)], dim=-1)).squeeze(-1))
        return ((1.0 - gate) * base + gate * physical).clamp(0.0, 1.0)


def build_timexer(root: Path, station_count: int, config: dict[str, Any]) -> nn.Module:
    source = root.parent / "work/UrbanEV-reproduction/audited/code-transformer"
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    model_path = source / "models/TimeXer.py"
    spec = importlib.util.spec_from_file_location("paris_qualification_official_timexer", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = SimpleNamespace(
        features="M",
        seq_len=168,
        pred_len=1,
        use_norm=1,
        patch_len=12,
        enc_in=station_count,
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        e_layers=int(config["e_layers"]),
        d_ff=int(config["d_ff"]),
        dropout=float(config["dropout"]),
        factor=1,
        activation="gelu",
        embed="timeF",
        freq="h",
    )
    return module.Model(namespace)


def _loader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, generator=generator if shuffle else None)


def _forward(model: nn.Module, model_name: str, batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    history = batch["history"].to(device, non_blocking=True)
    if model_name == TIMEXER_NAME:
        output = model(history, None, None, None)[:, -1, :]
        target = batch["target_scaled"].to(device, non_blocking=True)
    else:
        output = model(history, batch["phase"].to(device, non_blocking=True), batch["calendar"].to(device, non_blocking=True))
        target = batch["target_raw"].to(device, non_blocking=True)
    return output, target


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    dataset: ParisSamples,
    device: torch.device,
    batch_size: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in _loader(dataset, batch_size, 42, False):
        output, _ = _forward(model, model_name, batch, device)
        prediction = output.float().cpu().numpy()
        if model_name == TIMEXER_NAME:
            prediction = prediction * scale[None, :] + mean[None, :]
        predictions.append(prediction)
        targets.append(batch["target_raw"].numpy())
        masks.append(batch["mask"].numpy())
        indices.append(batch["target_index"].numpy())
    raw = np.concatenate(predictions).astype(np.float32)
    return {
        "raw_prediction": raw,
        "clipped_prediction": np.clip(raw, 0.0, 1.0),
        "target": np.concatenate(targets).astype(np.float32),
        "mask": np.concatenate(masks).astype(bool),
        "target_index": np.concatenate(indices).astype(np.int64),
    }


def train(
    model: nn.Module,
    model_name: str,
    train_set: ParisSamples,
    valid_set: ParisSamples,
    device: torch.device,
    config: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[nn.Module, pd.DataFrame, int, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    batch_size = int(config["batch_size"])
    best_state = copy.deepcopy(model.state_dict())
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        error_sum, count = 0.0, 0
        for batch in _loader(train_set, batch_size, 42 + epoch, True):
            optimizer.zero_grad(set_to_none=True)
            output, target = _forward(model, model_name, batch, device)
            mask = batch["mask"].to(device, non_blocking=True)
            if not bool(mask.any()):
                continue
            error = (output - target)[mask]
            loss = torch.mean(error * error)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            error_sum += float(torch.sum(error.detach() ** 2))
            count += int(error.numel())
        validation = evaluate(model, model_name, valid_set, device, batch_size, mean, scale)
        validation_rmse = masked_metrics(validation["clipped_prediction"], validation["target"], validation["mask"])["RMSE"]
        rows.append({"epoch": epoch, "train_masked_MSE": error_sum / max(count, 1), "validation_clipped_RMSE": validation_rmse})
        if validation_rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, stale = validation_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    if best_epoch <= 0:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch, time.perf_counter() - started


def _verified_success(run_root: Path, fingerprint: str) -> Path | None:
    if not run_root.exists():
        return None
    for attempt in sorted(run_root.glob("attempt_*")):
        status_path, config_path, receipt_path = attempt / "status.json", attempt / "config.json", attempt / "SUCCESS_RECEIPT.json"
        if status_path.exists() and config_path.exists() and receipt_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if status.get("status") == "success" and payload.get("manifest_fingerprint") == fingerprint:
                hashes = receipt.get("artifact_sha256", {})
                if hashes and all((attempt / name).exists() and sha256_file(attempt / name) == value for name, value in hashes.items()):
                    return attempt
    return None


def run_cell(root: Path, run_id: str, attempt_id: int, device_name: str) -> Path | None:
    model_config = load_model_config(root)
    folds, runs = load_manifests(root)
    matches = runs[runs.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown/nonunique run_id: {run_id}")
    row = matches.iloc[0]
    if int(row.seed) != 42 or int(row.lookback) != 168 or row.model not in ("Paris_CAPER_phase_only", TIMEXER_NAME):
        raise RuntimeError("frozen run config drift")
    fold_row = folds[(folds.fold == row.fold) & (folds.horizon == int(row.horizon))]
    if len(fold_row) != 1:
        raise AssertionError("fold manifest identity failure")
    output_root = root / "experiments/09_distill_v2/outputs/teacher_qualification/runs" / run_id
    if str(row.implementation_bundle_hash) != implementation_bundle_hash(root):
        raise RuntimeError("v1.1 implementation bundle drift")
    duplicate = _verified_success(output_root, str(row.fingerprint))
    if duplicate is not None:
        print(json.dumps({"status": "SKIPPED_DUPLICATE", "run_dir": str(duplicate)}))
        return None
    run_dir = output_root / f"attempt_{attempt_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=run_dir / "run.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    status_path = run_dir / "status.json"
    write_json_atomic(status_path, {"status": "running", "started_at": datetime.now().isoformat(), "run_id": run_id})
    total_started = time.perf_counter()
    try:
        set_deterministic(42)
        torch.set_num_threads(1)
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        bundle = load_bundle(root)
        state_source = np.load(bundle.paths["state"], allow_pickle=False)
        current_data_hashes = {
            "rate_sha256": sha256_file(bundle.paths["rate"]),
            "mask_sha256": sha256_file(bundle.paths["mask"]),
            "state_npz_sha256": sha256_file(bundle.paths["state"]),
            "capacity_registry_sha256_provenance_only": sha256_file(bundle.paths["capacity"]),
            "state_array_hash": array_hash(np.nan_to_num(state_source["state_rate"], nan=-9.0).astype("<f4")),
            "capacity_array_hash": array_hash(np.nan_to_num(state_source["capacity_count"], nan=-9.0).astype("<f4")),
            "global_mask_array_hash": array_hash(bundle.observed_mask.astype(np.uint8)),
        }
        if json_hash(current_data_hashes) != str(row.data_bundle_hash):
            raise RuntimeError("v1.1 development data bundle drift")
        if sha256_file(root / "experiments/09_distill_v2/PARIS_TEACHER_MODEL_CONFIG.json") != str(row.model_config_hash):
            raise RuntimeError("v1.1 model config drift")
        contract = cell_contract(bundle, fold_row.iloc[0], 168)
        horizon = int(row.horizon)
        datasets = {
            "train": ParisSamples(bundle, contract, contract["train_targets"], horizon),
            "validation": ParisSamples(bundle, contract, contract["validation_targets"], horizon),
            "evaluation": ParisSamples(bundle, contract, contract["evaluation_targets"], horizon),
        }
        caper_coefficients: dict[str, Any] | None = None
        if row.model == TIMEXER_NAME:
            active_config = model_config["time_xer"]
            model = build_timexer(root, len(bundle.stations), active_config)
        else:
            active_config = model_config["paris_caper"]
            caper_coefficients = fit_caper_coefficients(bundle, contract, horizon)
            model = ParisCAPERPhaseOnly(len(bundle.stations), contract["fold_capacity"], caper_coefficients, active_config)
        source_timexer = root.parent / "work/UrbanEV-reproduction/audited/code-transformer/models/TimeXer.py"
        config_payload = {
            "protocol_id": "PARIS_DEVELOPMENT_TEACHER_QUALIFICATION_V1_0",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "manifest_fingerprint": str(row.fingerprint),
            "model": str(row.model),
            "fold": str(row.fold),
            "horizon": horizon,
            "seed": 42,
            "lookback": 168,
            "objective": "direct_endpoint",
            "model_config": active_config,
            "caper_train_only_coefficients": caper_coefficients,
            "identity": contract["identity"],
            "fold_train_only_capacity": contract["fold_capacity"].astype(float).tolist(),
            "fold_train_only_capacity_hash": contract["fold_capacity_hash"],
            "split_counts": {key: len(dataset) for key, dataset in datasets.items()},
            "target_mask_is_model_input": False,
            "mask_usage": ["loss", "validation_selection", "metric"],
            "external_scaler": "per_station_zscore_train_observed_only",
            "scaler_fit_end_index": int(contract["fit_end_index"]),
            "loaded_external_checkpoint": False,
            "loaded_urbanev_coefficients": False,
            "formal_target_access": False,
            "protected_target_access": False,
            "implementation_bundle_hash": implementation_bundle_hash(root),
            "source_sha256": implementation_hashes(root),
            "timexer_source_status": "local_audited_adaptation_not_exact_upstream_official",
            "input_sha256": {key: sha256_file(path) for key, path in bundle.paths.items()},
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": device_name,
            },
        }
        config_payload["execution_config_hash"] = json_hash(config_payload)
        write_json_atomic(run_dir / "config.json", config_payload)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, history, best_epoch, train_seconds = train(
            model,
            str(row.model),
            datasets["train"],
            datasets["validation"],
            device,
            active_config,
            contract["scaler_mean"],
            contract["scaler_scale"],
        )
        validation = evaluate(model, str(row.model), datasets["validation"], device, int(active_config["batch_size"]), contract["scaler_mean"], contract["scaler_scale"])
        evaluation = evaluate(model, str(row.model), datasets["evaluation"], device, int(active_config["batch_size"]), contract["scaler_mean"], contract["scaler_scale"])
        if not np.array_equal(evaluation["target_index"], contract["evaluation_targets"]):
            raise AssertionError("saved evaluation target order drift")
        validation_metric = masked_metrics(validation["clipped_prediction"], validation["target"], validation["mask"])
        history.to_csv(run_dir / "history.csv", index=False)
        pd.DataFrame([{"split": "validation_for_early_stopping_only", **validation_metric}]).to_csv(run_dir / "validation_metrics.csv", index=False)
        np.savez_compressed(
            run_dir / "predictions.npz",
            **evaluation,
            target_time=bundle.time[evaluation["target_index"]].astype(str).to_numpy(),
            target_time_ns=bundle.time[evaluation["target_index"]].view("i8"),
            origin_index=evaluation["target_index"] - horizon,
            origin_time=bundle.time[evaluation["target_index"] - horizon].astype(str).to_numpy(),
            origin_time_ns=bundle.time[evaluation["target_index"] - horizon].view("i8"),
            station_ids=bundle.stations,
            validation_raw_prediction=validation["raw_prediction"],
            validation_clipped_prediction=validation["clipped_prediction"],
            validation_target=validation["target"],
            validation_mask=validation["mask"],
            validation_target_index=validation["target_index"],
            validation_target_time_ns=bundle.time[validation["target_index"]].view("i8"),
            validation_origin_time_ns=bundle.time[validation["target_index"] - horizon].view("i8"),
            eval_target_value_hash=np.asarray(contract["identity"]["eval_target_value_hash"]),
            eval_mask_hash=np.asarray(contract["identity"]["eval_mask_hash"]),
            fill_hash=np.asarray(contract["identity"]["fill_hash"]),
            scaler_hash=np.asarray(contract["identity"]["scaler_hash"]),
            fold_capacity_hash=np.asarray(contract["identity"]["fold_capacity_hash"]),
        )
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        status = {
            "status": "success",
            "run_id": run_id,
            "finished_at": datetime.now().isoformat(),
            "best_epoch": best_epoch,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "train_seconds": train_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "prediction_sha256": sha256_file(run_dir / "predictions.npz"),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
            "identity": contract["identity"],
            "evaluation_metrics_computed": False,
            "formal_target_access": False,
            "protected_target_access": False,
        }
        write_json_atomic(status_path, status)
        logging.info("cell completed without opening aggregate metrics")
        logging.shutdown()
        receipt_files = ("config.json", "status.json", "predictions.npz", "checkpoint.pt", "history.csv", "validation_metrics.csv", "run.log")
        write_json_atomic(
            run_dir / "SUCCESS_RECEIPT.json",
            {
                "status": "PASS",
                "manifest_fingerprint": str(row.fingerprint),
                "artifact_sha256": {name: sha256_file(run_dir / name) for name in receipt_files},
            },
        )
        print(json.dumps({"status": "success", "run_dir": str(run_dir)}))
        return run_dir
    except Exception as error:
        write_json_atomic(
            status_path,
            {
                "status": "failed",
                "run_id": run_id,
                "finished_at": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "formal_target_access": False,
                "protected_target_access": False,
            },
        )
        logging.exception("cell failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_cell(args.project_root.resolve(), args.run_id, args.attempt_id, args.device)


if __name__ == "__main__":
    main()
