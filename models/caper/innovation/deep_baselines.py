from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import random
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# CuBLAS requires this to be set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from innovation.canonical import (
    calendar_features,
    canonical_boundaries,
    canonical_target_indices,
    history_indices,
    legacy_common_target_indices,
)
from innovation.data import OccupancyData, load_occupancy
from repro.metrics import audited_metrics, official_metrics


ModelName = Literal["mlp", "lstm"]
HistoryTransform = Literal["none", "phase_shuffle168"]
ProtocolName = Literal["canonical_native_common168", "legacy_common"]
ContextMode = Literal["history_only", "full"]


@dataclass(frozen=True)
class DeepRunConfig:
    fold: int
    horizon: int
    model: ModelName
    history_length: int
    history_transform: HistoryTransform
    protocol: ProtocolName
    context_mode: ContextMode
    common_history_budget: int
    seed: int
    transform_seed: int
    epochs: int
    patience: int
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    weight_decay: float
    hidden1: int
    hidden2: int
    node_embedding_dim: int
    min_delta: float
    gradient_clip: float
    device: str
    amp: bool
    train_limit: int
    run_id: str | None = None
    attempt_id: int = 1


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class CanonicalSequenceDataset(Dataset):
    """One item is one target time and contains all 275 zones.

    Keeping time as the dataset axis avoids materialising a multi-gigabyte
    ``time x node x history`` tensor.  The model flattens time and node only
    after a batch reaches the GPU.
    """

    def __init__(
        self,
        data: OccupancyData,
        targets: np.ndarray,
        horizon: int,
        history_length: int,
        history_transform: HistoryTransform = "none",
        transform_seed: int = 0,
    ) -> None:
        self.rate = data.rate.astype(np.float32, copy=False)
        self.counts = data.counts.astype(np.float32, copy=False)
        self.calendar = calendar_features(data.time)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.horizon = int(horizon)
        self.history_length = int(history_length)
        self.history_transform = history_transform
        self.transform_seed = int(transform_seed)
        if self.targets.ndim != 1 or self.targets.size == 0:
            raise ValueError("targets must be a non-empty one-dimensional array")
        history_indices(int(self.targets.min()), self.horizon, self.history_length)
        if history_transform == "phase_shuffle168" and history_length != 168:
            raise ValueError("phase_shuffle168 requires exactly 168 history hours")
        if history_transform not in {"none", "phase_shuffle168"}:
            raise ValueError(f"unknown history transform: {history_transform}")

    def __len__(self) -> int:
        return int(self.targets.size)

    def _phase_shuffle(self, history: np.ndarray, target_index: int) -> np.ndarray:
        """Destroy exact daily phase while preserving each day's value multiset.

        The seven 24-hour blocks are permuted and independently circularly
        shifted by a non-zero offset.  The calendar tensor remains tied to the
        real timestamps, so the control cannot recover the original phase from
        a silently permuted calendar.  A fixed target-dependent seed makes the
        transformation deterministic across workers and reruns.
        """

        n_nodes = history.shape[0]
        blocks = history.reshape(n_nodes, 7, 24)
        rng = np.random.default_rng(self.transform_seed + 1_000_003 * int(target_index))
        order = rng.permutation(7)
        shifts = rng.integers(1, 24, size=7)
        transformed = np.empty_like(blocks)
        for destination, source in enumerate(order):
            transformed[:, destination, :] = np.roll(
                blocks[:, source, :], int(shifts[destination]), axis=-1
            )
        return transformed.reshape(n_nodes, 168)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        target_index = int(self.targets[item])
        observed = history_indices(target_index, self.horizon, self.history_length)
        if int(observed.max()) > target_index - self.horizon:
            raise AssertionError("future information entered a history window")
        history = np.ascontiguousarray(self.rate[observed].T)
        if self.history_transform == "phase_shuffle168":
            history = np.ascontiguousarray(self._phase_shuffle(history, target_index))
        history_calendar = np.ascontiguousarray(self.calendar[observed])
        target_calendar = np.ascontiguousarray(self.calendar[target_index])
        return (
            torch.from_numpy(history),
            torch.from_numpy(history_calendar),
            torch.from_numpy(target_calendar),
            torch.from_numpy(np.ascontiguousarray(self.rate[target_index])),
            torch.from_numpy(np.ascontiguousarray(self.counts[target_index])),
            torch.tensor(target_index, dtype=torch.long),
        )


class DenseHistoryMLP(nn.Module):
    """A shared per-zone MLP with node identity and known calendar context."""

    def __init__(
        self,
        history_length: int,
        capacity: np.ndarray,
        hidden1: int,
        hidden2: int,
        node_embedding_dim: int,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        n_nodes = int(len(capacity))
        self.node_embedding = nn.Embedding(n_nodes, node_embedding_dim)
        standardized_capacity = _standardized_log_capacity(capacity)
        self.register_buffer("capacity", torch.from_numpy(standardized_capacity))
        self.use_context = bool(use_context)
        input_dim = history_length + node_embedding_dim + (5 if self.use_context else 0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_calendar: torch.Tensor,
        target_calendar: torch.Tensor,
    ) -> torch.Tensor:
        del history_calendar
        batch, n_nodes, _ = history.shape
        node = self.node_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        features = [history, node]
        if self.use_context:
            capacity = self.capacity.view(1, n_nodes, 1).expand(batch, -1, -1)
            calendar = target_calendar.unsqueeze(1).expand(-1, n_nodes, -1)
            features.extend([capacity, calendar])
        features = torch.cat(features, dim=-1)
        return torch.sigmoid(self.network(features).squeeze(-1))


class DenseHistoryLSTM(nn.Module):
    """A parameter-matched sequence baseline shared across all zones."""

    def __init__(
        self,
        capacity: np.ndarray,
        hidden1: int,
        hidden2: int,
        node_embedding_dim: int,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        n_nodes = int(len(capacity))
        self.node_embedding = nn.Embedding(n_nodes, node_embedding_dim)
        self.register_buffer("capacity", torch.from_numpy(_standardized_log_capacity(capacity)))
        self.use_context = bool(use_context)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden1, num_layers=1, batch_first=True)
        head_input = hidden1 + node_embedding_dim + (5 if self.use_context else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_calendar: torch.Tensor,
        target_calendar: torch.Tensor,
    ) -> torch.Tensor:
        del history_calendar
        batch, n_nodes, length = history.shape
        sequence = history.reshape(batch * n_nodes, length, 1)
        _, (hidden, _) = self.lstm(sequence)
        last = hidden[-1]
        node = self.node_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        node = node.reshape(batch * n_nodes, -1)
        features = [last, node]
        if self.use_context:
            capacity = self.capacity.view(1, n_nodes, 1).expand(batch, -1, -1)
            capacity = capacity.reshape(batch * n_nodes, 1)
            target_context = target_calendar.unsqueeze(1).expand(-1, n_nodes, -1)
            target_context = target_context.reshape(batch * n_nodes, 4)
            features.extend([capacity, target_context])
        features = torch.cat(features, dim=-1)
        return torch.sigmoid(self.head(features).reshape(batch, n_nodes))


def _standardized_log_capacity(capacity: np.ndarray) -> np.ndarray:
    values = np.log1p(np.asarray(capacity, dtype=np.float32))
    return ((values - values.mean()) / max(float(values.std()), 1e-8)).astype(np.float32)


def build_model(config: DeepRunConfig, capacity: np.ndarray) -> nn.Module:
    if config.model == "mlp":
        return DenseHistoryMLP(
            history_length=config.history_length,
            capacity=capacity,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            node_embedding_dim=config.node_embedding_dim,
            use_context=config.context_mode == "full",
        )
    if config.model == "lstm":
        return DenseHistoryLSTM(
            capacity=capacity,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            node_embedding_dim=config.node_embedding_dim,
            use_context=config.context_mode == "full",
        )
    raise ValueError(f"unknown model: {config.model}")


def _target_sets(data: OccupancyData, config: DeepRunConfig) -> dict[str, np.ndarray]:
    bounds = canonical_boundaries(data.time, config.fold)
    target_function = (
        canonical_target_indices
        if config.protocol == "canonical_native_common168"
        else legacy_common_target_indices
    )
    targets = {
        split: target_function(
            bounds,
            config.horizon,
            split,
            common_history_budget=config.common_history_budget,
        )
        for split in ("train", "valid", "test")
    }
    if config.train_limit > 0:
        targets["train"] = targets["train"][: config.train_limit]
    if config.history_length > config.common_history_budget:
        raise ValueError("history_length exceeds the shared history budget")
    if np.intersect1d(targets["train"], targets["valid"]).size:
        raise AssertionError("train and validation labels overlap")
    if np.intersect1d(targets["valid"], targets["test"]).size:
        raise AssertionError("validation and test labels overlap")
    return targets


def build_datasets(
    data: OccupancyData, config: DeepRunConfig
) -> tuple[dict[str, CanonicalSequenceDataset], dict[str, np.ndarray]]:
    targets = _target_sets(data, config)
    datasets = {
        split: CanonicalSequenceDataset(
            data=data,
            targets=indices,
            horizon=config.horizon,
            history_length=config.history_length,
            history_transform=config.history_transform,
            transform_seed=config.transform_seed,
        )
        for split, indices in targets.items()
    }
    return datasets, targets


def target_index_hash(indices: np.ndarray) -> str:
    """Stable identifier proving that candidate models score identical labels."""

    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _loader(
    dataset: CanonicalSequenceDataset,
    config: DeepRunConfig,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for batch in loader:
        history, history_calendar, target_calendar, target, count, target_index = _move_batch(
            batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            prediction = model(history, history_calendar, target_calendar)
        prediction_parts.append(prediction.float().cpu().numpy())
        target_parts.append(target.float().cpu().numpy())
        count_parts.append(count.float().cpu().numpy())
        index_parts.append(target_index.cpu().numpy())
    return {
        "prediction": np.concatenate(prediction_parts, axis=0),
        "target": np.concatenate(target_parts, axis=0),
        "target_count": np.concatenate(count_parts, axis=0),
        "target_index": np.concatenate(index_parts, axis=0),
    }


def train_model(
    model: nn.Module,
    datasets: dict[str, CanonicalSequenceDataset],
    config: DeepRunConfig,
) -> tuple[nn.Module, pd.DataFrame, int, float]:
    device = torch.device(config.device)
    model.to(device)
    train_loader = _loader(datasets["train"], config, shuffle=True)
    valid_loader = _loader(datasets["valid"], config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and device.type == "cuda"
    )
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        objective_sum = 0.0
        element_count = 0
        iterator = iter(train_loader)
        while True:
            group = list(itertools.islice(iterator, config.accumulation_steps))
            if not group:
                break
            optimizer.zero_grad(set_to_none=True)
            group_elements = sum(int(batch[3].numel()) for batch in group)
            for batch in group:
                history, history_calendar, target_calendar, target, _, _ = _move_batch(
                    batch, device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=config.amp and device.type == "cuda",
                ):
                    prediction = model(history, history_calendar, target_calendar)
                    batch_loss = torch.mean((prediction - target) ** 2)
                    weighted_loss = batch_loss * (target.numel() / group_elements)
                scaler.scale(weighted_loss).backward()
                elements = int(target.numel())
                objective_sum += float(batch_loss.detach()) * elements
                element_count += elements
            scaler.unscale_(optimizer)
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        validation = evaluate(model, valid_loader, device, config.amp)
        valid_metrics = audited_metrics(validation["prediction"], validation["target"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_MSE": objective_sum / element_count,
                "valid_RMSE": valid_metrics["RMSE"],
                "valid_MAE": valid_metrics["MAE"],
            }
        )
        print(json.dumps(history_rows[-1]), flush=True)
        if valid_metrics["RMSE"] < best_rmse - config.min_delta:
            best_rmse = valid_metrics["RMSE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    runtime = time.perf_counter() - started
    if best_epoch <= 0 or not np.isfinite(best_rmse):
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows), best_epoch, runtime


def _default_run_id(config: DeepRunConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    transform = "raw" if config.history_transform == "none" else config.history_transform
    context = "F" if config.context_mode == "full" else "H"
    return (
        f"canonical_{config.model}_L{config.history_length}_{context}_{transform}_"
        f"f{config.fold}_h{config.horizon}_s{config.seed}_{stamp}"
    )


def run(root: Path, config: DeepRunConfig) -> Path:
    _validate_config(config)
    total_started = time.perf_counter()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if config.history_length > config.common_history_budget:
        raise ValueError("history_length cannot exceed common_history_budget")
    set_deterministic(config.seed)
    torch.set_num_threads(1)
    data = load_occupancy(root / "audited" / "data")
    datasets, targets = build_datasets(data, config)
    model = build_model(config, data.capacity)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_id = config.run_id or _default_run_id(config)
    run_dir = root / "innovation" / "deep_runs" / run_id / f"attempt_{config.attempt_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = {
        **asdict(config),
        "run_id": run_id,
        "attempt_id": config.attempt_id,
        "parameters": parameter_count,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "target_counts": {name: int(len(values)) for name, values in targets.items()},
        "target_index_sha256": {name: target_index_hash(values) for name, values in targets.items()},
        "target_ranges": {
            name: [int(values[0]), int(values[-1])] for name, values in targets.items()
        },
        "input_sha256": {
            name: _file_sha256(root / "audited" / "data" / name)
            for name in ("occupancy.csv", "inf.csv")
        },
        "source_sha256": {
            name: _file_sha256(Path(__file__).resolve().parent / name)
            for name in ("canonical.py", "deep_baselines.py")
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    try:
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, history, best_epoch, runtime = train_model(model, datasets, config)
        device = torch.device(config.device)
        test = evaluate(model, _loader(datasets["test"], config, shuffle=False), device, config.amp)
        validation = evaluate(
            model, _loader(datasets["valid"], config, shuffle=False), device, config.amp
        )
        rows: list[dict[str, float | int | str]] = []
        for split, artifact in (("valid", validation), ("test", test)):
            for semantics, metric_function in (
                ("audited", audited_metrics),
                ("official", official_metrics),
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "model": config.model,
                        "history_length": config.history_length,
                        "history_transform": config.history_transform,
                        "context_mode": config.context_mode,
                        "protocol": config.protocol,
                        "evaluation_boundary_semantics": (
                            "transformer_native"
                            if config.protocol == "canonical_native_common168"
                            else "conventional_legacy"
                        ),
                        "train_target_policy": f"common_history_budget_{config.common_history_budget}",
                        "objective": "direct_endpoint_mse",
                        "feature_set": (
                            "history+target_calendar+node_id+capacity"
                            if config.context_mode == "full"
                            else "history+node_id"
                        ),
                        "fold": config.fold,
                        "horizon": config.horizon,
                        "seed": config.seed,
                        "split": split,
                        "metric_semantics": semantics,
                        "samples": len(artifact["target_index"]),
                        "parameters": parameter_count,
                        "best_epoch": best_epoch,
                        "runtime_seconds": runtime,
                        **metric_function(artifact["prediction"], artifact["target"]),
                    }
                )
        metrics = pd.DataFrame(rows)
        history.to_csv(run_dir / "history.csv", index=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        np.savez_compressed(
            run_dir / "predictions.npz",
            prediction=test["prediction"],
            target=test["target"],
            target_count=test["target_count"],
            target_index=test["target_index"],
            input_start_index=test["target_index"] - config.horizon - config.history_length + 1,
            origin_index=test["target_index"] - config.horizon,
            target_time=data.time[test["target_index"]].astype(str).to_numpy(),
            zone_ids=np.asarray(data.zone_ids),
            validation_prediction=validation["prediction"],
            validation_target=validation["target"],
            validation_target_index=validation["target_index"],
        )
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        peak_vram = (
            int(torch.cuda.max_memory_allocated()) if config.device.startswith("cuda") else 0
        )
        status = {
            "status": "success",
            "finished_at": datetime.now().isoformat(),
            "best_epoch": best_epoch,
            "runtime_seconds": runtime,
            "total_runtime_seconds": time.perf_counter() - total_started,
            "peak_vram_bytes": peak_vram,
        }
        (run_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(metrics.to_string(index=False), flush=True)
        print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), **status}), flush=True)
        return run_dir
    except Exception as error:
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "finished_at": datetime.now().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def _auto_hidden(model: str, history_length: int) -> tuple[int, int]:
    if model == "mlp" and history_length <= 12:
        return 180, 96
    if model == "mlp":
        return 96, 48
    return 64, 64


def _validate_config(config: DeepRunConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.hidden1 <= 0 or config.hidden2 <= 0 or config.node_embedding_dim < 0:
        raise ValueError("model dimensions are invalid")
    if config.train_limit < 0:
        raise ValueError("train_limit cannot be negative")
    if config.seed < 0 or config.transform_seed < 0:
        raise ValueError("seed values cannot be negative")
    if config.attempt_id <= 0:
        raise ValueError("attempt_id must be positive")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--horizon", type=int, required=True, choices=(3, 6, 9, 12))
    parser.add_argument("--model", choices=("mlp", "lstm"), required=True)
    parser.add_argument("--history-length", type=int, required=True, choices=(12, 168))
    parser.add_argument(
        "--history-transform", choices=("none", "phase_shuffle168"), default="none"
    )
    parser.add_argument(
        "--protocol",
        choices=("canonical_native_common168", "legacy_common"),
        default="canonical_native_common168",
    )
    parser.add_argument("--context-mode", choices=("history_only", "full"), default="full")
    parser.add_argument("--common-history-budget", type=int, default=168)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden1", type=int)
    parser.add_argument("--hidden2", type=int)
    parser.add_argument("--node-embedding-dim", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id", type=int, default=1)
    args = parser.parse_args()
    default_hidden1, default_hidden2 = _auto_hidden(args.model, args.history_length)
    if args.model == "mlp" and args.history_length == 12 and args.context_mode == "history_only":
        default_hidden1 = 185
    config = DeepRunConfig(
        fold=args.fold,
        horizon=args.horizon,
        model=args.model,
        history_length=args.history_length,
        history_transform=args.history_transform,
        protocol=args.protocol,
        context_mode=args.context_mode,
        common_history_budget=args.common_history_budget,
        seed=args.seed,
        transform_seed=args.transform_seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size or (32 if args.model == "mlp" else 2),
        accumulation_steps=args.accumulation_steps or (1 if args.model == "mlp" else 16),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden1=args.hidden1 or default_hidden1,
        hidden2=args.hidden2 or default_hidden2,
        node_embedding_dim=args.node_embedding_dim,
        min_delta=args.min_delta,
        gradient_clip=args.gradient_clip,
        device=args.device,
        amp=args.amp,
        train_limit=args.train_limit,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
    )
    run(args.root, config)


if __name__ == "__main__":
    main()
