"""Run the frozen minimal CAPER-TimeXer router_v1 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FOLDS = (1, 2, 3, 4, 5, 6)
HORIZONS = (3, 6, 9, 12)


def _helpers(project_root: Path):
    diagnostics = project_root / "experiments" / "01_expert_diagnostics" / "scripts"
    fusions = project_root / "experiments" / "03_fixed_fusions" / "scripts"
    for path in (diagnostics, fusions):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from analyze_existing_experts import _caper_artifact, _sample_indices, _timexer_artifacts
    from analyze_state_features import _expert_features, _load_rate, _state_features
    from evaluate_fixed_fusions import _caper_validation

    return (
        _caper_artifact,
        _sample_indices,
        _timexer_artifacts,
        _expert_features,
        _load_rate,
        _state_features,
        _caper_validation,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    absolute = np.abs(error)
    target_sum = float(np.sum(np.abs(target)))
    rae_denominator = float(np.sum(np.abs(target - target.mean())))
    smape_denominator = np.abs(prediction) + np.abs(target)
    mask = smape_denominator > 1e-8
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(absolute)),
        "WAPE": float(np.sum(absolute) / target_sum) if target_sum else float("nan"),
        "sMAPE": float(np.mean(2 * absolute[mask] / smape_denominator[mask])) if mask.any() else float("nan"),
        "RAE": float(np.sum(absolute) / rae_denominator) if rae_denominator else float("nan"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GateMLP(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(len(HORIZONS), 4)
        self.network = nn.Sequential(
            nn.Linear(feature_count + 4, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        embedded = self.horizon_embedding(horizon)
        return self.network(torch.cat([features, embedded], dim=1)).squeeze(1)


def _gbdt(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _sample(dataset: dict[str, np.ndarray], maximum: int, seed: int, time_slice=None):
    rows, zones = dataset["target"].shape
    start, stop = (0, rows) if time_slice is None else time_slice
    flat_start, flat_stop = start * zones, stop * zones
    candidates = np.arange(flat_start, flat_stop, dtype=np.int64)
    if maximum > 0 and len(candidates) > maximum:
        rng = np.random.default_rng(seed)
        candidates = np.sort(rng.choice(candidates, size=maximum, replace=False))
    return {
        "features": dataset["features"][candidates],
        "horizon_index": dataset["horizon_index"][candidates],
        "caper": dataset["caper"].reshape(-1)[candidates],
        "timexer": dataset["timexer"].reshape(-1)[candidates],
        "target": dataset["target"].reshape(-1)[candidates],
        "label": dataset["label"].reshape(-1)[candidates],
    }


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _train_mlp(
    train: dict[str, np.ndarray],
    selection: dict[str, np.ndarray],
    seed: int,
    device: str,
) -> tuple[GateMLP, np.ndarray, np.ndarray, int, pd.DataFrame]:
    mean = train["features"].mean(axis=0).astype(np.float32)
    std = train["features"].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_x = ((train["features"] - mean) / std).astype(np.float32)
    selection_x = ((selection["features"] - mean) / std).astype(np.float32)
    train_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train["horizon_index"].astype(np.int64)),
        torch.from_numpy(train["caper"].astype(np.float32)),
        torch.from_numpy(train["timexer"].astype(np.float32)),
        torch.from_numpy(train["target"].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = GateMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.HuberLoss(delta=0.05)
    best_state = copy.deepcopy(model.state_dict())
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    rows = []
    selection_tensors = (
        torch.from_numpy(selection_x).to(device),
        torch.from_numpy(selection["horizon_index"].astype(np.int64)).to(device),
    )
    for epoch in range(1, 31):
        model.train()
        loss_sum, count = 0.0, 0
        for features, horizon, caper, timexer, target in loader:
            features, horizon = features.to(device), horizon.to(device)
            caper, timexer, target = caper.to(device), timexer.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            gate = model(features, horizon)
            prediction = caper + gate * (timexer - caper)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        model.eval()
        with torch.no_grad():
            gate = model(*selection_tensors).cpu().numpy()
        prediction = selection["caper"] + gate * (
            selection["timexer"] - selection["caper"]
        )
        rmse = float(np.sqrt(np.mean((prediction - selection["target"]) ** 2)))
        rows.append({"epoch": epoch, "train_Huber": loss_sum / count, "selection_RMSE": rmse})
        if rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    if best_epoch == 0:
        raise RuntimeError("MLP gate produced no finite selection checkpoint")
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch, pd.DataFrame(rows)


@torch.no_grad()
def _predict_mlp(
    model: GateMLP,
    features: np.ndarray,
    horizon_index: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    normalized = ((features - mean) / std).astype(np.float32)
    output = []
    for start in range(0, len(normalized), 65_536):
        stop = min(start + 65_536, len(normalized))
        gate = model(
            torch.from_numpy(normalized[start:stop]).to(device),
            torch.from_numpy(horizon_index[start:stop].astype(np.int64)).to(device),
        )
        output.append(gate.cpu().numpy())
    return np.concatenate(output)


def _time_block_indices(length: int, block_length: int, rng: np.random.Generator):
    width = min(length, block_length)
    count = int(np.ceil(length / width))
    starts = rng.integers(0, length - width + 1, size=count)
    return np.concatenate(
        [np.arange(start, start + width, dtype=np.int64) for start in starts]
    )[:length]


def _bootstrap(
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    gains = []
    differences = []
    for _ in range(iterations):
        router_values, fixed_values = [], []
        for router, fixed, target in arrays:
            index = _time_block_indices(len(target), 24, rng)
            router_values.append(float(np.sqrt(np.mean((router[index] - target[index]) ** 2))))
            fixed_values.append(float(np.sqrt(np.mean((fixed[index] - target[index]) ** 2))))
        router_macro = float(np.mean(router_values))
        fixed_macro = float(np.mean(fixed_values))
        difference = fixed_macro - router_macro
        differences.append(difference)
        gains.append(100 * difference / fixed_macro)
    return {
        "iterations": iterations,
        "block_unit": "target_timestamp_all_275_zones_together",
        "block_length_hours": 24,
        "RMSE_difference_95_CI": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "gain_percent_95_CI": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
        "probability_gain_positive": float(np.mean(np.asarray(gains) > 0)),
    }


def run(
    project_root: Path,
    source_root: Path,
    output_dir: Path,
    max_samples_per_cell: int,
    bootstrap_iterations: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    if seed != 42:
        raise ValueError("router_v1 freezes seed 42")
    if max_samples_per_cell <= 0 or bootstrap_iterations <= 0:
        raise ValueError("sample and bootstrap budgets must be positive")
    sanity = max_samples_per_cell < 25_000 or bootstrap_iterations < 1_000
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _set_seed(seed)
    (
        caper_artifact,
        _,
        timexer_artifacts_fn,
        expert_features,
        load_rate,
        state_features,
        caper_validation,
    ) = _helpers(project_root)
    rate, time_index, capacity, zone_ids = load_rate(source_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from innovation.canonical import canonical_boundaries, canonical_target_indices

    tx_paths = timexer_artifacts_fn(source_root)
    router_config_path = (
        project_root
        / "experiments"
        / "05_router_v1"
        / "configs"
        / "ROUTER_V1.json"
    )
    router_config = json.loads(router_config_path.read_text(encoding="utf-8"))
    if (
        router_config.get("protocol_version") != "router_v1"
        or router_config.get("seed") != seed
        or router_config.get("horizons") != list(HORIZONS)
        or router_config.get("router_features") != "state21_v1"
        or router_config.get("probability_calibration") != "raw"
        or router_config.get("mlp")
        != {
            "horizon_embedding": 4,
            "hidden": [32, 16],
            "dropout": 0.1,
            "loss": "Huber",
            "learning_rate": 0.001,
            "batch_size": 4096,
            "epochs": 30,
            "patience": 5,
        }
        or router_config.get("gate4")
        != {
            "macro_rmse_gain_percent_min": 1.0,
            "fold_wins_min": 5,
            "cell_wins_min": 18,
            "paired_block_ci_lower_gt": 0.0,
            "max_horizon_deterioration_percent": 1.0,
        }
    ):
        raise RuntimeError("hardcoded router implementation diverges from ROUTER_V1.json")
    fixed_weights_path = (
        project_root
        / "experiments"
        / "03_fixed_fusions"
        / "outputs"
        / "FIXED_FUSION_WEIGHTS.csv"
    )
    fixed_summary_path = (
        project_root
        / "experiments"
        / "03_fixed_fusions"
        / "outputs"
        / "FIXED_FUSION_SUMMARY.json"
    )
    fixed_weights = pd.read_csv(fixed_weights_path)
    fixed_summary = json.loads(fixed_summary_path.read_text(encoding="utf-8"))
    best_fixed_method = fixed_summary["best_fixed_method"]
    expected_fixed_keys = {(fold, horizon) for fold in FOLDS for horizon in HORIZONS}
    actual_fixed_keys = {
        (int(row.evaluation_fold), int(row.horizon))
        for row in fixed_weights.itertuples()
    }
    if (
        len(fixed_weights) != 24
        or actual_fixed_keys != expected_fixed_keys
        or fixed_weights.duplicated(["evaluation_fold", "horizon"]).any()
        or best_fixed_method not in {"global_fixed", "horizon_fixed"}
        or fixed_summary.get("target_identity_verified") is not True
        or fixed_summary.get("gate4_reference_frozen") is not True
    ):
        raise RuntimeError("fixed-fusion reference is incomplete, duplicated, or unfrozen")
    datasets: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    input_artifact_hashes: dict[str, str] = {}
    for fold in FOLDS:
        boundaries = canonical_boundaries(time_index, fold)
        for horizon_index, horizon in enumerate(HORIZONS):
            tx = np.load(tx_paths[(fold, horizon)], allow_pickle=False)
            caper_test_path = caper_artifact(source_root, fold, horizon)
            caper_valid_path = caper_validation(source_root, fold, horizon)
            caper_test = np.load(caper_test_path, allow_pickle=False)
            caper_valid = np.load(caper_valid_path, allow_pickle=False)
            input_artifact_hashes[f"TimeXer_f{fold}_h{horizon}"] = _sha256(
                tx_paths[(fold, horizon)]
            )
            input_artifact_hashes[f"CAPER_test_f{fold}_h{horizon}"] = _sha256(
                caper_test_path
            )
            input_artifact_hashes[f"CAPER_valid_f{fold}_h{horizon}"] = _sha256(
                caper_valid_path
            )
            if not np.array_equal(zone_ids.astype(str), tx["zone_ids"].astype(str)):
                raise AssertionError(f"audited zone order mismatch f{fold} h{horizon}")
            if not np.array_equal(tx["zone_ids"], caper_test["zone_ids"]):
                raise AssertionError(f"test expert zone mismatch f{fold} h{horizon}")
            if not np.array_equal(tx["zone_ids"], caper_valid["zone_ids"]):
                raise AssertionError(f"validation expert zone mismatch f{fold} h{horizon}")
            for split, tx_index_key, tx_target_key, tx_prediction_key, caper_data in (
                ("test", "target_index", "target", "clipped_prediction", caper_test),
                (
                    "valid",
                    "validation_target_index",
                    "validation_target",
                    "validation_clipped_prediction",
                    caper_valid,
                ),
            ):
                expected_index = canonical_target_indices(
                    boundaries,
                    horizon,
                    split,
                    common_history_budget=168,
                )
                if not np.array_equal(tx[tx_index_key], expected_index):
                    raise AssertionError(
                        f"canonical {split} index mismatch f{fold} h{horizon}"
                    )
                if not np.array_equal(tx[tx_index_key], caper_data["target_index"]):
                    raise AssertionError(f"index mismatch {split} f{fold} h{horizon}")
                if not np.array_equal(tx[tx_target_key], caper_data["target"]):
                    raise AssertionError(f"target mismatch {split} f{fold} h{horizon}")
                source_target = rate[tx[tx_index_key].astype(np.int64)]
                if not np.allclose(source_target, tx[tx_target_key], atol=1e-7, rtol=0):
                    raise AssertionError(f"audited target mismatch {split} f{fold} h{horizon}")
                caper_prediction = np.clip(caper_data["prediction"].astype(np.float64), 0, 1)
                timexer_prediction = np.clip(tx[tx_prediction_key].astype(np.float64), 0, 1)
                target = tx[tx_target_key].astype(np.float64)
                features = np.concatenate(
                    [
                        expert_features(caper_prediction, timexer_prediction, horizon),
                        state_features(
                            rate,
                            time_index,
                            capacity,
                            tx[tx_index_key],
                            horizon,
                        ),
                    ],
                    axis=1,
                ).astype(np.float32)
                label = (
                    np.abs(timexer_prediction - target)
                    < np.abs(caper_prediction - target)
                ).astype(np.int8)
                datasets[(split, fold, horizon)] = {
                    "features": features,
                    "horizon_index": np.full(len(features), horizon_index, dtype=np.int64),
                    "caper": caper_prediction,
                    "timexer": timexer_prediction,
                    "target": target,
                    "label": label,
                    "target_index": tx[tx_index_key].astype(np.int64),
                }

    fingerprint_payload = {
        "protocol_config_sha256": _sha256(router_config_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "fixed_weights_sha256": _sha256(fixed_weights_path),
        "fixed_summary_sha256": _sha256(fixed_summary_path),
        "input_artifact_sha256": input_artifact_hashes,
        "max_samples_per_cell": max_samples_per_cell,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
    }
    run_id = "router_v1_c" + hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_root = output_dir / "runs" / run_id
    for prior_status_path in sorted(run_root.glob("attempt_*/status.json")):
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior_status.get("status") == "success":
            summary_path = prior_status_path.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "decision": "SKIPPED_DUPLICATE",
                        "run_id": run_id,
                        "artifact_root": str(prior_status_path.parent),
                    },
                    ensure_ascii=False,
                )
            )
            return summary
    attempts = sorted(run_root.glob("attempt_*"))
    run_dir = run_root / f"attempt_{len(attempts) + 1:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status_path = run_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {"status": "running", "run_id": run_id, "started_at": datetime.now().isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    failure_guard = {"completed": False}
    previous_excepthook = sys.excepthook

    def _record_failure(error_type, error, trace) -> None:
        if not failure_guard["completed"]:
            status_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "run_id": run_id,
                        "finished_at": datetime.now().isoformat(),
                        "error_type": error_type.__name__,
                        "error": str(error),
                        "traceback": "".join(
                            traceback.format_exception(error_type, error, trace)
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        previous_excepthook(error_type, error, trace)

    sys.excepthook = _record_failure
    rows = []
    histories = []
    for fold in FOLDS:
        if fold == 1:
            gbdt_keys = [("valid", 1, horizon) for horizon in HORIZONS]
            source_label = "fold1_pretest_validation"
        else:
            gbdt_keys = [
                ("test", prior_fold, horizon)
                for prior_fold in FOLDS
                if prior_fold < fold
                for horizon in HORIZONS
            ]
            source_label = f"strict_prior_fold_oof_test_1_to_{fold-1}"
        gbdt_train = _concatenate(
            [
                _sample(datasets[key], max_samples_per_cell, seed + key[1] * 100 + key[2])
                for key in gbdt_keys
            ]
        )
        gbdt = _gbdt(seed)
        gbdt.fit(gbdt_train["features"], gbdt_train["label"])

        if fold == 1:
            train_parts, selection_parts = [], []
            for horizon in HORIZONS:
                dataset = datasets[("valid", 1, horizon)]
                split = int(dataset["target"].shape[0] * 0.8)
                train_parts.append(
                    _sample(dataset, max_samples_per_cell, seed + horizon, (0, split))
                )
                selection_parts.append(
                    _sample(
                        dataset,
                        max_samples_per_cell,
                        seed + 1000 + horizon,
                        (split, dataset["target"].shape[0]),
                    )
                )
        elif fold == 2:
            train_parts, selection_parts = [], []
            for horizon in HORIZONS:
                dataset = datasets[("test", 1, horizon)]
                split = int(dataset["target"].shape[0] * 0.8)
                train_parts.append(
                    _sample(dataset, max_samples_per_cell, seed + horizon, (0, split))
                )
                selection_parts.append(
                    _sample(
                        dataset,
                        max_samples_per_cell,
                        seed + 1000 + horizon,
                        (split, dataset["target"].shape[0]),
                    )
                )
        else:
            train_parts = [
                _sample(
                    datasets[("test", prior_fold, horizon)],
                    max_samples_per_cell,
                    seed + prior_fold * 100 + horizon,
                )
                for prior_fold in FOLDS
                if prior_fold < fold - 1
                for horizon in HORIZONS
            ]
            selection_parts = [
                _sample(
                    datasets[("test", fold - 1, horizon)],
                    max_samples_per_cell,
                    seed + 1000 + (fold - 1) * 100 + horizon,
                )
                for horizon in HORIZONS
            ]
        mlp_train = _concatenate(train_parts)
        mlp_selection = _concatenate(selection_parts)
        mlp, feature_mean, feature_std, best_epoch, history = _train_mlp(
            mlp_train, mlp_selection, seed, device
        )
        torch.save(
            {
                "model_state": mlp.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "best_epoch": best_epoch,
                "seed": seed,
            },
            model_dir / f"direct_loss_mlp_fold{fold}.pt",
        )
        gbdt.booster_.save_model(str(model_dir / f"gbdt_gate_fold{fold}.txt"))
        history.insert(0, "evaluation_fold", fold)
        histories.append(history)
        for horizon in HORIZONS:
            current = datasets[("test", fold, horizon)]
            probability = gbdt.predict_proba(current["features"])[:, 1]
            hard_gate = (probability > 0.5).astype(np.float64)
            soft_gate = probability.astype(np.float64)
            mlp_gate = _predict_mlp(
                mlp,
                current["features"],
                current["horizon_index"],
                feature_mean,
                feature_std,
                device,
            ).reshape(current["target"].shape)
            hard_gate = hard_gate.reshape(current["target"].shape)
            soft_gate = soft_gate.reshape(current["target"].shape)
            caper = current["caper"]
            timexer = current["timexer"]
            target = current["target"]
            hard = np.clip(caper + hard_gate * (timexer - caper), 0, 1)
            soft = np.clip(caper + soft_gate * (timexer - caper), 0, 1)
            direct = np.clip(caper + mlp_gate * (timexer - caper), 0, 1)
            weight = fixed_weights[
                (fixed_weights.evaluation_fold == fold)
                & (fixed_weights.horizon == horizon)
            ].iloc[0]
            global_fixed = np.clip(
                caper + float(weight.global_alpha_timexer) * (timexer - caper), 0, 1
            )
            horizon_fixed = np.clip(
                caper + float(weight.horizon_alpha_timexer) * (timexer - caper), 0, 1
            )
            oracle = np.where(
                np.abs(timexer - target) < np.abs(caper - target), timexer, caper
            )
            methods = {
                "CAPER": caper,
                "TimeXer": timexer,
                "global_fixed": global_fixed,
                "horizon_fixed": horizon_fixed,
                "hard_router": hard,
                "soft_router": soft,
                "direct_loss_mlp": direct,
                "oracle": oracle,
            }
            row = {
                "fold": fold,
                "horizon": horizon,
                "samples": len(current["target_index"]),
                "training_source": source_label,
                "gbdt_AUC": float(roc_auc_score(current["label"].reshape(-1), probability)),
                "hard_gate_mean": float(hard_gate.mean()),
                "soft_gate_mean": float(soft_gate.mean()),
                "mlp_gate_mean": float(mlp_gate.mean()),
                "mlp_best_epoch": best_epoch,
                "mlp_parameters": sum(parameter.numel() for parameter in mlp.parameters()),
            }
            for method, prediction in methods.items():
                row.update(
                    {
                        f"{method}_{metric}": value
                        for metric, value in _metrics(prediction, target).items()
                    }
                )
            rows.append(row)
            np.savez_compressed(
                prediction_dir / f"router_v1_f{fold}_h{horizon}.npz",
                target_index=current["target_index"],
                target=target,
                caper=caper,
                timexer=timexer,
                global_fixed=global_fixed,
                horizon_fixed=horizon_fixed,
                hard_router=hard,
                soft_router=soft,
                direct_loss_mlp=direct,
                oracle=oracle,
                hard_gate=hard_gate,
                soft_gate=soft_gate,
                mlp_gate=mlp_gate,
                zone_ids=zone_ids,
            )

    frame = pd.DataFrame(rows)
    methods = (
        "CAPER",
        "TimeXer",
        "global_fixed",
        "horizon_fixed",
        "hard_router",
        "soft_router",
        "direct_loss_mlp",
        "oracle",
    )
    macro = pd.DataFrame(
        [
            {
                "method": method,
                "cells": len(frame),
                **{
                    metric: float(frame[f"{method}_{metric}"].mean())
                    for metric in ("RMSE", "MAE", "WAPE", "sMAPE", "RAE")
                },
            }
            for method in methods
        ]
    )
    horizon_rows = []
    for method in methods:
        for horizon in HORIZONS:
            subset = frame[frame.horizon == horizon]
            horizon_rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "cells": len(subset),
                    **{
                        metric: float(subset[f"{method}_{metric}"].mean())
                        for metric in ("RMSE", "MAE", "WAPE", "sMAPE", "RAE")
                    },
                }
            )
    by_horizon = pd.DataFrame(horizon_rows)
    fixed_rmse = float(macro.loc[macro.method == best_fixed_method, "RMSE"].iloc[0])
    router_rmse = float(macro.loc[macro.method == "direct_loss_mlp", "RMSE"].iloc[0])
    oracle_rmse = float(macro.loc[macro.method == "oracle", "RMSE"].iloc[0])
    gain = 100 * (fixed_rmse - router_rmse) / fixed_rmse
    cell_wins = int(
        (
            frame["direct_loss_mlp_RMSE"]
            < frame[f"{best_fixed_method}_RMSE"]
        ).sum()
    )
    fold_values = frame.groupby("fold", as_index=False).agg(
        router=("direct_loss_mlp_RMSE", "mean"),
        fixed=(f"{best_fixed_method}_RMSE", "mean"),
    )
    fold_wins = int((fold_values.router < fold_values.fixed).sum())
    router_horizon = by_horizon[by_horizon.method == "direct_loss_mlp"].set_index("horizon")
    fixed_horizon = by_horizon[by_horizon.method == best_fixed_method].set_index("horizon")
    deterioration = 100 * (router_horizon.RMSE - fixed_horizon.RMSE) / fixed_horizon.RMSE
    max_deterioration = float(deterioration.max())
    bootstrap_arrays = []
    for fold in FOLDS:
        for horizon in HORIZONS:
            artifact = np.load(prediction_dir / f"router_v1_f{fold}_h{horizon}.npz")
            bootstrap_arrays.append(
                (
                    artifact["direct_loss_mlp"].astype(np.float64),
                    artifact[best_fixed_method].astype(np.float64),
                    artifact["target"].astype(np.float64),
                )
            )
    bootstrap = _bootstrap(bootstrap_arrays, bootstrap_iterations, seed + 1)
    ci_lower = float(bootstrap["gain_percent_95_CI"][0])
    oracle_capture = (
        (fixed_rmse - router_rmse) / (fixed_rmse - oracle_rmse)
        if fixed_rmse > oracle_rmse
        else float("nan")
    )
    criteria = {
        "beats_TimeXer": router_rmse
        < float(macro.loc[macro.method == "TimeXer", "RMSE"].iloc[0]),
        "beats_global_fixed": router_rmse
        < float(macro.loc[macro.method == "global_fixed", "RMSE"].iloc[0]),
        "beats_horizon_fixed": router_rmse
        < float(macro.loc[macro.method == "horizon_fixed", "RMSE"].iloc[0]),
        "macro_gain_ge_1_percent": gain >= 1.0,
        "fold_wins_ge_5_of_6": fold_wins >= 5,
        "cell_wins_ge_18_of_24": cell_wins >= 18,
        "paired_block_CI_lower_gt_0": ci_lower > 0,
        "max_horizon_deterioration_le_1_percent": max_deterioration <= 1.0,
    }
    summary = {
        "status": "complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "artifact_root": str(run_dir.resolve()),
        "protocol": "router_v1",
        "sanity": sanity,
        "seed": seed,
        "target_identity_verified": True,
        "probability_calibration": "raw",
        "best_fixed_method": best_fixed_method,
        "best_fixed_RMSE": fixed_rmse,
        "direct_loss_mlp_RMSE": router_rmse,
        "direct_loss_mlp_gain_vs_best_fixed_percent": gain,
        "fold_wins": fold_wins,
        "cell_wins": cell_wins,
        "max_horizon_deterioration_percent": max_deterioration,
        "oracle_gap_capture": oracle_capture,
        "bootstrap": bootstrap,
        "gate4_criteria": criteria,
        "gate4_pass": all(criteria.values()),
        "methods": macro.to_dict(orient="records"),
        "claim_boundary": "Internal rolling evidence only; protected test remains unopened.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, value in (
        ("ROUTER_V1_CELLS", frame),
        ("ROUTER_V1_MACRO", macro),
        ("ROUTER_V1_HORIZON", by_horizon),
        ("ROUTER_V1_FOLD", fold_values),
        ("ROUTER_V1_TRAINING_HISTORY", pd.concat(histories, ignore_index=True)),
    ):
        value.to_csv(run_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        value.to_csv(output_dir / f"{name}_{stamp}.csv", index=False, encoding="utf-8-sig")
        value.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (run_dir / "summary.json").write_text(text, encoding="utf-8")
    status_path.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": run_id,
                "finished_at": datetime.now().isoformat(),
                "summary": str((run_dir / "summary.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    failure_guard["completed"] = True
    sys.excepthook = previous_excepthook
    (output_dir / f"ROUTER_V1_SUMMARY_{stamp}.json").write_text(text, encoding="utf-8")
    (output_dir / "ROUTER_V1_SUMMARY.json").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--max-samples-per-cell", type=int, default=25_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.project_root.resolve(),
        args.source_root.resolve(),
        args.output_dir.resolve(),
        args.max_samples_per_cell,
        args.bootstrap_iterations,
        args.seed,
        args.device,
    )


if __name__ == "__main__":
    main()
