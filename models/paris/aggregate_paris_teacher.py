"""Close and evaluate the strict-prequential Paris teacher after all 32 jobs."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paris_teacher_common import (
    FOLDS,
    HORIZONS,
    MODELS,
    array_hash,
    cell_contract,
    implementation_bundle_hash,
    implementation_hashes,
    load_bundle,
    load_manifests,
    load_model_config,
    masked_metrics,
    project_root,
    sha256_file,
    write_json_atomic,
)


def _fit_alpha(cells: list[dict[str, np.ndarray]], prefix: str) -> float:
    caper, timexer, target = [], [], []
    key_prefix = f"{prefix}_" if prefix else ""
    for cell in cells:
        mask = cell[f"{key_prefix}mask"]
        caper.append(cell[f"{key_prefix}caper"][mask].reshape(-1))
        timexer.append(cell[f"{key_prefix}timexer"][mask].reshape(-1))
        target.append(cell[f"{key_prefix}target"][mask].reshape(-1))
    c = np.concatenate(caper).astype(np.float64)
    t = np.concatenate(timexer).astype(np.float64)
    y = np.concatenate(target).astype(np.float64)
    delta = t - c
    denominator = float(delta @ delta)
    return float(np.clip((y - c) @ delta / denominator, 0.0, 1.0)) if denominator > 0 else 0.5


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson, float(np.corrcoef(xr, yr)[0, 1])


def _bootstrap_gain(
    cells: list[dict[str, np.ndarray]], replicates: int, seed: int
) -> tuple[float, float, np.ndarray, dict[str, int], dict[str, Any]]:
    """Fold-synchronous 24h blocks; re-select best single in every replicate."""
    rng = np.random.default_rng(seed)
    gains = np.empty(replicates, np.float64)
    fold_rules: dict[str, Any] = {}
    fold_blocks: dict[str, list[np.ndarray]] = {}
    for fold in FOLDS:
        fold_cells = [cell for cell in cells if cell["fold"] == fold]
        common = fold_cells[0]["target_time_ns"]
        for cell in fold_cells[1:]:
            common = np.intersect1d(common, cell["target_time_ns"], assume_unique=True)
        common = np.sort(common)
        if common.size < 24 or (np.diff(common) != 3_600_000_000_000).any():
            raise AssertionError(f"fold {fold} has no contiguous common hourly timestamp support")
        fold_blocks[fold] = [common[start : start + 24] for start in range(0, len(common), 24)]
        fold_rules[fold] = {
            "common_timestamp_count": int(len(common)),
            "blocks": len(fold_blocks[fold]),
            "common_start_ns": int(common[0]),
            "common_end_ns": int(common[-1]),
        }
    selection_counts = {"caper": 0, "timexer": 0}
    for repeat in range(replicates):
        rmse = {"caper": [], "timexer": [], "teacher": []}
        selected_by_fold: dict[str, np.ndarray] = {}
        for fold in FOLDS:
            blocks = fold_blocks[fold]
            selected_by_fold[fold] = np.concatenate(
                [blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))]
            )
        for cell in cells:
            sampled = np.searchsorted(cell["target_time_ns"], selected_by_fold[cell["fold"]])
            if (sampled >= len(cell["target_time_ns"])).any() or not np.array_equal(
                cell["target_time_ns"][sampled], selected_by_fold[cell["fold"]]
            ):
                raise AssertionError("synchronized bootstrap timestamp mapping failed")
            mask = cell["mask"][sampled]
            target = cell["target"][sampled][mask]
            for method in rmse:
                prediction = cell[method][sampled][mask]
                rmse[method].append(float(np.sqrt(np.mean((prediction - target) ** 2))))
        macro = {method: float(np.mean(values)) for method, values in rmse.items()}
        selected = "caper" if macro["caper"] <= macro["timexer"] else "timexer"
        selection_counts[selected] += 1
        gains[repeat] = 100.0 * (macro[selected] - macro["teacher"]) / macro[selected]
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return float(lower), float(upper), gains, selection_counts, fold_rules


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_model_config(root)
    folds, runs = load_manifests(root)
    bundle = load_bundle(root)
    frozen_hashes = json.loads(str(runs.implementation_hashes_json.iloc[0]))
    current_hashes = implementation_hashes(root)
    amendment_path = root / "experiments/09_distill_v2/PARIS_TEACHER_AGGREGATION_AMENDMENT_V1_1_1.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    other_match = all(current_hashes[name] == digest for name, digest in frozen_hashes.items() if name != "aggregate_paris_teacher.py")
    amendment_match = amendment.get("old_aggregate_sha256") == frozen_hashes["aggregate_paris_teacher.py"] and amendment.get("new_aggregate_sha256") == current_hashes["aggregate_paris_teacher.py"] and amendment.get("predictions_changed") is False and amendment.get("training_changed") is False
    if not (runs.implementation_bundle_hash.nunique() == 1 and other_match and amendment_match):
        raise RuntimeError("aggregate implementation/amendment does not match v1.1 training manifest")
    queue_root = root / "experiments/09_distill_v2/outputs/teacher_qualification/queue"
    state_path = queue_root / "queue_state.json"
    closure_path = queue_root / "QUEUE_CLOSED.json"
    if not state_path.exists() or not closure_path.exists():
        raise RuntimeError("metrics remain locked until the persistent queue closes")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if state.get("status") != "closed_success" or state.get("metrics_unlocked") is not True or closure.get("jobs") != 32:
        raise RuntimeError("32-cell closure barrier not satisfied")
    artifacts = {job["run_id"]: Path(job["artifact"]) for job in state["jobs"] if job.get("status") == "success"}
    if len(artifacts) != 32:
        raise RuntimeError("queue artifact map is incomplete")
    loaded: dict[tuple[str, int, str], dict[str, Any]] = {}
    identity_pass = True
    for row in runs.itertuples(index=False):
        run_dir = artifacts.get(row.run_id)
        if run_dir is None:
            raise RuntimeError(f"missing artifact: {row.run_id}")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        receipt = json.loads((run_dir / "SUCCESS_RECEIPT.json").read_text(encoding="utf-8"))
        if not receipt.get("artifact_sha256") or not all(
            (run_dir / name).exists() and sha256_file(run_dir / name) == digest
            for name, digest in receipt["artifact_sha256"].items()
        ):
            raise AssertionError(f"success receipt hash drift: {row.run_id}")
        prediction_path = run_dir / "predictions.npz"
        if (
            status.get("status") != "success"
            or status.get("evaluation_metrics_computed") is not False
            or payload.get("manifest_fingerprint") != row.fingerprint
        ):
            raise AssertionError(f"status/config drift: {row.run_id}")
        if status.get("prediction_sha256") != sha256_file(prediction_path):
            raise AssertionError(f"prediction hash drift: {row.run_id}")
        prediction = np.load(prediction_path, allow_pickle=False)
        fold_row = folds[(folds.fold == row.fold) & (folds.horizon == int(row.horizon))]
        if len(fold_row) != 1:
            raise AssertionError("fold manifest lookup failure")
        contract = cell_contract(bundle, fold_row.iloc[0], 168)
        expected_indices = contract["evaluation_targets"]
        expected_mask = bundle.observed_mask[expected_indices]
        expected_target = bundle.rate_observed[expected_indices]
        if not np.array_equal(prediction["target_index"], expected_indices):
            raise AssertionError(f"development target index replay drift: {row.run_id}")
        if not np.array_equal(prediction["mask"].astype(bool), expected_mask):
            raise AssertionError(f"development target mask replay drift: {row.run_id}")
        if not np.allclose(prediction["target"][expected_mask], expected_target[expected_mask], atol=1e-7, rtol=0.0):
            raise AssertionError(f"development target value replay drift: {row.run_id}")
        for key in ("eval_target_value_hash", "eval_mask_hash", "fill_hash", "scaler_hash", "fold_capacity_hash"):
            if str(prediction[key].item()) != contract["identity"][key] or payload["identity"][key] != contract["identity"][key]:
                raise AssertionError(f"development {key} replay drift: {row.run_id}")
        expected_identity = {
            "forecast_origin_hash": row.forecast_origin_hash,
            "target_timestamp_hash": row.target_timestamp_hash,
            "station_order_hash": row.station_order_hash,
            "mask_hash": row.mask_hash,
        }
        identity_pass &= all(payload["identity"].get(key) == value for key, value in expected_identity.items())
        identity_pass &= array_hash(prediction["target_time_ns"].astype("<i8")) == row.target_timestamp_hash
        identity_pass &= array_hash(prediction["origin_time_ns"].astype("<i8")) == row.forecast_origin_hash
        identity_pass &= array_hash(prediction["station_ids"].astype("U")) == row.station_order_hash
        loaded[(str(row.fold), int(row.horizon), str(row.model))] = {
            "artifact": prediction,
            "config": payload,
            "run_dir": run_dir,
        }
    cells_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        for horizon in HORIZONS:
            caper_entry = loaded[(fold, horizon, "Paris_CAPER_phase_only")]
            timexer_entry = loaded[(fold, horizon, "TimeXer_local_audited_compact_L168")]
            caper, timexer = caper_entry["artifact"], timexer_entry["artifact"]
            for key in ("target", "mask", "target_index", "target_time_ns", "origin_time_ns", "station_ids"):
                same = np.allclose(caper[key], timexer[key], equal_nan=True, atol=0.0, rtol=0.0) if key == "target" else np.array_equal(caper[key], timexer[key])
                if not same:
                    raise AssertionError(f"paired evaluation identity mismatch {fold} h{horizon}: {key}")
            for key in ("validation_target", "validation_mask", "validation_target_index", "validation_target_time_ns", "validation_origin_time_ns"):
                same = np.allclose(caper[key], timexer[key], equal_nan=True, atol=0.0, rtol=0.0) if key == "validation_target" else np.array_equal(caper[key], timexer[key])
                if not same:
                    raise AssertionError(f"paired validation identity mismatch {fold} h{horizon}: {key}")
            if caper_entry["config"]["identity"]["shared_contract_hash"] != timexer_entry["config"]["identity"]["shared_contract_hash"]:
                raise AssertionError(f"shared fill/scaler contract mismatch {fold} h{horizon}")
            cells_by_key[(fold, horizon)] = {
                "caper": caper["clipped_prediction"].astype(np.float64),
                "timexer": timexer["clipped_prediction"].astype(np.float64),
                "target": caper["target"].astype(np.float64),
                "mask": caper["mask"].astype(bool),
                "valid_caper": caper["validation_clipped_prediction"].astype(np.float64),
                "valid_timexer": timexer["validation_clipped_prediction"].astype(np.float64),
                "valid_target": caper["validation_target"].astype(np.float64),
                "valid_mask": caper["validation_mask"].astype(bool),
                "target_index": caper["target_index"].astype(np.int64),
                "target_time_ns": caper["target_time_ns"].astype("<i8"),
                "origin_time_ns": caper["origin_time_ns"].astype("<i8"),
                "station_ids": caper["station_ids"].astype("U"),
            }
    weights: list[dict[str, Any]] = []
    ordered_cells: list[dict[str, np.ndarray]] = []
    for fold_index, fold in enumerate(FOLDS):
        if fold_index == 0:
            fit_cells = [cells_by_key[(fold, horizon)] for horizon in HORIZONS]
            prefix, source = "valid", "fold1_pre_evaluation_validation_all_horizons"
        else:
            fit_cells = [cells_by_key[(prior, horizon)] for prior in FOLDS[:fold_index] for horizon in HORIZONS]
            prefix, source = "", f"strict_prior_fold_oof_1_to_{fold_index}"
        alpha = _fit_alpha(fit_cells, prefix)
        weights.append({"fold": fold, "alpha_TimeXer": alpha, "alpha_CAPER": 1.0 - alpha, "fit_source": source, "fit_cells": len(fit_cells)})
        for horizon in HORIZONS:
            cell = cells_by_key[(fold, horizon)]
            teacher = np.clip(cell["caper"] + alpha * (cell["timexer"] - cell["caper"]), 0.0, 1.0)
            oracle = np.where(np.abs(cell["timexer"] - cell["target"]) < np.abs(cell["caper"] - cell["target"]), cell["timexer"], cell["caper"])
            cell["teacher"], cell["oracle"] = teacher, oracle
            cell["fold"], cell["horizon"] = fold, horizon
            ordered_cells.append(cell)
    metric_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    teacher_dir = root / "experiments/09_distill_v2/outputs/teacher_qualification/teacher_cells"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    for cell in ordered_cells:
        for method in ("caper", "timexer", "teacher", "oracle"):
            metric_rows.append({"fold": cell["fold"], "horizon": cell["horizon"], "method": method, **masked_metrics(cell[method], cell["target"], cell["mask"])})
        observed = cell["mask"]
        residual_caper = (cell["caper"] - cell["target"])[observed]
        residual_timexer = (cell["timexer"] - cell["target"])[observed]
        pearson, spearman = _corr(residual_caper, residual_timexer)
        correlation_rows.append({"fold": cell["fold"], "horizon": cell["horizon"], "pearson": pearson, "spearman": spearman, "observed_cells": int(observed.sum())})
        np.savez_compressed(
            teacher_dir / f"teacher_{cell['fold']}_h{cell['horizon']}.npz",
            prediction=cell["teacher"].astype(np.float32),
            target=cell["target"].astype(np.float32),
            mask=cell["mask"].astype(np.uint8),
            target_index=cell["target_index"],
            target_time_ns=cell["target_time_ns"],
            origin_time_ns=cell["origin_time_ns"],
            station_ids=cell["station_ids"],
        )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby("method", as_index=False)[["RMSE", "MAE", "WAPE", "sMAPE", "RAE"]].mean()
    single = macro[macro.method.isin(["caper", "timexer"])].sort_values("RMSE").iloc[0]
    reference = str(single.method)
    teacher_macro = float(macro.loc[macro.method == "teacher", "RMSE"].iloc[0])
    reference_macro = float(single.RMSE)
    macro_gain = 100.0 * (reference_macro - teacher_macro) / reference_macro
    folds_frame = metrics[metrics.method.isin([reference, "teacher"])].groupby(["fold", "method"], as_index=False).RMSE.mean().pivot(index="fold", columns="method", values="RMSE").reset_index()
    folds_frame["teacher_wins"] = folds_frame["teacher"] < folds_frame[reference]
    horizons_frame = metrics[metrics.method.isin([reference, "teacher"])].groupby(["horizon", "method"], as_index=False).RMSE.mean().pivot(index="horizon", columns="method", values="RMSE").reset_index()
    horizons_frame["teacher_deterioration_percent"] = 100.0 * (horizons_frame["teacher"] - horizons_frame[reference]) / horizons_frame[reference]
    paired = metrics[metrics.method.isin([reference, "teacher"])].pivot(index=["fold", "horizon"], columns="method", values="RMSE").reset_index()
    paired["teacher_wins"] = paired["teacher"] < paired[reference]
    lower, upper, bootstrap, bootstrap_selection, bootstrap_rules = _bootstrap_gain(
        ordered_cells,
        int(config["aggregation"]["bootstrap_replicates"]),
        int(config["aggregation"]["bootstrap_seed"]),
    )
    gate_spec = json.loads((root / "experiments/09_distill_v2/PARIS_TEACHER_GATE_SPEC.json").read_text(encoding="utf-8"))
    gate = {
        "macro_gain_pass": macro_gain >= float(gate_spec["macro_RMSE_gain_percent_min"]),
        "fold_wins_pass": int(folds_frame.teacher_wins.sum()) >= int(gate_spec["fold_wins_min"]),
        "cell_wins_pass": int(paired.teacher_wins.sum()) >= int(gate_spec["cell_wins_min"]),
        "bootstrap_CI_pass": lower > float(gate_spec["paired_24h_block_CI_lower_gt"]),
        "horizon_protection_pass": float(horizons_frame.teacher_deterioration_percent.max()) <= float(gate_spec["max_horizon_deterioration_percent"]),
        "identity_pass": bool(identity_pass),
    }
    qualified = all(gate.values())
    out = root / "experiments/09_distill_v2/outputs/teacher_qualification/aggregate"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames = {
        "PARIS_TEACHER_CELLS": metrics,
        "PARIS_TEACHER_MACRO": macro,
        "PARIS_TEACHER_FOLDS": folds_frame,
        "PARIS_TEACHER_HORIZONS": horizons_frame,
        "PARIS_TEACHER_WEIGHTS": pd.DataFrame(weights),
        "PARIS_TEACHER_RESIDUAL_CORRELATION": pd.DataFrame(correlation_rows),
    }
    for name, frame in frames.items():
        _write_csv(out / f"{name}_{stamp}.csv", frame)
        _write_csv(out / f"{name}.csv", frame)
    np.save(out / f"PARIS_TEACHER_BOOTSTRAP_GAINS_{stamp}.npy", bootstrap)
    summary = {
        "status": "PASS" if qualified else "FAIL",
        "qualified_teacher": qualified,
        "failure_policy_if_failed": gate_spec["failure_policy"],
        "reference_best_single": reference,
        "reference_macro_RMSE": reference_macro,
        "teacher_macro_RMSE": teacher_macro,
        "teacher_gain_percent": macro_gain,
        "fold_wins": int(folds_frame.teacher_wins.sum()),
        "cell_wins": int(paired.teacher_wins.sum()),
        "paired_24h_block_CI_gain_percent": [lower, upper],
        "bootstrap_best_single_selection_counts": bootstrap_selection,
        "bootstrap_common_timestamp_rules": bootstrap_rules,
        "max_horizon_deterioration_percent": float(horizons_frame.teacher_deterioration_percent.max()),
        "gate": gate,
        "cells": 16,
        "model_jobs": 32,
        "student_training_authorized": bool(qualified),
        "formal_target_access": False,
        "protected_target_access": False,
        "queue_state_sha256": sha256_file(state_path),
    }
    write_json_atomic(out / f"PARIS_TEACHER_QUALIFICATION_SUMMARY_{stamp}.json", summary)
    write_json_atomic(out / "PARIS_TEACHER_QUALIFICATION_SUMMARY.json", summary)
    report = "\n".join(
        [
            "# Paris development teacher qualification",
            "",
            f"Decision: **{'PASS' if qualified else 'FAIL'}**.",
            f"Best single: `{reference}` (macro RMSE {reference_macro:.6f}); strict teacher: {teacher_macro:.6f}; gain {macro_gain:.3f}%.",
            f"Fold wins: {int(folds_frame.teacher_wins.sum())}/4; cell wins: {int(paired.teacher_wins.sum())}/16; paired 24h-block CI: [{lower:.3f}%, {upper:.3f}%].",
            f"Maximum horizon deterioration: {float(horizons_frame.teacher_deterioration_percent.max()):.3f}%.",
            "",
            "Formal and protected targets were not accessed. Student training is authorized only when all six frozen gates pass.",
        ]
    )
    (out / f"PARIS_TEACHER_QUALIFICATION_REPORT_{stamp}.md").write_text(report, encoding="utf-8")
    (out / "PARIS_TEACHER_QUALIFICATION_REPORT.md").write_text(report, encoding="utf-8")
    write_json_atomic(out / "METRICS_CLOSED.json", {"status": "PASS", "closed_jobs": 32, "summary_sha256": sha256_file(out / "PARIS_TEACHER_QUALIFICATION_SUMMARY.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
