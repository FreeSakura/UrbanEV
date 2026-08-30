"""Evaluate strict prequential global- and horizon-fixed CAPER/TimeXer fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = (1, 2, 3, 4, 5, 6)
HORIZONS = (3, 6, 9, 12)


def _load_helpers(project_root: Path):
    helper_dir = project_root / "experiments" / "01_expert_diagnostics" / "scripts"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from analyze_existing_experts import _caper_artifact, _timexer_artifacts

    return _caper_artifact, _timexer_artifacts


def _latest_successful_attempt(run_root: Path) -> Path:
    candidates = [
        path
        for path in sorted(run_root.glob("attempt_*"))
        if (path / "status.json").is_file()
        and json.loads((path / "status.json").read_text(encoding="utf-8")).get(
            "status"
        )
        == "success"
    ]
    if not candidates:
        raise RuntimeError(f"no successful CAPER validation attempt: {run_root}")
    return candidates[-1]


def _caper_validation(source_root: Path, fold: int, horizon: int) -> Path:
    run_id = f"strict_causal_phase_stability_v1_phase_only_f{fold}_h{horizon}_s42"
    run_root = (
        source_root
        / "innovation"
        / "stock_flow_runs"
        / run_id
    )
    path = _latest_successful_attempt(run_root) / "validation_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _fit_alpha(caper: list[np.ndarray], timexer: list[np.ndarray], target: list[np.ndarray]) -> float:
    c = np.concatenate([value.reshape(-1) for value in caper]).astype(np.float64)
    t = np.concatenate([value.reshape(-1) for value in timexer]).astype(np.float64)
    y = np.concatenate([value.reshape(-1) for value in target]).astype(np.float64)
    delta = t - c
    denominator = float(np.dot(delta, delta))
    if denominator <= 0:
        return 0.5
    return float(np.clip(np.dot(y - c, delta) / denominator, 0.0, 1.0))


def run(project_root: Path, source_root: Path, output_dir: Path) -> dict[str, object]:
    caper_artifact, timexer_artifacts_fn = _load_helpers(project_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from innovation.canonical import canonical_boundaries, canonical_target_indices
    from innovation.data import load_occupancy

    audited_data = load_occupancy(source_root / "audited" / "data")
    tx_paths = timexer_artifacts_fn(source_root)
    expected = {(fold, horizon) for fold in FOLDS for horizon in HORIZONS}
    if set(tx_paths) != expected:
        raise RuntimeError("incomplete TimeXer matrix")
    cells: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    artifact_rows = []
    for fold, horizon in sorted(expected):
        tx_path = tx_paths[(fold, horizon)]
        caper_test_path = caper_artifact(source_root, fold, horizon)
        caper_valid_path = _caper_validation(source_root, fold, horizon)
        tx = np.load(tx_path, allow_pickle=False)
        caper_test = np.load(caper_test_path, allow_pickle=False)
        caper_valid = np.load(caper_valid_path, allow_pickle=False)
        boundaries = canonical_boundaries(audited_data.time, fold)
        expected_validation_index = canonical_target_indices(
            boundaries, horizon, "valid", common_history_budget=168
        )
        expected_test_index = canonical_target_indices(
            boundaries, horizon, "test", common_history_budget=168
        )
        if not np.array_equal(tx["validation_target_index"], expected_validation_index):
            raise AssertionError(f"canonical validation index mismatch f{fold} h{horizon}")
        if not np.array_equal(tx["target_index"], expected_test_index):
            raise AssertionError(f"canonical test index mismatch f{fold} h{horizon}")
        source_validation_target = audited_data.rate[expected_validation_index]
        source_test_target = audited_data.rate[expected_test_index]
        if not np.allclose(
            source_validation_target,
            tx["validation_target"].astype(np.float64),
            atol=1e-7,
            rtol=0.0,
        ):
            raise AssertionError(f"audited validation target mismatch f{fold} h{horizon}")
        if not np.allclose(
            source_test_target,
            tx["target"].astype(np.float64),
            atol=1e-7,
            rtol=0.0,
        ):
            raise AssertionError(f"audited test target mismatch f{fold} h{horizon}")
        for tx_key, caper_key in (
            ("target_index", "target_index"),
            ("target", "target"),
            ("zone_ids", "zone_ids"),
        ):
            if not np.array_equal(tx[tx_key], caper_test[caper_key]):
                raise AssertionError(f"test identity mismatch f{fold} h{horizon}: {tx_key}")
        for tx_key, caper_key in (
            ("validation_target_index", "target_index"),
            ("validation_target", "target"),
            ("zone_ids", "zone_ids"),
        ):
            if not np.array_equal(tx[tx_key], caper_valid[caper_key]):
                raise AssertionError(f"validation identity mismatch f{fold} h{horizon}: {tx_key}")
        cells[(fold, horizon)] = {
            "test_caper": np.clip(caper_test["prediction"].astype(np.float64), 0, 1),
            "test_timexer": np.clip(tx["clipped_prediction"].astype(np.float64), 0, 1),
            "test_target": tx["target"].astype(np.float64),
            "test_index": tx["target_index"].astype(np.int64),
            "valid_caper": np.clip(caper_valid["prediction"].astype(np.float64), 0, 1),
            "valid_timexer": np.clip(tx["validation_clipped_prediction"].astype(np.float64), 0, 1),
            "valid_target": tx["validation_target"].astype(np.float64),
            "valid_index": tx["validation_target_index"].astype(np.int64),
        }
        for expert, path in (
            ("TimeXer", tx_path),
            ("CAPER_test", caper_test_path),
            ("CAPER_validation", caper_valid_path),
        ):
            artifact_rows.append(
                {
                    "fold": fold,
                    "horizon": horizon,
                    "artifact": expert,
                    "path": str(path),
                    "sha256": _sha256(path),
                }
            )

    result_rows = []
    weight_rows = []
    for fold in FOLDS:
        if fold == 1:
            fit_keys = [(fold, horizon) for horizon in HORIZONS]
            source_prefix = "fold1_pretest_validation"
            prefix = "valid"
        else:
            fit_keys = [
                (prior_fold, horizon)
                for prior_fold in FOLDS
                if prior_fold < fold
                for horizon in HORIZONS
            ]
            source_prefix = f"strict_prior_fold_oof_test_1_to_{fold-1}"
            prefix = "test"
        global_alpha = _fit_alpha(
            [cells[key][f"{prefix}_caper"] for key in fit_keys],
            [cells[key][f"{prefix}_timexer"] for key in fit_keys],
            [cells[key][f"{prefix}_target"] for key in fit_keys],
        )
        for horizon in HORIZONS:
            horizon_keys = [key for key in fit_keys if key[1] == horizon]
            horizon_alpha = _fit_alpha(
                [cells[key][f"{prefix}_caper"] for key in horizon_keys],
                [cells[key][f"{prefix}_timexer"] for key in horizon_keys],
                [cells[key][f"{prefix}_target"] for key in horizon_keys],
            )
            current = cells[(fold, horizon)]
            caper = current["test_caper"]
            timexer = current["test_timexer"]
            target = current["test_target"]
            global_prediction = np.clip(caper + global_alpha * (timexer - caper), 0, 1)
            horizon_prediction = np.clip(caper + horizon_alpha * (timexer - caper), 0, 1)
            oracle = np.where(
                np.abs(timexer - target) < np.abs(caper - target), timexer, caper
            )
            predictions = {
                "CAPER": caper,
                "TimeXer": timexer,
                "global_fixed": global_prediction,
                "horizon_fixed": horizon_prediction,
                "oracle": oracle,
            }
            metrics = {name: _metrics(prediction, target) for name, prediction in predictions.items()}
            result_rows.append(
                {
                    "fold": fold,
                    "horizon": horizon,
                    "samples": len(current["test_index"]),
                    "fit_source": source_prefix,
                    "global_alpha_timexer": global_alpha,
                    "horizon_alpha_timexer": horizon_alpha,
                    **{
                        f"{name}_{metric}": value
                        for name, values in metrics.items()
                        for metric, value in values.items()
                    },
                }
            )
            weight_rows.append(
                {
                    "evaluation_fold": fold,
                    "horizon": horizon,
                    "fit_source": source_prefix,
                    "fit_cells": len(horizon_keys),
                    "global_alpha_timexer": global_alpha,
                    "horizon_alpha_timexer": horizon_alpha,
                }
            )

    frame = pd.DataFrame(result_rows)
    weights = pd.DataFrame(weight_rows)
    methods = ("CAPER", "TimeXer", "global_fixed", "horizon_fixed", "oracle")
    macro_rows = []
    horizon_rows = []
    for method in methods:
        macro_rows.append(
            {
                "method": method,
                "cells": len(frame),
                **{
                    metric: float(frame[f"{method}_{metric}"].mean())
                    for metric in ("RMSE", "MAE", "WAPE", "sMAPE", "RAE")
                },
            }
        )
        for horizon in HORIZONS:
            subset = frame[frame["horizon"] == horizon]
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
    macro = pd.DataFrame(macro_rows)
    by_horizon = pd.DataFrame(horizon_rows)
    fixed = macro[macro["method"].isin(["global_fixed", "horizon_fixed"])].sort_values("RMSE")
    best_fixed = fixed.iloc[0]
    timexer_rmse = float(macro.loc[macro.method == "TimeXer", "RMSE"].iloc[0])
    summary = {
        "status": "complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paired_cells": len(frame),
        "target_identity_verified": True,
        "weight_exposure": "fold1 pre-test validation; folds2-6 strict prior-fold OOF test only",
        "best_fixed_method": str(best_fixed.method),
        "best_fixed_macro_RMSE": float(best_fixed.RMSE),
        "best_fixed_gain_vs_timexer_percent": 100
        * (timexer_rmse - float(best_fixed.RMSE))
        / timexer_rmse,
        "methods": macro.to_dict(orient="records"),
        "gate4_reference_frozen": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, value in (
        ("FIXED_FUSION_CELLS", frame),
        ("FIXED_FUSION_WEIGHTS", weights),
        ("FIXED_FUSION_HORIZON", by_horizon),
        ("FIXED_FUSION_MACRO", macro),
        ("INPUT_ARTIFACT_HASHES", pd.DataFrame(artifact_rows)),
    ):
        value.to_csv(output_dir / f"{name}_{stamp}.csv", index=False, encoding="utf-8-sig")
        value.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (output_dir / f"FIXED_FUSION_SUMMARY_{stamp}.json").write_text(text, encoding="utf-8")
    (output_dir / "FIXED_FUSION_SUMMARY.json").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    run(args.project_root.resolve(), args.source_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
