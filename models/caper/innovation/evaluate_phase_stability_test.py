"""One-shot formal-test evaluation for the frozen phase stability matrix.

The command refuses to construct test targets unless the complete validation
matrix has been aggregated and the caller supplies the exact frozen queue id.
No model is retrained and no test-dependent selection is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from innovation.canonical import canonical_boundaries, canonical_target_indices
from innovation.deep_baselines import target_index_hash
from innovation.hybrid_expert_experiment import evaluate
from innovation.multires_state import load_five_minute_states
from innovation.physical_constraints import load_physical_data
from innovation.physical_multitask import fit_physical_scales, set_deterministic
from innovation.state_ordered_gate_experiment import StatePhaseDataset
from innovation.stock_flow_error_correction_experiment import (
    Config,
    StockFlowErrorCorrectionModel,
)
from repro.metrics import audited_metrics, official_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _attempt(root: Path) -> Path:
    candidates = [
        path for path in sorted(root.glob("attempt_*"))
        if (path / "status.json").exists()
        and json.loads((path / "status.json").read_text(encoding="utf-8"))["status"] == "success"
    ]
    if not candidates:
        raise RuntimeError(f"no successful validation attempt: {root}")
    return candidates[-1]


def _config(payload: dict, device: str) -> Config:
    allowed = {item.name for item in fields(Config)}
    values = {key: value for key, value in payload.items() if key in allowed}
    values["device"] = device
    return Config(**values)


def run(root: Path, source_root: Path, queue_id: str, confirmation: str, device: str) -> Path:
    if confirmation != queue_id:
        raise RuntimeError("formal test confirmation does not match the frozen queue id")
    decision_path = root / "innovation" / "phase_stability_results" / "PHASE_STABILITY_DECISION.json"
    if not decision_path.exists():
        raise RuntimeError("aggregate the complete validation matrix before formal test")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("complete") or decision.get("queue_id") != queue_id:
        raise RuntimeError("validation matrix is incomplete or not the confirmed queue")
    if not decision.get("retain_over_2_percent"):
        raise RuntimeError("the frozen candidate failed the preregistered retention gate")
    state = json.loads(
        (root / "innovation" / "phase_stability_queues" / queue_id / "state.json").read_text(
            encoding="utf-8"
        )
    )
    if state["counts"].get("failed", 0) != 0 or len(state["jobs"]) != state["total"]:
        raise RuntimeError("queue contains failures or unrecorded jobs")

    data = load_physical_data(root / "audited" / "data")
    fine = load_five_minute_states(root, source_root, data)
    state_hour = fine.state_rate[::12].astype(np.float32)
    output = root / "innovation" / "phase_stability_test_results"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in state["jobs"]:
        if job["variant"] != "phase_only" and job["seed"] != 42:
            continue
        attempt = _attempt(root / "innovation" / "stock_flow_runs" / job["run_id"])
        payload = json.loads((attempt / "config.json").read_text(encoding="utf-8"))
        config = _config(payload, device)
        if not config.causal_auxiliary_history:
            raise AssertionError("formal candidate is not strict-causal")
        set_deterministic(config.seed)
        bounds = canonical_boundaries(data.time, config.fold)
        scales = fit_physical_scales(data, bounds.train_end)
        test_index = canonical_target_indices(
            bounds, config.horizon, "test", config.common_history_budget
        )
        dataset = StatePhaseDataset(
            data, scales, state_hour, test_index, config.dataset_config()
        )
        coefficients = payload["train_only_coefficients"]
        model = StockFlowErrorCorrectionModel(data, scales, config, coefficients)
        model.load_state_dict(torch.load(attempt / "checkpoint.pt", map_location=device))
        model.to(torch.device(device))
        result = evaluate(model, dataset, config)
        run_output = output / job["run_id"]
        run_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_output / "test_predictions.npz",
            **result,
            zone_ids=np.asarray(data.zone_ids),
        )
        for semantics, metric in (("audited", audited_metrics), ("official", official_metrics)):
            rows.append(
                {
                    "run_id": job["run_id"],
                    "variant": config.variant,
                    "fold": config.fold,
                    "horizon": config.horizon,
                    "seed": config.seed,
                    "split": "formal_test",
                    "metric_semantics": semantics,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "test_target_hash": target_index_hash(test_index),
                    **metric(result["prediction"], result["target"]),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "PHASE_STABILITY_FORMAL_TEST_RUNS.csv", index=False)
    audited = metrics[metrics.metric_semantics == "audited"]
    summary = (
        audited.groupby(["variant", "horizon"], as_index=False)
        .agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
             MAE_mean=("MAE", "mean"), runs=("RMSE", "size"))
    )
    summary.to_csv(output / "PHASE_STABILITY_FORMAL_TEST_SUMMARY.csv", index=False)
    manifest = {
        "status": "complete",
        "queue_id": queue_id,
        "confirmation": confirmation,
        "data_source": "real_UrbanEV_release_from_user_workspace",
        "validation_decision_sha256": _sha256(decision_path),
        "no_test_dependent_training_or_selection": True,
        "formal_test_arrays_loaded": True,
        "runs": int(len(audited)),
    }
    (output / "FORMAL_TEST_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\UrbanEV\UrbanEV-main"))
    parser.add_argument("--queue-id", default="strict_causal_phase_stability_v1")
    parser.add_argument("--confirm-frozen-matrix", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.root.resolve(), args.source_root.resolve(), args.queue_id,
        args.confirm_frozen_matrix, args.device,
    )


if __name__ == "__main__":
    main()
