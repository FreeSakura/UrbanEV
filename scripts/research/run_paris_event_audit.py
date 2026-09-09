#!/usr/bin/env python3
"""Audit an explicitly supplied Paris DEVELOPMENT shard; export aggregates only."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urbanev_audit.paired_events import event_bounds, joint_brier_bounds


def prefix_csv(path: Path):
    with path.open("rb") as handle:
        payload = b"".join(handle.readline() for _ in range(3625))
    frame = pd.read_csv(io.BytesIO(payload), index_col=0, parse_dates=True)
    expected = pd.date_range("2020-07-03", periods=3624, freq="h")
    if frame.shape != (3624, 50) or not frame.index.equals(expected):
        raise ValueError("Only the registered 3624x50 development prefix is allowed")
    return frame, hashlib.sha256(payload).hexdigest()


def run(development: Path, output: Path):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    rate, rate_hash = prefix_csv(development / "non_available_port_rate_observed.csv")
    mask_frame, mask_hash = prefix_csv(development / "observation_mask.csv")
    if list(rate.columns) != list(mask_frame.columns):
        raise ValueError("Rate and mask station order differs")
    raw_mask = mask_frame.to_numpy()
    if not np.isin(raw_mask, (0, 1, False, True)).all():
        raise ValueError("Mask must contain Boolean or 0/1 values")
    mask, values = raw_mask.astype(bool), rate.to_numpy(float)
    if not np.array_equal(np.isfinite(values), mask):
        raise ValueError("Missing values must remain missing, with an identical mask")
    if ((values[mask] < 0) | (values[mask] > 1)).any():
        raise ValueError("Observed rates violate their measurement contract")
    output.mkdir(parents=True)
    protocol = {
        "id": "PARIS_NATURAL_MISSING_PAIRED_AUDIT_V3_20260909",
        "evidence": "retrospective_development_only",
        "window": 4, "run_length": 2, "threshold": .8,
        "train_stop": "2020-09-01", "evaluation_stop_exclusive": "2020-12-01",
        "smoothing_pseudocount": 20.,
        "rate_prefix_sha256": rate_hash, "mask_prefix_sha256": mask_hash,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_code_sha256": hashlib.sha256((ROOT / "src/urbanev_audit/paired_events.py").read_bytes()).hexdigest(),
        "formal_or_protected_access": False,
    }
    (output / "protocol_receipt.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    threshold, K, L = .8, 4, 2
    observation = np.where(mask, (values >= threshold).astype(np.int8), -1)
    stop = int(np.searchsorted(rate.index, pd.Timestamp("2020-09-01")))
    train_starts = np.arange(1, stop-K+1)
    eval_starts = np.arange(stop, len(rate)-K+1)
    lower = np.empty((len(rate)-K+1, 50), bool)
    upper = np.empty_like(lower)
    for station in range(50):
        lower[:, station], upper[:, station] = event_bounds(observation[:, station], K, L)
    global_lo = float(lower[train_starts].mean())
    global_hi = float(upper[train_starts].mean())
    station_index = np.arange(50)[None, :]
    def contexts(starts):
        previous = observation[starts-1]
        state = np.where(previous == -1, 2, previous)
        time_bin = np.asarray(rate.index[starts-1].hour//4)[:, None]
        return station_index*18 + state*6 + time_bin
    context = contexts(train_starts)
    counts = np.bincount(context.ravel(), minlength=900)
    sums_lo = np.bincount(context.ravel(), weights=lower[train_starts].ravel(), minlength=900)
    sums_hi = np.bincount(context.ravel(), weights=upper[train_starts].ravel(), minlength=900)
    conditional_lo = (sums_lo+20*global_lo)/(counts+20)
    conditional_hi = (sums_hi+20*global_hi)/(counts+20)
    a = np.full((len(eval_starts), 50), (global_lo+global_hi)/2)
    b = ((conditional_lo+conditional_hi)/2)[contexts(eval_starts)]
    # Forecasts are fixed before examining evaluation labels or bounds.
    prediction_hash = hashlib.sha256(np.ascontiguousarray(np.stack((a, b))).tobytes()).hexdigest()
    joint_lo = joint_hi = ind_lo = ind_hi = 0.
    queries = []
    improved_stations = 0
    for station in range(50):
        result = joint_brier_bounds(observation[stop:, station], a[:, station], b[:, station], K, L, query_planning=True)
        joint_lo += result["lower_sum"]
        joint_hi += result["upper_sum"]
        ind_lo += result["independent_lower_sum"]
        ind_hi += result["independent_upper_sum"]
        iw = result["independent_upper_sum"]-result["independent_lower_sum"]
        jw = result["upper_sum"]-result["lower_sum"]
        improved_stations += iw-jw > 1e-10
        # Never serialize candidate timestamps, station IDs, or hidden values.
        queries.extend(q["guaranteed_width_reduction_sum"] for q in result["query_planning"])
    total = a.size
    eval_lo, eval_hi = lower[eval_starts], upper[eval_starts]
    identified = eval_lo == eval_hi
    difference = (eval_lo.astype(float)-a)**2-(eval_lo.astype(float)-b)**2
    naive = float(difference[identified].mean()) if identified.any() else None
    independent_width = (ind_hi-ind_lo)/total
    joint_width = (joint_hi-joint_lo)/total
    summary = {
        "protocol_id": protocol["id"], "scope": "real_observation_mask_development_finite_panel",
        "source_shape": [3624, 50], "source_missing_snapshots": int((~mask).sum()),
        "evaluation_snapshots": int(mask[stop:].size), "evaluation_missing_snapshots": int((~mask[stop:]).sum()),
        "evaluation_windows": total,
        "certain_positive_events": int(eval_lo.sum()), "possible_positive_events": int(eval_hi.sum()),
        "ambiguous_windows": int((~identified).sum()),
        "independent_mean_difference_bounds": [ind_lo/total, ind_hi/total],
        "joint_mean_difference_bounds": [joint_lo/total, joint_hi/total],
        "independent_width": independent_width, "joint_width": joint_width,
        "width_reduction_percent": 100*(1-joint_width/independent_width) if independent_width > 1e-15 else 0.,
        "stations_with_strict_tightening": int(improved_stations),
        "identified_only_mean_difference": naive,
        "identified_only_is_not_full_panel_risk": True,
        "best_single_snapshot_guaranteed_width_reduction": max(queries, default=0.)/total,
        "single_snapshot_queries_with_positive_guarantee": int(np.sum(np.asarray(queries)>1e-10)),
        "query_truths_obtained": 0, "query_result_is_hypothetical_planning_only": True,
        "finite_panel_context_dominates_global": bool(joint_lo > 0),
        "full_panel_true_Brier_known": bool(identified.all()),
        "prediction_sha256": prediction_hash,
        "new_independent_test": False, "statistical_significance_claim": False,
        "model_superiority_generalization_claim": False,
        "elapsed_seconds": time.perf_counter()-started,
    }
    assert ind_lo <= joint_lo+1e-8 and joint_hi <= ind_hi+1e-8
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.development_dir, args.output_dir)
