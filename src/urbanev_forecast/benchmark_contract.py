"""Data-free building blocks for an explicit UrbanEV comparison contract.

These functions do not load data, train models, or authorize a benchmark run.
The pinned upstream loader is the source for fold and complete-window indexing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from numbers import Integral
from typing import Mapping

import numpy as np

PREFIX_HOURS = (720, 1464, 2184, 2928, 3672, 4344)
HORIZONS = (3, 6, 9, 12)
UPSTREAM_REVISION = "44f2aa0c8d89f192bce00bafb0def74a21b39c68"
CONTRACTS = ("UPSTREAM_TRANSFORMER", "UPSTREAM_CLASSICAL", "MATCHED_TERMINAL_168")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Fold:
    number: int
    train_stop: int
    validation_stop: int
    test_stop: int

    def interval(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.train_stop
        if split == "validation":
            return self.train_stop, self.validation_stop
        if split == "test":
            return self.validation_stop, self.test_stop
        raise ValueError("split must be train, validation or test")


def fold_plan(number: int, contract="UPSTREAM_TRANSFORMER") -> Fold:
    number = _integer(number, "fold", 1)
    if number > 6:
        raise ValueError("fold must be in 1..6")
    if contract not in CONTRACTS:
        raise ValueError("unknown contract")
    # Compute by year-month boundary, not month-number membership across years.
    next_month = datetime(2022 + (8 + number)//12, (8 + number)%12 + 1, 1)
    end = int((next_month-datetime(2022, 9, 1)).total_seconds()//3600)
    # Integer expressions agree with pinned int(0.8*N), int(0.1*N) at all six N.
    train = end * 8 // 10
    validation = (10*train+end)//10 if contract == "UPSTREAM_CLASSICAL" else end-end//10
    return Fold(number, train, validation, end)


def origins(fold: int, split: str, history: int, horizon: int, *, stride: int = 1,
            contract="UPSTREAM_TRANSFORMER") -> np.ndarray:
    """Origin is the first target hour; history ends at origin-1.

    All H target hours must be inside the split, including for endpoint scoring.
    Past values may precede validation/test start, as in the upstream loader.
    """
    history = _integer(history, "history", 1)
    horizon = _integer(horizon, "horizon", 1)
    stride = _integer(stride, "stride", 1)
    start, stop = fold_plan(fold, contract).interval(split)
    if contract == "MATCHED_TERMINAL_168":
        if history != 168:
            raise ValueError("matched contract fixes history=168")
        first, exclusive_last = max(start, 169), stop-horizon+1
    elif contract == "UPSTREAM_CLASSICAL":
        first, exclusive_last = start+history, stop-horizon
    else:
        first, exclusive_last = max(start, history), stop-horizon+1
    result = np.arange(first, exclusive_last, stride, dtype=np.int64)
    if result.size == 0 and contract != "UPSTREAM_CLASSICAL":
        raise ValueError("no complete forecast windows")
    return result


def origin_digest(values) -> str:
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("origins must be nonempty nonnegative integer vector")
    if np.any(a[1:] <= a[:-1]):
        raise ValueError("origins must be unique and strictly chronological")
    return hashlib.sha256(a.astype("<i8", copy=False).tobytes()).hexdigest()


def cell_metadata(fold: int, horizon: int, history: int, split="test", contract="UPSTREAM_TRANSFORMER") -> dict:
    o = origins(fold, split, history, horizon, contract=contract)
    common = {**asdict(fold_plan(fold, contract)), "contract": contract, "split": split, "history": history,
              "horizon": horizon, "count": len(o)}
    if not len(o):
        return {**common, "first_origin": None, "last_origin": None, "origin_sha256": None,
                "first_target": None, "last_target": None, "max_occupancy_input": None,
                "max_lagged_duration_input": None}
    return {**common,
            "horizon": horizon, "count": len(o), "first_origin": int(o[0]),
            "last_origin": int(o[-1]), "first_target": int(o[0]),
            "last_target": int(o[-1] + horizon - 1), "origin_sha256": origin_digest(o),
            "max_occupancy_input": int(o[-1]-1),
            "max_lagged_duration_input": int(o[-1]-2) if contract == "MATCHED_TERMINAL_168" else None}


def score_arrays(prediction, target, *, scope: str, postprocess: str) -> dict:
    """Score matching [origin, horizon, zone] arrays; no implicit clipping/slicing.

    Terminal consumes exactly the last output hour, matching the pinned metric.
    Path is a separate metric and weights repeated target hours across origins.
    """
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 3 or p.shape != y.shape or not p.size or not all(p.shape):
        raise ValueError("expected matching nonempty [origin,horizon,zone] arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite prediction or target")
    if postprocess == "clip_0_1":
        p = np.clip(p, 0, 1)
    elif postprocess != "raw":
        raise ValueError("postprocess must be explicit: raw or clip_0_1")
    if scope == "terminal":
        p, y = p[:, -1, :], y[:, -1, :]
    elif scope != "path":
        raise ValueError("scope must be explicit: terminal or path")
    error = p - y
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    if not np.isfinite(mse) or not np.isfinite(mae):
        raise ValueError("nonfinite metric (possible overflow)")
    return {"rmse": float(np.sqrt(mse)), "mae": mae, "mse": mse,
            "element_count": int(error.size), "scope": scope, "postprocess": postprocess}


def paired_cell(prediction, target, actual_origins, *, fold, horizon, history,
                split, scope, postprocess, contract="UPSTREAM_TRANSFORMER"):
    """Require exact order and full support; intersections never silently pass."""
    expected = origins(fold, split, history, horizon, contract=contract)
    actual = np.asarray(actual_origins)
    origin_digest(actual)
    if not np.array_equal(actual, expected):
        raise ValueError("origin support/order differs from planned complete cell")
    p, y = np.asarray(prediction), np.asarray(target)
    if p.ndim != 3 or p.shape[:2] != (len(expected), horizon) or p.shape[2] != 275:
        raise ValueError("cell must have all planned origins, H outputs and 275 zones")
    return {**cell_metadata(fold, horizon, history, split, contract),
            **score_arrays(p, y, scope=scope, postprocess=postprocess)}


def macro24(cells: Mapping[tuple[int, int], Mapping]) -> dict:
    """Equal mean of 24 cell RMSE/MAE, never sqrt(mean of cell MSE).

    A single seed and information/selection protocol must be supplied by caller.
    Missing cells are rejected instead of producing a misleading overall number.
    """
    required = {(f, h) for f in range(1, 7) for h in HORIZONS}
    if set(cells) != required:
        raise ValueError("exactly six folds by four horizons required")
    rows = [cells[key] for key in sorted(required)]
    for field in ("scope", "postprocess", "split", "history"):
        if any(field not in r for r in rows) or len({r[field] for r in rows}) != 1:
            raise ValueError(f"missing or mixed {field}")
    if rows[0]["scope"] not in ("terminal", "path") or rows[0]["postprocess"] not in ("raw", "clip_0_1"):
        raise ValueError("unknown score convention")
    result = {field: rows[0][field] for field in ("scope", "postprocess", "split", "history")}
    for metric in ("rmse", "mae"):
        values = np.array([r[metric] for r in rows], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("metrics must be finite and nonnegative")
        result[metric] = float(np.mean(values))
    return {**result, "cells": 24, "aggregation": "equal_cell_metric_mean"}


def declared_rates(values, region_ids, *, raw_unit, capacity_level, capacity_aggregation,
                   capacity_records=()):
    """Explicit unit conversion on supplied arrays, never infer units by range.

    Records are (region_id, capacity) with one row per station or per region.
    Station capacities sum by region. The float32 division matches the declared
    prepared-rate contract; passing declared rates performs no second division.
    """
    ids = tuple(region_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("target region IDs must be nonempty and unique")
    v = np.asarray(values, dtype=np.float32)
    if v.ndim < 1 or v.shape[-1] != len(ids) or not v.size or not np.isfinite(v).all():
        raise ValueError("invalid values or region axis")
    policies = {"station": "sum_by_TAZID", "region": "already_aggregated"}
    if policies.get(capacity_level) != capacity_aggregation:
        raise ValueError("capacity level and aggregation disagree")
    if raw_unit == "busy_count":
        capacities = {}
        for region, capacity in capacity_records:
            value = float(capacity)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("capacity must be finite and positive")
            if capacity_level == "region" and region in capacities:
                raise ValueError("duplicate region-level capacity")
            capacities[region] = capacities.get(region, 0.) + value
        if set(capacities) != set(ids):
            raise ValueError("capacity IDs must match target regions exactly")
        cap = np.asarray([capacities[k] for k in ids], dtype=np.float32)
        if not np.isfinite(cap).all() or np.any(cap <= 0):
            raise ValueError("invalid aggregated capacity")
        v = v / cap
    elif raw_unit != "occupancy_rate":
        raise ValueError("raw_unit must explicitly be busy_count or occupancy_rate")
    if np.any(v < 0) or np.any(v > 1):
        raise ValueError("declared occupancy rates must lie in [0,1]")
    return v.copy()


def validation_rmse(batch_sums):
    """Global endpoint SSE/count; unequal batches are never averaged equally."""
    total, count = 0., 0
    for sse, n in batch_sums:
        n = _integer(n, "element count", 1)
        if not np.isfinite(sse) or sse < 0:
            raise ValueError("invalid squared-error sum")
        total += float(sse)
        count += n
    if count == 0 or not np.isfinite(total):
        raise ValueError("empty/overflowed validation scores")
    return float(np.sqrt(total/count))


def select_checkpoint(checkpoints, *, split, scope, postprocess):
    if (split, scope, postprocess) != ("validation", "terminal", "raw"):
        raise ValueError("selection requires validation raw terminal errors")
    if not checkpoints:
        raise ValueError("no registered checkpoints")
    scored = []
    counts = []
    for epoch, batches in checkpoints.items():
        epoch = _integer(epoch, "checkpoint epoch")
        batches = tuple(batches)
        scored.append((validation_rmse(batches), epoch))
        counts.append(sum(n for _, n in batches))
    if len(set(counts)) != 1:
        raise ValueError("checkpoint validation sample counts differ")
    value, epoch = min(scored)
    return {"epoch": epoch, "validation_raw_terminal_rmse": value, "element_count": counts[0]}


def training_partitions(factory):
    """A future runner may use this helper; it never requests a test partition."""
    return {split: factory(split) for split in ("train", "validation")}


def validate_training_spec(spec):
    """Preparation has no defaults that could silently launch a real experiment."""
    required = ("epochs", "seeds", "search_candidates", "training_objective", "prediction_mode")
    if any(k not in spec or spec[k] is None for k in required):
        raise ValueError("explicit epochs, seeds, search candidates, objective and prediction mode required")
    _integer(spec["epochs"], "epochs", 1)
    for name in ("seeds", "search_candidates"):
        if not isinstance(spec[name], list) or not spec[name]:
            raise ValueError(f"{name} must be a nonempty explicit list")
    for s in spec["seeds"]:
        _integer(s, "seed")
    if len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("seeds must be unique")
    if spec["training_objective"] not in ("raw_path_mse", "raw_terminal_mse"):
        raise ValueError("unknown training objective")
    if spec["prediction_mode"] not in ("horizon_specific", "joint_12"):
        raise ValueError("unknown prediction mode")
    return dict(spec)


def upstream_percentage_mirror(prediction, target):
    """Diagnostic MAPE/RAE operation order from the upstream metric, not ranking.

    A zero RAE denominator intentionally returns nan/inf, as upstream does.
    """
    p, y = np.asarray(prediction, dtype=np.float64).copy(), np.asarray(target, dtype=np.float64).copy()
    if p.shape != y.shape or not p.size or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("matching finite nonempty inputs required")
    y[y <= .02] = np.abs(y[y <= .02]) + .02
    p[y <= .02] = np.abs(p[y <= .02]) + .02
    mape = np.mean(np.abs(p-y)/np.maximum(np.abs(y), np.finfo(np.float64).eps))
    with np.errstate(divide="ignore", invalid="ignore"):
        rae = np.sum(np.abs(p-y))/np.sum(np.abs(np.mean(y)-y))
    return {"mape": float(mape), "rae": float(rae)}
