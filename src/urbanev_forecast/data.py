"""Hourly windows with target-contained chronological partitions."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import numpy as np
import pandas as pd


def fold_bounds(fold: int) -> tuple[int, int, int]:
    if isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= 6:
        raise ValueError("fold must be an integer in 1..6")
    end = int((pd.Timestamp("2022-09-01") + pd.DateOffset(months=fold) - pd.Timestamp("2022-09-01")) / pd.Timedelta(hours=1))
    train_end = int(.8 * end)
    return train_end, end-int(.1*end), end


def window_starts(length: int, history: int, horizon: int, target_start: int, target_stop: int):
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (length, history, horizon, target_start, target_stop)):
        raise ValueError("window arguments must be integers")
    if history < 1 or horizon < 1 or not 0 <= target_start < target_stop <= length:
        raise ValueError("invalid window or partition")
    starts = np.arange(max(history, target_start), target_stop-horizon+1, dtype=np.int64)
    if not len(starts):
        raise ValueError("partition has no complete forecast windows")
    return starts


def load_rate_prefix(path: Path, hours: int):
    """Read only the requested prepared-rate prefix; raw counts are not rates."""
    with path.open("rb") as handle:
        payload = b"".join(handle.readline() for _ in range(hours+1))
    frame = pd.read_csv(io.BytesIO(payload), index_col=0, parse_dates=True)
    expected = pd.date_range("2022-09-01", periods=hours, freq="h")
    values = frame.to_numpy(dtype=np.float32)
    if frame.shape != (hours, 275) or not frame.index.equals(expected):
        raise ValueError("Expected hourly prepared rates, 275 columns, beginning 2022-09-01")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Prepared occupancy rates must be finite and in [0,1]")
    return values, {"prefix_sha256": hashlib.sha256(payload).hexdigest(), "rows": hours, "channels": 275,
                    "column_order_sha256": hashlib.sha256("\n".join(map(str,frame.columns)).encode()).hexdigest()}


def scores(prediction, target):
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.size == 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Scores require matching, nonempty, finite arrays")
    return {"rmse": float(np.sqrt(np.mean((p-y)**2))), "mae": float(np.mean(np.abs(p-y)))}
