"""Exact finite-panel Brier comparison with shared missing binary observations.

Missing observations are -1. No probability law or missing-at-random assumption
is imposed. The returned bounds are identification bounds, not confidence limits.
"""
from __future__ import annotations
import numpy as np


def _inputs(observed, a, b, window, run_length):
    o = np.asarray(observed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if o.ndim != 1 or not np.isin(o, (-1, 0, 1)).all():
        raise ValueError("observed must be a one-dimensional -1/0/1 sequence")
    if not isinstance(window, (int, np.integer)) or not 1 <= window <= min(12, len(o)):
        raise ValueError("window must be an integer between 1 and min(12, T)")
    if not isinstance(run_length, (int, np.integer)) or not 1 <= run_length <= window:
        raise ValueError("run_length must be an integer between 1 and window")
    if a.shape != (len(o)-window+1,) or b.shape != a.shape:
        raise ValueError("one probability per complete sliding window is required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    return o.astype(np.int8), a, b


def event_bounds(observed, window, run_length):
    """Lower/upper binary event labels; unknowns are filled only for bounding."""
    o = np.asarray(observed)
    if o.ndim != 1 or not np.isin(o, (-1, 0, 1)).all() or not 1 <= run_length <= window <= len(o):
        raise ValueError("invalid observation or window specification")
    views = np.lib.stride_tricks.sliding_window_view(o, window)
    def event(bits):
        sums = np.concatenate([np.zeros((len(bits), 1), int), np.cumsum(bits, axis=1)], axis=1)
        return np.any(sums[:, run_length:] - sums[:, :-run_length] == run_length, axis=1)
    return event(views == 1), event(views != 0)


def paired_brier_bounds(a, b, lower, upper):
    a, b, lo, hi = map(np.asarray, (a, b, lower, upper))
    if a.shape != b.shape or a.shape != lo.shape or lo.shape != hi.shape:
        raise ValueError("paired arrays must have identical shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("invalid probabilities")
    if not np.isin(lo, (0, 1)).all() or not np.isin(hi, (0, 1)).all() or (lo > hi).any():
        raise ValueError("invalid binary label bounds")
    a, b = a.astype(float), b.astype(float)
    constant, slope = a*a-b*b, 2*(b-a)
    return constant + np.minimum(slope*lo, slope*hi), constant + np.maximum(slope*lo, slope*hi)


def joint_brier_bounds(observed, a, b, window=4, run_length=2, *, query_planning=False):
    """Exact sum bounds, plus optional conditional single-bit query bounds.

    a is the reference and b the candidate; positive difference favors b.
    Runtime is O(T*2**window). Query planning stores O(T*2**(window-1))
    forward/backward tables. The window cap is a practical resource guard.
    """
    o, a, b = _inputs(observed, a, b, window, run_length)
    T, count = len(o), 1 << (window-1)
    state = np.arange(count)
    dest, hit = [], []
    for bit in (0, 1):
        pattern = 2*state+bit
        active = pattern.copy()
        for shift in range(1, run_length):
            active &= pattern >> shift
        dest.append(pattern & (count-1))
        hit.append((active != 0).astype(float))
    coefficient, constant = 2*(b-a), float(np.sum(a*a-b*b))
    shape = (T+1, count) if query_planning else (2, count)
    forward_lo = np.full(shape, np.inf)
    forward_hi = np.full(shape, -np.inf)
    forward_lo[0, 0] = forward_hi[0, 0] = 0
    def weight(t):
        return 0.0 if t < window-1 else coefficient[t-window+1]
    for t in range(T):
        prev, nxt = (t, t+1) if query_planning else (t % 2, (t+1) % 2)
        forward_lo[nxt].fill(np.inf)
        forward_hi[nxt].fill(-np.inf)
        for bit in ((0, 1) if o[t] == -1 else (int(o[t]),)):
            edge = weight(t)*hit[bit]
            np.minimum.at(forward_lo[nxt], dest[bit], forward_lo[prev]+edge)
            np.maximum.at(forward_hi[nxt], dest[bit], forward_hi[prev]+edge)
    last = T if query_planning else T % 2
    lower = constant + float(forward_lo[last].min())
    upper = constant + float(forward_hi[last].max())
    label_lo, label_hi = event_bounds(o, window, run_length)
    independent_lo, independent_hi = paired_brier_bounds(a, b, label_lo, label_hi)
    result = {
        "window_count": len(a), "missing_snapshots": int((o == -1).sum()),
        "ambiguous_windows": int(np.sum(label_lo != label_hi)),
        "lower_sum": lower, "upper_sum": upper,
        "independent_lower_sum": float(independent_lo.sum()),
        "independent_upper_sum": float(independent_hi.sum()),
        "query_planning": [],
    }
    if not query_planning:
        return result
    backward_lo = np.zeros((T+1, count))
    backward_hi = np.zeros((T+1, count))
    for t in range(T-1, -1, -1):
        lows, highs = [], []
        for bit in ((0, 1) if o[t] == -1 else (int(o[t]),)):
            edge = weight(t)*hit[bit]
            lows.append(edge+backward_lo[t+1, dest[bit]])
            highs.append(edge+backward_hi[t+1, dest[bit]])
        backward_lo[t] = np.minimum.reduce(lows)
        backward_hi[t] = np.maximum.reduce(highs)
    for t in np.flatnonzero(o == -1):
        branches = []
        for bit in (0, 1):
            edge = weight(t)*hit[bit]
            lo = constant + float(np.min(forward_lo[t]+edge+backward_lo[t+1, dest[bit]]))
            hi = constant + float(np.max(forward_hi[t]+edge+backward_hi[t+1, dest[bit]]))
            branches.append({"revealed_bit": bit, "lower_sum": lo, "upper_sum": hi})
        worst_width = max(x["upper_sum"]-x["lower_sum"] for x in branches)
        result["query_planning"].append({
            "index": int(t), "branches": branches,
            "guaranteed_width_reduction_sum": max(0., upper-lower-worst_width),
        })
    return result
