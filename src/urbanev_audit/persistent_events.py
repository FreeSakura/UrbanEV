"""Exact weighted event bounds using trailing-run length and last-hit age.

For an event of L consecutive ones in a K-window the state space is at most
L*(K-L+2), instead of keeping 2**(K-1) suffixes. Predictions are fixed inputs;
unknown observations (-1) are optimized jointly, never imputed as truth.
"""
from functools import lru_cache

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


@lru_cache(maxsize=32)
def automaton(window, run_length):
    """Build only reachable states; r is capped at L-1, age at K-L+1."""
    if not 1 <= run_length <= window:
        raise ValueError("Require 1 <= L <= K")
    expired = window - run_length + 1
    states = [(0, expired)]
    index = {states[0]: 0}
    edges = []
    for r, age in states:
        row = []
        for bit in (0, 1):
            hit = bit == 1 and r == run_length - 1
            next_r = min(run_length - 1, r + 1) if bit else 0
            next_age = 0 if hit else min(expired, age + 1)
            dest = (next_r, next_age)
            if dest not in index:
                index[dest] = len(states)
                states.append(dest)
            row.append(index[dest])
        edges.append(row)
    return np.asarray(edges, dtype=np.int64), np.asarray([age < expired for _, age in states]), tuple(states)


@njit(cache=True)
def _extrema(observed, coefficients, window, edges, event):
    count = len(edges)
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    low[0] = high[0] = 0.0
    for t in range(len(observed)):
        next_low = np.full(count, np.inf)
        next_high = np.full(count, -np.inf)
        weight = coefficients[t - window + 1] if t >= window - 1 else 0.0
        for s in range(count):
            if not np.isfinite(low[s]):
                continue
            for bit in range(2):
                if observed[t] != -1 and observed[t] != bit:
                    continue
                dest = edges[s, bit]
                cost = weight if event[dest] else 0.0
                next_low[dest] = min(next_low[dest], low[s] + cost)
                next_high[dest] = max(next_high[dest], high[s] + cost)
        low, high = next_low, next_high
    return low.min(), high.max()


@njit(cache=True)
def _event_labels(bits, window, run_length):
    labels = np.empty(len(bits) - window + 1, dtype=np.int8)
    trailing, last_end = 0, -len(bits) - window
    for t in range(len(bits)):
        trailing = trailing + 1 if bits[t] else 0
        if trailing >= run_length:
            last_end = t
        if t >= window - 1:
            labels[t - window + 1] = last_end >= t - window + run_length
    return labels


def event_labels(bits, window, run_length):
    return _event_labels(np.asarray(bits, dtype=np.int8), window, run_length)


def label_bounds(observed, window, run_length):
    o = np.asarray(observed)
    return event_labels(o == 1, window, run_length), event_labels(o != 0, window, run_length)


def weighted_bounds(observed, coefficients, window, run_length):
    o = np.asarray(observed, dtype=np.int8)
    c = np.asarray(coefficients, dtype=np.float64)
    if o.ndim != 1 or not np.isin(observed, (-1, 0, 1)).all():
        raise ValueError("Expected a one-dimensional -1/0/1 observation sequence")
    if not 1 <= run_length <= window <= len(o) or c.shape != (len(o) - window + 1,) or not np.isfinite(c).all():
        raise ValueError("Invalid window or weighted sliding-panel dimensions")
    edges, event, states = automaton(window, run_length)
    low, high = _extrema(o, c, window, edges, event)
    lo, hi = label_bounds(o, window, run_length)
    return {"lower_sum": float(low), "upper_sum": float(high),
            "independent_lower_sum": float(np.minimum(c * lo, c * hi).sum()),
            "independent_upper_sum": float(np.maximum(c * lo, c * hi).sum()),
            "window_count": len(c), "ambiguous_windows": int(np.sum(lo != hi)),
            "missing_snapshots": int(np.sum(o == -1)), "reachable_states": len(states),
            "state_slot_bound": run_length * (window - run_length + 2)}


def paired_bounds(observed, a, b, window, run_length):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError("Expected matching finite probabilities in [0,1]")
    result = weighted_bounds(observed, 2 * (b - a), window, run_length)
    constant = float(np.sum(a * a - b * b))
    for key in ("lower_sum", "upper_sum", "independent_lower_sum", "independent_upper_sum"):
        result[key] += constant
    return result


def component_sign_certificate(observed, coefficients, window, run_length):
    """Sufficient equality condition using shared-unknown dependency components.

    Connect nonconstant, nonzero-weight windows when they contain a common
    unknown bit. Same-sign weights within every component imply exact equality
    with independent bounds, even when the entire panel has mixed signs.
    """
    o = np.asarray(observed)
    c = np.asarray(coefficients)
    lower, upper = label_bounds(o, window, run_length)
    active = np.flatnonzero((lower != upper) & (c != 0))
    missing = np.flatnonzero(o == -1)
    components = []
    end = -1
    for s in active:
        first = int(np.searchsorted(missing, s))
        last = int(np.searchsorted(missing, s + window)) - 1
        if not components or first > end:
            components.append({"first_window": int(s), "last_window": int(s), "positive": 0, "negative": 0})
            end = last
        else:
            end = max(end, last)
        components[-1]['last_window'] = int(s)
        components[-1]['positive' if c[s] > 0 else 'negative'] += 1
    mixed = sum(item['positive'] > 0 and item['negative'] > 0 for item in components)
    return {"active_ambiguous_windows": len(active), "components": len(components),
            "mixed_sign_components": mixed, "independent_bounds_provably_exact": mixed == 0,
            "component_details": components}


@njit(cache=True)
def _feasible(observed, required_labels, window, edges, event):
    reachable = np.zeros(len(edges), dtype=np.bool_)
    reachable[0] = True
    for t in range(len(observed)):
        nxt = np.zeros(len(edges), dtype=np.bool_)
        required = required_labels[t-window+1] if t >= window-1 else -1
        for s in range(len(edges)):
            if reachable[s]:
                for bit in range(2):
                    dest = edges[s, bit]
                    if (observed[t] == -1 or observed[t] == bit) and (required == -1 or required == event[dest]):
                        nxt[dest] = True
        reachable = nxt
        if not reachable.any():
            return False
    return reachable.any()


def endpoint_certificate(observed, coefficients, window, run_length, endpoint='lower', *, explain=False):
    """Independent endpoint is exact iff its preferred label pattern is feasible.

    With explain=True, remove redundant requirements to return a deletion-minimal
    inconsistent window set. This is not a minimum-cardinality core.
    """
    if endpoint not in ('lower', 'upper'):
        raise ValueError('endpoint must be lower or upper')
    o = np.asarray(observed, dtype=np.int8)
    c = np.asarray(coefficients, dtype=float)
    lower, upper = label_bounds(o, window, run_length)
    required = np.full(len(c), -1, dtype=np.int8)
    active = (lower != upper) & (c != 0)
    positive = c > 0 if endpoint == 'lower' else c < 0
    required[active] = np.where(positive[active], lower[active], upper[active])
    edges, event, _ = automaton(window, run_length)
    feasible = bool(_feasible(o, required, window, edges, event))
    core = []
    if explain and not feasible:
        for index in np.flatnonzero(required != -1):
            prior = required[index]
            required[index] = -1
            if _feasible(o, required, window, edges, event):
                required[index] = prior
        core = [{'window_start': int(i), 'required_event': int(required[i]), 'coefficient': float(c[i])}
                for i in np.flatnonzero(required != -1)]
    return {'endpoint': endpoint, 'independent_endpoint_attainable': feasible,
            'conflict_core': core, 'core_kind': 'deletion-minimal, not necessarily minimum cardinality'}


@njit(cache=True)
def suffix_extrema(observed, coefficients, window, run_length):
    """Same compiled DP kernel style, exponential state baseline for timing."""
    count = 1 << (window - 1)
    low, high = np.full(count, np.inf), np.full(count, -np.inf)
    low[0] = high[0] = 0.0
    for t in range(len(observed)):
        next_low, next_high = np.full(count, np.inf), np.full(count, -np.inf)
        weight = coefficients[t-window+1] if t >= window-1 else 0.0
        for s in range(count):
            if not np.isfinite(low[s]):
                continue
            for bit in range(2):
                if observed[t] != -1 and observed[t] != bit:
                    continue
                pattern = 2*s+bit
                active = pattern
                for shift in range(1, run_length):
                    active &= pattern >> shift
                dest = pattern & (count-1)
                cost = weight if active else 0.0
                next_low[dest] = min(next_low[dest], low[s]+cost)
                next_high[dest] = max(next_high[dest], high[s]+cost)
        low, high = next_low, next_high
    return low.min(), high.max()
