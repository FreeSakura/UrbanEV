"""Explicit target-scope scoring for benchmark bridges; no legacy defaults changed."""
from __future__ import annotations

import numpy as np


def scoped_scores(prediction, target, *, target_scope: str, postprocess: str):
    """Score (origin, horizon, region) arrays with a named target and output rule."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 3 or p.shape != y.shape or p.size == 0:
        raise ValueError('Expected nonempty matching (origin, horizon, region) arrays')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Predictions and targets must be finite')
    if target_scope not in {'terminal_H', 'path_1_to_H'}:
        raise ValueError('target_scope must be explicit: terminal_H or path_1_to_H')
    if postprocess == 'clip_0_1':
        if ((y < 0) | (y > 1)).any():
            raise ValueError('Rate clipping requires rate targets in [0,1]')
        p = np.clip(p, 0, 1)
    elif postprocess != 'raw':
        raise ValueError('postprocess must be explicit: raw or clip_0_1')
    if target_scope == 'terminal_H':
        p, y = p[:, -1:, :], y[:, -1:, :]
    error = p-y
    return {'target_scope':target_scope, 'postprocess':postprocess, 'scored_values':int(error.size), 'rmse':float(np.sqrt(np.mean(error**2))), 'mae':float(np.mean(np.abs(error)))}
