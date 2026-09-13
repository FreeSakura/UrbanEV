"""Direct finite-step scoring and descriptive loss accounting; no optimization."""
import math
import numpy as np


def mean(values):
    a = np.asarray(values)
    return math.fsum(a.ravel()) / a.size


def score_cell(y, p, direction, alpha, decompose=True):
    y, p, h = (np.asarray(x, dtype=np.float64) for x in (y, p, direction))
    if (not y.size or y.shape != p.shape or y.shape != h.shape
            or not all(np.isfinite(x).all() for x in (y, p, h))
            or ((y < 0) | (y > 1) | (p < 0) | (p > 1)).any()
            or not math.isfinite(alpha) or alpha < 0):
        raise ValueError('Invalid bounded finite-step input')
    raw = p + alpha*h
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite finite-step prediction')
    q = np.clip(raw, 0., 1.)
    error = y-p
    raw_error = y-raw
    clipped_error = y-q
    baseline_mae = mean(np.abs(error))
    raw_mae = mean(np.abs(raw_error))
    mae = mean(np.abs(clipped_error))
    result = {'n': y.size, 'alpha': float(alpha),
              'rmse': math.sqrt(mean(clipped_error*clipped_error)), 'mae': mae,
              'unclipped_move_rmse': math.sqrt(mean(raw_error*raw_error)), 'unclipped_move_mae': raw_mae,
              'baseline_rmse': math.sqrt(mean(error*error)), 'baseline_mae': baseline_mae}
    if decompose:
        result.update(decompose_cell(y, p, h, alpha, result))
    return result


def decompose_cell(y, p, h, alpha, metrics):
    """Account for one already selected point; never chooses a step."""
    y, p, h = (np.asarray(x, dtype=np.float64) for x in (y, p, h))
    raw = p+alpha*h
    q = np.clip(raw, 0., 1.)
    error = y-p
    derivative = mean(-np.sign(error)*h+(error == 0)*np.abs(h))
    linear = alpha*derivative
    crossing = 2*mean(np.maximum(alpha*np.abs(h)-np.abs(error), 0.)*(error*h > 0))
    clipping = -mean(np.maximum(-raw, 0.)+np.maximum(raw-1., 0.))
    mse_linear = -2*alpha*mean(error*h)
    mse_quadratic = alpha*alpha*mean(h*h)
    mse_clipping = -mean((raw-y)**2-(q-y)**2)
    delta_mse = mse_linear+mse_quadratic+mse_clipping
    denominator = metrics['rmse']+metrics['baseline_rmse']
    delta_rmse = delta_mse/denominator if denominator else 0.
    return {'mae_derivative_at_zero': derivative, 'mae_linear_term': linear,
            'mae_crossing_term': crossing, 'mae_clipping_adjustment': clipping,
            'raw_mae_identity_error': metrics['unclipped_move_mae']-metrics['baseline_mae']-linear-crossing,
            'clipped_mae_identity_error': metrics['mae']-metrics['baseline_mae']-linear-crossing-clipping,
            'mse_linear_term': mse_linear, 'mse_quadratic_term': mse_quadratic,
            'mse_clipping_adjustment': mse_clipping, 'reconstructed_mse_change': delta_mse,
            'reconstructed_rmse_change': delta_rmse,
            'rmse_identity_error': delta_rmse-(metrics['rmse']-metrics['baseline_rmse']),
            'raw_outside_fraction': mean((raw < 0) | (raw > 1)),
            'error_crossing_fraction': mean((error*h > 0) & (alpha*np.abs(h) > np.abs(error))),
            'zero_direction_fraction': mean(h == 0)}



def macro_score(cells):
    if not cells:
        raise ValueError('At least one complete cell required')
    # Every cell has equal weight, independent of horizon length or sample count.
    keys = ('rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae', 'baseline_rmse', 'baseline_mae',
            'mae_linear_term', 'mae_crossing_term', 'mae_clipping_adjustment')
    return {key: math.fsum(c[key] for c in cells)/len(cells) for key in keys if all(key in c for c in cells)}
