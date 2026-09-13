"""Fixed-budget dynamic volume increment, conditional on duration/static power."""
import math
import numpy as np
from .dual_risk_finite_step import GRID, choose_step, step_feasible, tangent
from .finite_step_metrics import score_cell, macro_score, decompose_cell

SYSTEMS = ('native', 'OD', 'OD_STATIC', 'OD_DUPLICATE', 'OD_VOLUME', 'OD_MISALIGNED')
REFERENCE = ('native', 'OD', 'OD_STATIC', 'OD_DUPLICATE')
WIDTHS = {'OD': 36, 'OD_STATIC': 48, 'OD_DUPLICATE': 59, 'OD_VOLUME': 59, 'OD_MISALIGNED': 59}
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)


def historical_power(duration, volume):
    if duration.shape != volume.shape or len(duration) < 719:
        raise ValueError('Power reference prefix missing')
    d, v = duration[:719], volume[:719]
    if not np.isfinite(d).all() or not np.isfinite(v).all() or (d < 0).any() or (v < 0).any():
        raise ValueError('Invalid reference data')
    ds, vs = d.sum(axis=0, dtype=np.float64), v.sum(axis=0, dtype=np.float64)
    if ds.sum() == 0:
        raise ValueError('DURATION_REFERENCE_EMPTY')
    global_ratio = float(vs.sum()/ds.sum())
    b = np.divide(vs, ds, out=np.full_like(vs, global_ratio), where=ds > 0)
    if not np.isfinite(b).all() or (b < 0).any():
        raise ValueError('Invalid historical power reference')
    return b, {'reference_range': [0, 719], 'zero_denominator_fallback_regions': int((ds == 0).sum()),
               'global_weighted_reference': global_ratio, 'reference_min': float(b.min()),
               'reference_median': float(np.median(b)), 'reference_max': float(b.max()),
               'interpretation': 'historical duration-weighted estimated power, not rated/actual power truth'}


def derive_series(duration, volume, capacities, power):
    d = duration/capacities[None, :]
    v = volume/capacities[None, :]
    s = d*power[None, :]
    z = v-s
    if not all(np.isfinite(a).all() for a in (d, v, s, z)) or ((d < 0) | (d > 1)).any() or (v < 0).any():
        raise ValueError('INPUT_BLOCKED: duration/capacity or volume values invalid')
    return {'d': d, 's': s, 'z': z}, {'rows': len(d), 'D_zero_V_positive': int(((duration == 0) & (volume > 0)).sum())}


def calendar(o, period):
    phase = o % period
    if phase == 0:
        return 0., 1.
    if phase == period//2:
        return 0., -1.
    return math.sin(2*math.pi*phase/period), math.cos(2*math.pi*phase/period)


def history(a, origin):
    if origin-169 < 0 or origin-1 > len(a):
        raise ValueError('Covariate history outside available prefix')
    return np.column_stack([a[origin-lag-1] for lag in LAGS]+
                           [a[origin-25:origin-1].mean(axis=0), a[origin-169:origin-1].mean(axis=0)])


def design(occupancy, derived, capacities, power, native, cut, horizon, system):
    if system not in WIDTHS or native.shape != (14, horizon, len(capacities)):
        raise ValueError('Unknown system or native cache shape')
    p = np.clip(native, 0., 1.)
    blocks = []
    n = len(capacities)
    perm = (np.arange(n)+137) % n
    for index, o in enumerate(range(cut, cut+168, 12)):
        if o > len(occupancy) or o-168 < 0:
            raise ValueError('Occupancy history outside available prefix')
        lags = np.column_stack([occupancy[o-lag] for lag in LAGS])
        stats = np.column_stack([occupancy[o-k:o].mean(axis=0) if op == 'mean' else
                                  occupancy[o-k:o].std(axis=0, ddof=0)
                                  for k in (24, 168) for op in ('mean', 'std')])
        hd, hs, hz = (history(derived[key], o) for key in ('d', 's', 'z'))
        clocks = calendar(o, 24)+calendar(o, 168)
        for j in range(horizon):
            base = np.column_stack([np.ones(n), lags, stats, p[index, j], p[index, j]**2,
                      np.full(n, horizon/12), np.full(n, (j+1)/horizon),
                      *[np.full(n, value) for value in clocks], native[index, j] <= 0,
                      native[index, j] >= 1, np.log1p(capacities), hd])
            if system != 'OD':
                base = np.column_stack([base, np.log1p(power), hs])
                if system == 'OD_DUPLICATE':
                    base = np.column_stack([base, hd])
                elif system == 'OD_VOLUME':
                    base = np.column_stack([base, hz])
                elif system == 'OD_MISALIGNED':
                    base = np.column_stack([base, hz[perm]])
            blocks.append(base)
    x = np.concatenate(blocks)
    if x.shape[1] != WIDTHS[system] or not np.isfinite(x).all():
        raise ValueError('Invalid registered design')
    return x, p.ravel()


def fit_scalar(cells):
    x = np.concatenate([a for a, _ in cells])
    y = np.concatenate([b for _, b in cells])
    w = np.concatenate([np.full(len(a), 1/(len(cells)*len(a))) for a, _ in cells])
    mu = w @ x[:, 1:]
    centered = x[:, 1:]-mu
    sigma = np.sqrt(w @ (centered*centered))
    inactive = sigma <= 1e-10*np.maximum(1., np.abs(mu))
    safe = np.where(inactive, 1., sigma)
    z = centered/safe
    z[:, inactive] = 0.
    a = np.column_stack([np.ones(len(x)), z])
    penalty = np.eye(a.shape[1])*.01
    penalty[0, 0] = 0.
    coefficient = np.linalg.solve(a.T @ (w[:, None]*a)+penalty, a.T @ (w*y))
    if not all(np.isfinite(v).all() for v in (mu, sigma, coefficient)):
        raise ValueError('Nonfinite fitted model')
    return {'center': mu, 'scale': sigma, 'inactive': inactive, 'coefficient': coefficient,
            'rows': len(x), 'columns': x.shape[1]}


def predict_direction(model, x, p):
    z = (x[:, 1:]-model['center'])/np.where(model['inactive'], 1., model['scale'])
    z[:, model['inactive']] = 0.
    residual = np.column_stack([np.ones(len(x)), z]) @ model['coefficient']
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite residual prediction')
    return tangent(p, np.clip(residual, -1., 1.))


def features_for(system, raw, derived, capacities, power, cache, cut, h):
    return design(raw['occupancy'], derived, capacities, power, cache[cut, h, 'native'], cut, h, system)


def fit_models(raw, derived, capacities, power, cache):
    models = {}
    for system in SYSTEMS[1:]:
        cells = []
        for h in (3, 12):
            x, p = features_for(system, raw, derived, capacities, power, cache, 720, h)
            cells.append((x, cache[720, h, 'truth'].ravel()-p))
        models[system] = fit_scalar(cells)
    return models


def directions(models, raw, derived, capacities, power, cache, cut):
    p = {h: np.clip(cache[cut, h, 'native'], 0., 1.).ravel() for h in (3, 12)}
    u = {'native': {h: np.zeros_like(p[h]) for h in (3, 12)}}
    for system in SYSTEMS[1:]:
        u[system] = {}
        for h in (3, 12):
            x, point = features_for(system, raw, derived, capacities, power, cache, cut, h)
            u[system][h] = predict_direction(models[system], x, point)
    return p, u


def calibrate(cache, p, u):
    base_cells = [dict(horizon=h, **score_cell(cache[1056, h, 'truth'].ravel(), p[h], u['native'][h],
                                            0., decompose=False)) for h in (3, 12)]
    base = {'alpha': 0., 'cells': base_cells, 'macro': macro_score(base_cells)}
    selected, grid = {'native': base}, {}
    for system in SYSTEMS[1:]:
        grid[system] = []
        for alpha in GRID:
            cells = base_cells if alpha == 0 else [dict(horizon=h, **score_cell(cache[1056, h, 'truth'].ravel(),
                      p[h], u[system][h], alpha, decompose=False)) for h in (3, 12)]
            grid[system].append({'alpha': alpha, 'cells': cells, 'macro': macro_score(cells)})
        selected[system] = choose_step(grid[system], base)
    return selected, grid


def gate(selected):
    if set(selected) != set(SYSTEMS):
        raise ValueError('All six registered systems required')
    name = min(REFERENCE, key=lambda s: selected[s]['macro']['rmse'])
    primary, base, ref, neg = (selected[s] for s in ('OD_VOLUME', 'native', name, 'OD_MISALIGNED'))
    pm, bm, rm, nm = (s['macro'] for s in (primary, base, ref, neg))
    checks = {'positive_alpha': primary['alpha'] > 0, 'positive_reference_rmse': rm['rmse'] > 0,
              'rmse_gain_at_least1percent': rm['rmse'] > 0 and pm['rmse'] <= .99*rm['rmse'],
              'mae_no_worse_than_native': pm['mae'] <= bm['mae'],
              'mae_no_worse_than_Cstar': pm['mae'] <= rm['mae'],
              'rmse_better_than_misaligned': pm['rmse'] < nm['rmse'],
              'mae_no_worse_than_misaligned': pm['mae'] <= nm['mae']}
    for i, h in enumerate((3, 12)):
        checks[f'H{h}_rmse_harm_native_at_most1percent'] = primary['cells'][i]['rmse'] <= 1.01*base['cells'][i]['rmse']
        checks[f'H{h}_rmse_harm_Cstar_at_most1percent'] = primary['cells'][i]['rmse'] <= 1.01*ref['cells'][i]['rmse']
    comparisons = {s: {'rmse_gain_percent': 100*(1-pm['rmse']/selected[s]['macro']['rmse']) if selected[s]['macro']['rmse'] else None,
                      'mae_difference': pm['mae']-selected[s]['macro']['mae']} for s in SYSTEMS if s != 'OD_VOLUME'}
    return {'passed': all(checks.values()), 'Cstar': name, 'conditions': checks,
            'comparisons': comparisons, 'new_structure_claim_tested': False}


def decompositions(cache, cut, p, u, selected):
    rows = []
    for system in SYSTEMS:
        for i, h in enumerate((3, 12)):
            row = decompose_cell(cache[cut, h, 'truth'].ravel(), p[h], u[system][h], selected[system]['alpha'], selected[system]['cells'][i])
            if max(abs(row[k]) for k in ('raw_mae_identity_error', 'clipped_mae_identity_error', 'rmse_identity_error')) > 1e-10:
                raise ValueError('Loss identity failed')
            rows.append({'cut': cut, 'system': system, 'horizon': h, 'alpha': selected[system]['alpha'], **row})
    return rows
