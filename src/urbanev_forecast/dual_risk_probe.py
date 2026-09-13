"""Frozen conditional-moment probe; no alpha search or foundation inference."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np

SYSTEMS = ('native', 'occupancy', 'raw_duration', 'duplicate', 'permuted_orthogonal')
REPRESENTATIONS = ('base', 'duration', 'duplicate', 'permuted')


def load_whitelist(manifest, roots):
    """Verify every byte identity before any numerical array is decoded."""
    expected = {(c, h, s) for c in (720, 1056, 1392) for h in (3, 12)
                for s in ('truth',) + SYSTEMS}
    rows = manifest['files']
    keys = [(r['cut'], r['horizon'], r['system']) for r in rows]
    if len(keys) != 36 or set(keys) != expected:
        raise ValueError('PROBE_BLOCKED: whitelist must contain exactly 36 unique arrays')
    payloads, ledger = {}, []
    for row in rows:
        c, h, s = row['cut'], row['horizon'], row['system']
        role = 'truth' if s == 'truth' else 'v1'
        name = f'private_truth_{c}_h{h}.npy' if s == 'truth' else f'private_{c}_h{h}_{s}_a1.npy'
        if row['root_role'] != role or row['file'] != name or row['shape'] != [14, h, 275]:
            raise ValueError('PROBE_BLOCKED: file role, name or shape contract')
        root = Path(roots[role]).resolve()
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError('PROBE_BLOCKED: file outside allowed root')
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != row['sha256']:
            raise ValueError('PROBE_BLOCKED: source hash mismatch')
        payloads[c, h, s] = blob
        ledger.append({**row, 'actual_sha256': digest, 'bytes': len(blob)})
    arrays = {}
    for key, blob in payloads.items():
        a = np.load(io.BytesIO(blob), allow_pickle=False)
        if a.dtype.kind not in 'fi' or a.shape != (14, key[1], 275) or not np.isfinite(a).all():
            raise ValueError('PROBE_BLOCKED: invalid numerical array')
        if key[2] == 'truth' and ((a < 0).any() or (a > 1).any()):
            raise ValueError('PROBE_BLOCKED: truth outside occupancy-rate range')
        arrays[key] = a.astype(np.float64)
    return arrays, ledger


def design(arrays, cut, horizon, representation):
    raw = arrays[cut, horizon, 'native']
    p = np.clip(raw, 0., 1.)
    occupied = arrays[cut, horizon, 'occupancy']
    shape = raw.shape
    origins = np.arange(cut, cut + 168, 12)[:, None, None]
    hour = origins % 24
    sine = np.sin(2 * np.pi * hour / 24)
    sine = np.where((hour == 0) | (hour == 12), 0., sine)
    cosine = np.cos(2 * np.pi * hour / 24)
    fields = [np.ones(shape), p, p*p, occupied-raw,
              np.full(shape, horizon/12),
              np.broadcast_to(np.arange(1, horizon+1)[None, :, None]/horizon, shape),
              np.broadcast_to(sine, shape), np.broadcast_to(cosine, shape), raw <= 0, raw >= 1]
    extra = {'duration': 'raw_duration', 'duplicate': 'duplicate', 'permuted': 'permuted_orthogonal'}
    if representation in extra:
        fields.append(arrays[cut, horizon, extra[representation]] - occupied)
    elif representation != 'base':
        raise ValueError('Unknown representation')
    return np.column_stack([v.ravel() for v in fields]), p.ravel()


def moment_targets(y, p):
    return np.column_stack((y-p, y < p, y == p, y > p)).astype(np.float64)


def fit_ridge(cells, regularization=0.01):
    """Minimize sum_i w_i ||T_i-X_i B||² + lambda ||B_nonintercept||²."""
    if regularization <= 0 or not cells:
        raise ValueError('Invalid ridge contract')
    x = np.concatenate([a for a, _ in cells])
    target = np.concatenate([b for _, b in cells])
    w = np.concatenate([np.full(len(a), 1/(len(cells)*len(a))) for a, _ in cells])
    z = x[:, 1:]
    center = w @ z
    centered = z-center
    scale = np.sqrt(w @ (centered*centered))
    constant = np.ptp(z, axis=0) == 0
    scale[constant] = 0.
    safe_scale = np.where(scale == 0, 1., scale)
    standardized = centered/safe_scale
    standardized[:, scale == 0] = 0
    matrix = np.column_stack((np.ones(len(x)), standardized))
    penalty = np.eye(matrix.shape[1])*regularization
    penalty[0, 0] = 0.
    coefficient = np.linalg.solve(matrix.T @ (w[:, None]*matrix)+penalty,
                                  matrix.T @ (w[:, None]*target))
    return {'center': center, 'scale': scale, 'coefficient': coefficient,
            'training_cells': len(cells), 'training_rows': len(x)}


def predict_moments(model, x):
    scale = model['scale']
    z = (x[:, 1:]-model['center'])/np.where(scale == 0, 1., scale)
    z[:, scale == 0] = 0.
    pred = np.column_stack((np.ones(len(x)), z)) @ model['coefficient']
    probabilities = project_simplex(pred[:, 1:])
    return pred[:, 0], probabilities


def project_simplex(values):
    values = np.asarray(values, dtype=float)
    u = np.sort(values, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)-1
    positive = u-cssv/np.arange(1, values.shape[1]+1) > 0
    rho = positive.sum(axis=1)-1
    theta = cssv[np.arange(len(values)), rho]/(rho+1)
    return np.maximum(values-theta[:, None], 0.)


def common_direction(mean, probabilities, p):
    sign_imbalance = probabilities[:, 0]-probabilities[:, 2]
    atom = probabilities[:, 1]
    k = np.maximum(-sign_imbalance*np.sign(mean)-atom, 0.)
    direction = np.clip(mean, -1., 1.)*k
    direction = np.where((p == 0) & (direction < 0), 0., direction)
    return np.where((p == 1) & (direction > 0), 0., direction)


def average(values):
    return math.fsum(np.asarray(values).ravel())/np.asarray(values).size


def directional_scores(y, p, direction):
    e = y-p
    rmse = math.sqrt(average(e*e))
    if rmse == 0:
        raise ValueError('PROBE_BLOCKED: zero native RMSE; registered derivative undefined')
    return {'native_rmse': rmse,
            'mae_derivative': average(np.sign(p-y)*direction+(y == p)*np.abs(direction)),
            'rmse_derivative': -average(e*direction)/rmse,
            'direction_mean_square': average(direction*direction),
            'zero_direction_fraction': average(direction == 0),
            'observed_atom_fraction': average(y == p)}


def gate(window_rows):
    checks = []
    for cut in (1056, 1392):
        by_rep = {r['representation']: r for r in window_rows if r['evaluation_cut'] == cut}
        if set(by_rep) != set(REPRESENTATIONS):
            raise ValueError('PROBE_BLOCKED: missing representation')
        candidate = by_rep['duration']
        controls = [r for rep, r in by_rep.items() if rep != 'duration' and r['mae_derivative'] < 0]
        reference = max((r['normalized_rmse_descent'] for r in controls), default=0.)
        conditions = {'nonzero_direction': candidate['direction_norm'] > 0,
                      'mae_strict_descent': candidate['mae_derivative'] < 0,
                      'rmse_strict_descent': candidate['rmse_derivative'] < 0,
                      'better_than_compatible_controls': candidate['normalized_rmse_descent'] > reference}
        checks.append({'evaluation_cut': cut, 'conditions': conditions,
                       'compatible_controls': [r['representation'] for r in controls],
                       'reference_efficiency': reference, 'passed': all(conditions.values())})
    supported = all(c['passed'] for c in checks)
    return {'status': ('MECHANISM_SUPPORTED_FOR_NEW_PROTOCOL_DESIGN' if supported else
                       'MECHANISM_NOT_SUPPORTED_IN_REGISTERED_REPRESENTATION'),
            'window_checks': checks, 'old_information_gate': False, 'old_structure_gate': False,
            'sota_claim': False, 'automatic_stage_advance': False}


def execute(arrays):
    moment_rows, cell_rows, window_rows, fits = [], [], [], []
    for cut, training_cuts in ((1056, (720,)), (1392, (720, 1056))):
        for rep in REPRESENTATIONS:
            training = []
            for train_cut in training_cuts:
                for h in (3, 12):
                    x, p = design(arrays, train_cut, h, rep)
                    y = arrays[train_cut, h, 'truth'].ravel()
                    training.append((x, moment_targets(y, p)))
            model = fit_ridge(training)
            fits.append({'evaluation_cut': cut, 'representation': rep, 'training_cuts': list(training_cuts),
                         'training_rows': model['training_rows'], 'training_cells': model['training_cells']})
            scores = []
            for h in (3, 12):
                x, p = design(arrays, cut, h, rep)
                y = arrays[cut, h, 'truth'].ravel()
                mean, probabilities = predict_moments(model, x)
                direction = common_direction(mean, probabilities, p)
                actual = moment_targets(y, p)
                score = directional_scores(y, p, direction)
                identity = {'representation': rep, 'evaluation_cut': cut, 'horizon': h}
                moment_rows.append({**identity, 'mean_residual_mse': average((mean-actual[:, 0])**2),
                                    'mean_residual_bias': average(mean-actual[:, 0]),
                                    'multiclass_brier': average(np.sum((probabilities-actual[:, 1:])**2, axis=1)),
                                    'lower_probability_mse': average((probabilities[:, 0]-actual[:, 1])**2),
                                    'atom_probability_mse': average((probabilities[:, 1]-actual[:, 2])**2),
                                    'upper_probability_mse': average((probabilities[:, 2]-actual[:, 3])**2),
                                    'predicted_atom_mean': average(probabilities[:, 1])})
                cell_rows.append({**identity, **score})
                scores.append(score)
            macro = {k: math.fsum(s[k] for s in scores)/2 for k in scores[0]}
            norm = math.sqrt(macro['direction_mean_square'])
            window_rows.append({'representation': rep, 'evaluation_cut': cut, **macro,
                                'direction_norm': norm,
                                'normalized_rmse_descent': -macro['rmse_derivative']/norm if norm else 0.})
    if len(fits) != 8:
        raise ValueError('PROBE_BLOCKED: fit count')
    return {'gate': gate(window_rows), 'windows': window_rows, 'cells': cell_rows,
            'moment_scores': moment_rows, 'fits': fits}
