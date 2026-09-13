"""Registered finite grid experiment, separated from the earlier direction probe."""
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np

from .dual_risk_probe import SYSTEMS as SOURCE_SYSTEMS, REPRESENTATIONS
from .dual_risk_probe import design, fit_ridge, moment_targets, predict_moments, common_direction
from .finite_step_metrics import score_cell, macro_score, decompose_cell

GRID = (0., .125, .25, .5, 1., 2., 4., 8.)
SYSTEMS = ('native', 'mean_base', 'gate_base', 'mean_duration', 'hard_duration',
           'gate_duration', 'gate_duplicate', 'gate_permuted')
NON_DURATION = ('native', 'mean_base', 'gate_base', 'gate_duplicate')
PRIMARY = 'gate_duration'


class StagedCache:
    def __init__(self, records, roots, config_sha256):
        expected = {(c, h, s) for c in (720, 1056, 1392) for h in (3, 12)
                    for s in ('truth',)+SOURCE_SYSTEMS}
        keys = [(x['cut'], x['horizon'], x['system']) for x in records]
        if len(keys) != 36 or set(keys) != expected:
            raise ValueError('Invalid36-file whitelist')
        self.blobs, self.rows, self.ledger = {}, {}, []
        self.decoded = set()
        self.state = 'PRECHECK'
        self.config_sha256 = config_sha256
        for row in records:
            c, h, s = row['cut'], row['horizon'], row['system']
            role = 'truth' if s == 'truth' else 'v1'
            name = f'private_truth_{c}_h{h}.npy' if s == 'truth' else f'private_{c}_h{h}_{s}_a1.npy'
            if row['file'] != name or row['root_role'] != role or row['shape'] != [14, h, 275]:
                raise ValueError('Invalid source identity or shape')
            root = Path(roots[role]).resolve()
            path = (root/name).resolve()
            if path.parent != root:
                raise ValueError('Source outside allowed root')
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != row['sha256']:
                raise ValueError('Source hash mismatch')
            self.blobs[c, h, s] = blob
            self.rows[c, h, s] = row
            self.ledger.append({'kind': 'opaque_byte_hash', 'file': name, 'sha256': digest,
                                'bytes': len(blob), 'numeric_values_decoded': False})

    def _decode(self, cut, systems):
        allowed = {'FIT': (720, set(('truth',)+SOURCE_SYSTEMS)),
                   'CALIBRATE': (1056, set(('truth',)+SOURCE_SYSTEMS)),
                   'EVALUATE_INPUTS': (1392, set(SOURCE_SYSTEMS)),
                   'EVALUATE_TRUTH': (1392, {'truth'})}
        if self.state not in allowed or (cut, set(systems)) != allowed[self.state]:
            raise ValueError('Numeric decode outside current stage')
        if any((cut, h, s) in self.decoded for h in (3, 12) for s in systems):
            raise ValueError('Array already numerically decoded')
        arrays = {}
        for h in (3, 12):
            for s in systems:
                key = cut, h, s
                a = np.load(io.BytesIO(self.blobs[key]), allow_pickle=False)
                if a.shape != (14, h, 275) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
                    raise ValueError('Invalid source array')
                if s == 'truth' and ((a < 0) | (a > 1)).any():
                    raise ValueError('Truth outside[0,1]')
                arrays[key] = a.astype(np.float64)
                self.decoded.add(key)
                self.ledger.append({'kind': 'numeric_decode', 'stage': self.state,
                                    'file': self.rows[key]['file'], 'cut': cut, 'horizon': h,
                                    'system': s, 'shape': list(a.shape), 'max_target_index': cut+156+h-1})
        return arrays

    def fit_arrays(self):
        if self.state != 'PRECHECK':
            raise ValueError('Wrong FIT stage')
        self.state = 'FIT'
        return self._decode(720, ('truth',)+SOURCE_SYSTEMS)

    def models_saved(self, identities):
        if self.state != 'FIT' or set(identities) != set(REPRESENTATIONS):
            raise ValueError('Four saved models required before CALIBRATE')
        self.model_identities = identities
        self.state = 'MODELS_FROZEN'

    def calibration_arrays(self):
        if self.state != 'MODELS_FROZEN':
            raise ValueError('Wrong CALIBRATE stage')
        self.state = 'CALIBRATE'
        return self._decode(1056, ('truth',)+SOURCE_SYSTEMS)

    def evaluation_inputs(self, selection_bytes, expected_hash):
        if self.state != 'CALIBRATE':
            raise ValueError('Wrong EVALUATE stage')
        if hashlib.sha256(selection_bytes).hexdigest() != expected_hash:
            raise ValueError('Frozen selection hash mismatch')
        selected = json.loads(selection_bytes)
        if (selected['calibration_information_gate'] is not True
                or selected['config_sha256'] != self.config_sha256
                or selected['models'] != self.model_identities
                or set(selected['alphas']) != set(SYSTEMS)
                or any(a not in GRID for a in selected['alphas'].values())):
            raise ValueError('Evaluation not admitted by complete frozen selection')
        self.selection_hash = expected_hash
        self.state = 'EVALUATE_INPUTS'
        return self._decode(1392, SOURCE_SYSTEMS)

    def evaluation_truth(self, saved_predictions):
        expected = {(s, h) for s in SYSTEMS for h in (3, 12)}
        if (self.state != 'EVALUATE_INPUTS' or len(saved_predictions) != 16
                or {(x['system'], x['horizon']) for x in saved_predictions} != expected
                or not all(len(x['sha256']) == 64 for x in saved_predictions)):
            raise ValueError('All16 fixed predictions must be saved before evaluation truth')
        self.state = 'EVALUATE_TRUTH'
        return self._decode(1392, ('truth',))


def fit_only_720(arrays):
    models = {}
    for rep in REPRESENTATIONS:
        cells = []
        for h in (3, 12):
            x, p = design(arrays, 720, h, rep)
            cells.append((x, moment_targets(arrays[720, h, 'truth'].ravel(), p)))
        models[rep] = fit_ridge(cells, regularization=.01)
    return models


def tangent(p, v):
    v = np.where((p == 0) & (v < 0), 0., v)
    return np.where((p == 1) & (v > 0), 0., v)


def fixed_directions(models, arrays, cut):
    points, directions = {}, {s: {} for s in SYSTEMS}
    for h in (3, 12):
        for rep in REPRESENTATIONS:
            x, p = design(arrays, cut, h, rep)
            m, probabilities = predict_moments(models[rep], x)
            points[h] = p
            g = common_direction(m, probabilities, p)
            directions['gate_'+rep][h] = g
            if rep in ('base', 'duration'):
                directions['mean_'+rep][h] = tangent(p, np.clip(m, -1., 1.))
            if rep == 'duration':
                k = np.maximum(-(probabilities[:, 0]-probabilities[:, 2])*np.sign(m)-probabilities[:, 1], 0.)
                directions['hard_duration'][h] = tangent(p, np.clip(m, -1., 1.)*(k > 0))
        directions['native'][h] = np.zeros_like(points[h])
    return points, directions


def step_feasible(point, native):
    return (point['macro']['mae'] <= native['macro']['mae']
            and all(point['cells'][i]['rmse'] <= 1.01*native['cells'][i]['rmse'] for i in (0, 1)))


def choose_step(points, native):
    if tuple(p['alpha'] for p in points) != GRID:
        raise ValueError('Complete frozen grid required')
    if any(points[0]['macro'][k] != native['macro'][k] for k in ('rmse', 'mae')):
        raise ValueError('Alpha0 does not recover native')
    feasible = [p for p in points if step_feasible(p, native)]
    if not feasible or not step_feasible(points[0], native):
        raise ValueError('Alpha0 must be feasible')
    best = min(p['macro']['rmse'] for p in feasible)
    chosen = min((p for p in feasible if p['macro']['rmse'] <= best+1e-12), key=lambda p: p['alpha'])
    return {**chosen, 'feasible_alphas': [p['alpha'] for p in feasible],
            'minimum_feasible_macro_rmse': best, 'tie_tolerance': 1e-12,
            'selection_status': ('GRID_ZERO_ONLY_FEASIBLE' if len(feasible) == 1 else
                                 'SELECTED_ZERO' if chosen['alpha'] == 0 else 'SELECTED_NONZERO')}


def calibrate(arrays, points, directions):
    native_cells = [dict(horizon=h, **score_cell(arrays[1056, h, 'truth'].ravel(), points[h],
                          directions['native'][h], 0., decompose=False)) for h in (3, 12)]
    native = {'alpha': 0., 'cells': native_cells, 'macro': macro_score(native_cells)}
    selected, grid = {'native': native}, {}
    for system in SYSTEMS[1:]:
        candidates = []
        for alpha in GRID:
            cells = (native_cells if alpha == 0 else
                     [dict(horizon=h, **score_cell(arrays[1056, h, 'truth'].ravel(), points[h],
                           directions[system][h], alpha, decompose=False)) for h in (3, 12)])
            candidates.append({'alpha': alpha, 'cells': cells, 'macro': macro_score(cells)})
        grid[system] = candidates
        selected[system] = choose_step(candidates, native)
    return selected, grid


def compare_gate(selected):
    if set(selected) != set(SYSTEMS):
        raise ValueError('All8 systems are required')
    reference_name = min(NON_DURATION, key=lambda s: selected[s]['macro']['rmse'])
    primary, native = selected[PRIMARY], selected['native']
    reference, permuted = selected[reference_name], selected['gate_permuted']
    pm, nm, rm, xm = (r['macro'] for r in (primary, native, reference, permuted))
    conditions = {
        'positive_selected_alpha': primary['alpha'] > 0,
        'reference_has_positive_rmse': rm['rmse'] > 0,
        'macro_rmse_gain_at_least1percent': rm['rmse'] > 0 and pm['rmse'] <= .99*rm['rmse'],
        'macro_mae_no_worse_than_native': pm['mae'] <= nm['mae'],
        'macro_mae_no_worse_than_Cstar': pm['mae'] <= rm['mae'],
        'beats_permuted_rmse': pm['rmse'] < xm['rmse'],
        'mae_no_worse_than_permuted': pm['mae'] <= xm['mae'],
    }
    for i, h in enumerate((3, 12)):
        conditions[f'h{h}_rmse_harm_native_at_most1percent'] = primary['cells'][i]['rmse'] <= 1.01*native['cells'][i]['rmse']
        conditions[f'h{h}_rmse_harm_Cstar_at_most1percent'] = primary['cells'][i]['rmse'] <= 1.01*reference['cells'][i]['rmse']
    method = {}
    for name in ('mean_duration', 'hard_duration'):
        control = selected[name]
        tests = {'positive_reference_rmse': control['macro']['rmse'] > 0,
                 'macro_rmse_gain_at_least1percent': control['macro']['rmse'] > 0 and pm['rmse'] <= .99*control['macro']['rmse'],
                 'macro_mae_no_worse': pm['mae'] <= control['macro']['mae']}
        for i, h in enumerate((3, 12)):
            tests[f'h{h}_rmse_harm_at_most1percent'] = primary['cells'][i]['rmse'] <= 1.01*control['cells'][i]['rmse']
        method[name] = {'conditions': tests, 'passed': all(tests.values())}
    return {'Cstar': reference_name, 'conditions': conditions, 'information_pass': all(conditions.values()),
            'rmse_gain_percent_vs_Cstar': 100*(1-pm['rmse']/rm['rmse']) if rm['rmse'] else None,
            'primary_macro': pm, 'reference_macro': rm, 'method_comparisons': method,
            'method_pass': all(x['passed'] for x in method.values())}


def locked_decompositions(arrays, cut, points, directions, selected):
    rows = []
    for system in SYSTEMS:
        for i, h in enumerate((3, 12)):
            row = decompose_cell(arrays[cut, h, 'truth'].ravel(), points[h], directions[system][h],
                                 selected[system]['alpha'], selected[system]['cells'][i])
            if max(abs(row[k]) for k in ('raw_mae_identity_error', 'clipped_mae_identity_error', 'rmse_identity_error')) > 1e-10:
                raise ValueError('Finite-step loss identity verification failed')
            rows.append({'cut': cut, 'horizon': h, 'system': system, 'alpha': selected[system]['alpha'], **row})
    return rows


def evaluate_locked(truth, points, directions, alphas, saved_raw_predictions):
    if set(alphas) != set(SYSTEMS) or any(a not in GRID for a in alphas.values()):
        raise ValueError('Evaluation requires one frozen alpha for every system')
    selected = {}
    for system in SYSTEMS:
        cells = []
        for h in (3, 12):
            raw = points[h]+alphas[system]*directions[system][h]
            if not np.array_equal(raw, saved_raw_predictions[system, h]):
                raise ValueError('Evaluation prediction changed after label release')
            cells.append(dict(horizon=h, **score_cell(truth[1392, h, 'truth'].ravel(), points[h],
                        directions[system][h], alphas[system], decompose=False)))
        selected[system] = {'alpha': alphas[system], 'cells': cells, 'macro': macro_score(cells)}
    return selected
