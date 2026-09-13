"""Execute one separately registered dynamic-volume information experiment."""
import argparse
import csv
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import urbanev_forecast.volume_composition as core
from urbanev_forecast.volume_inputs import VolumeInputs, LIMITS

CONFIG = ROOT/'configs/research/VOLUME_COMPOSITION_INCREMENT_V1.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def grid_csv(path, grid, selected):
    names = ['system', 'alpha', 'scope', 'rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae', 'feasible', 'selected']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names, lineterminator='\n'); writer.writeheader()
        for system, points in grid.items():
            for point in points:
                for label, row in [('H3', point['cells'][0]), ('H12', point['cells'][1]), ('macro', point['macro'])]:
                    writer.writerow({'system': system, 'alpha': point['alpha'], 'scope': label,
                                     **{k: row[k] for k in names[3:7]},
                                     'feasible': core.step_feasible(point, selected['native']),
                                     'selected': selected[system]['alpha'] == point['alpha']})


def evaluation_csv(path, selected=None):
    names = ['status', 'system', 'alpha', 'horizon', 'rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names, lineterminator='\n'); writer.writeheader()
        if selected is None:
            writer.writerow({'status': 'NOT_RUN'})
        else:
            for system, point in selected.items():
                for row in point['cells']:
                    writer.writerow({'status': 'RUN', 'system': system, 'alpha': point['alpha'],
                                     **{k: row[k] for k in names[3:]}})


def save_fixed(output, stage, p, u, alphas):
    records, restored = [], {}
    for system in core.SYSTEMS:
        for h in (3, 12):
            move = p[h]+alphas[system]*u[system][h]
            for role, a in [('direction', u[system][h]), ('unclipped_move', move)]:
                path = output/f'private_{stage}_{system}_h{h}_{role}.npy'
                np.save(path, a.reshape(14, h, 275))
                read = np.load(path, allow_pickle=False).ravel()
                if not np.array_equal(read, a):
                    raise ValueError('Saved prediction/direction changed')
                if role == 'unclipped_move':
                    restored[system, h] = read
                    records.append({'system': system, 'horizon': h, 'file': path.name, 'sha256': digest(path)})
    return records, restored


def run(inputs, output, config_hash, assert_identity, monitor):
    raw, cache = inputs.fit()
    power, summary = core.historical_power(raw['duration'], raw['volume'])
    power_path = output/'private_power_reference.npy'
    np.save(power_path, power)
    if not np.array_equal(np.load(power_path, allow_pickle=False), power):
        raise ValueError('Power reference write mismatch')
    power_identity = {'file': power_path.name, 'sha256': digest(power_path), 'reference_range': [0, 719]}
    derived, consistency = core.derive_series(raw['duration'], raw['volume'], inputs.capacities, power)
    models = core.fit_models(raw, derived, inputs.capacities, power, cache)
    monitor['fitting_closed'] = True
    identities, dimensions = {}, {}
    for system, model in models.items():
        path = output/f'private_model_{system}.npz'
        np.savez(path, **{k: model[k] for k in ('center', 'scale', 'inactive', 'coefficient')})
        with np.load(path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k], model[k]) for k in saved.files):
                raise ValueError('Model write verification failed')
        identities[system] = {'file': path.name, 'sha256': digest(path), 'fit_cut': 720}
        dimensions[system] = {'rows': model['rows'], 'columns': model['columns'],
                              'zeroed_nonintercept_columns': int(model['inactive'].sum())}
    summary = {'power_reference': summary, 'consistency': {'FIT': consistency}, 'designs': dimensions,
               'OD_VOLUME_dynamic_history_training_scales': models['OD_VOLUME']['scale'][-11:].tolist(),
               'OD_VOLUME_dynamic_history_zeroed_columns': int(models['OD_VOLUME']['inactive'][-11:].sum()),
               'reference_vector_private': True, 'dynamic_z_not_statistically_orthogonal': True,
               'static_controls_use_early_historical_volume': True, 'source_units_and_live_semantics_unverified': True}
    inputs.freeze_models(identities, power_identity)
    assert_identity()
    raw, cache = inputs.calibration()
    derived, consistency = core.derive_series(raw['duration'], raw['volume'], inputs.capacities, power)
    summary['consistency']['CALIBRATE'] = consistency
    p, u = core.directions(models, raw, derived, inputs.capacities, power, cache, 1056)
    selected, grid = core.calibrate(cache, p, u)
    calibration_gate = core.gate(selected)
    decomposition = {'calibration': core.decompositions(cache, 1056, p, u, selected), 'evaluation': {'status': 'NOT_RUN'}}
    alphas = {s: selected[s]['alpha'] for s in core.SYSTEMS}
    save_fixed(output, 'calibration', p, u, alphas)
    grid_csv(output/'calibration_grid.csv', grid, selected)
    dump(output/'calibration_selected.json', selected)
    selection = {'config_sha256': config_hash, 'models': identities, 'power_reference': power_identity,
                 'alphas': alphas, 'fit_cut': 720, 'calibration_cut': 1056,
                 'calibration_information_gate': calibration_gate['passed'],
                 'output_family': 'clip(clipped_native + alpha * plain_ridge_direction,0,1)',
                 'refit_after_selection': False}
    dump(output/'frozen_selection.json', selection)
    selection_hash = digest(output/'frozen_selection.json')
    if json.loads((output/'frozen_selection.json').read_text(encoding='utf-8')) != selection:
        raise ValueError('Frozen selection write mismatch')
    evaluation_gate = {'status': 'NOT_RUN', 'reason': 'CALIBRATION_GATE_FAILED'}
    status = 'VOLUME_INCREMENT_CALIBRATION_NO_GO'
    if calibration_gate['passed']:
        assert_identity()
        if digest(power_path) != power_identity['sha256'] or any(digest(output/m['file']) != m['sha256'] for m in identities.values()):
            raise ValueError('Frozen model/reference changed')
        raw, cache = inputs.evaluation_inputs((output/'frozen_selection.json').read_bytes(), selection_hash)
        derived, consistency = core.derive_series(raw['duration'], raw['volume'], inputs.capacities, power)
        summary['consistency']['EVALUATE_INPUTS'] = consistency
        ep, eu = core.directions(models, raw, derived, inputs.capacities, power, cache, 1392)
        prediction_records, predictions = save_fixed(output, 'evaluation', ep, eu, alphas)
        dump(output/'evaluation_predictions_frozen.json', {'selection_sha256': selection_hash, 'predictions': prediction_records})
        assert_identity()
        if digest(output/'frozen_selection.json') != selection_hash:
            raise ValueError('Selection changed before evaluation truth')
        cache.update(inputs.evaluation_truth(prediction_records))
        evaluation = {}
        for system in core.SYSTEMS:
            cells = []
            for h in (3, 12):
                if not np.array_equal(predictions[system, h], ep[h]+alphas[system]*eu[system][h]):
                    raise ValueError('Prediction changed after label release')
                cells.append(dict(horizon=h, **core.score_cell(cache[1392, h, 'truth'].ravel(), ep[h], eu[system][h], alphas[system], decompose=False)))
            evaluation[system] = {'alpha': alphas[system], 'cells': cells, 'macro': core.macro_score(cells)}
        evaluation_gate = core.gate(evaluation)
        decomposition['evaluation'] = core.decompositions(cache, 1392, ep, eu, evaluation)
        dump(output/'evaluation_selected.json', evaluation)
        evaluation_csv(output/'evaluation_scores.csv', evaluation)
        status = 'VOLUME_INCREMENT_DEV_PASS_REVIEW_REQUIRED' if evaluation_gate['passed'] else 'VOLUME_INCREMENT_EVALUATION_NO_GO'
    else:
        evaluation_csv(output/'evaluation_scores.csv')
    result = {'status': status, 'calibration': calibration_gate, 'evaluation': evaluation_gate,
              'new_structure_claim_tested': False, 'old_information_and_structure_gates': 'UNCHANGED',
              'cross_window_confirmation': 'NOT_EVALUATED', 'four_horizon_confirmation': 'NOT_EVALUATED',
              'independent_confirmation': False, 'sota_claim': False, 'automatic_stage_advance': False}
    dump(output/'feature_construction_summary.json', summary)
    dump(output/'gate_decision.json', result)
    dump(output/'loss_decomposition.json', decomposition)
    return result, selection_hash


class ForbiddenImports(importlib.abc.MetaPathFinder):
    attempts = 0
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'chronos', 'timesfm', 'transformers'} or fullname.startswith('urbanev_forecast.continuous') or fullname == 'urbanev_forecast.foundation':
            self.attempts += 1
            raise RuntimeError('Foundation/continuous solver forbidden')
        return None


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'v1-root', 'truth-root', 'output', 'claim'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New independent output required')
    content = CONFIG.read_bytes().replace(b'\r\n', b'\n')
    config = json.loads(content); config_hash = hashlib.sha256(content).hexdigest()
    if (config['id'] != 'VOLUME_COMPOSITION_INCREMENT_V1_20260913' or config['grid'] != list(core.GRID)
            or config['systems'] != list(core.SYSTEMS) or config['stage_limits'] != LIMITS):
        raise ValueError('Wrong frozen specification')
    parent = ROOT/config['cache_parent_manifest']
    if hashlib.sha256(parent.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != config['cache_parent_manifest_sha256_lf']:
        raise ValueError('Parent manifest changed')
    old = [r for r in json.loads(parent.read_text(encoding='utf-8'))['files'] if r['system'] in ('native', 'truth')]
    if old != config['cache_files']:
        raise ValueError('Cache identity changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frozen = subprocess.check_output(['git', 'show', 'HEAD:configs/research/VOLUME_COMPOSITION_INCREMENT_V1.json'], cwd=ROOT)
    if frozen.replace(b'\r\n', b'\n') != content:
        raise ValueError('Configuration not committed')
    def assert_identity():
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head or CONFIG.read_bytes().replace(b'\r\n', b'\n') != content:
            raise ValueError('Execution/configuration changed')
        for file, expected in config['code_sha256_lf'].items():
            if hashlib.sha256((ROOT/file).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
                raise ValueError('Code identity changed')
    assert_identity()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *config['code_sha256_lf']], cwd=ROOT, check=True)
    args.claim.parent.mkdir(parents=True, exist_ok=True)
    with args.claim.open('x', encoding='utf-8') as f:
        json.dump({'id': config['id'], 'execution_commit': head, 'status': 'CLAIMED_ONCE'}, f)
    args.output.mkdir()
    dump(args.output/'preregistration.json', config)
    monitor = {'ridge_solves': 0, 'power_reference_passes': 0, 'formal_scores': 0, 'fitting_closed': False}
    original_solve, original_ref, original_score = np.linalg.solve, core.historical_power, core.score_cell
    def solve(*a, **k):
        if monitor['fitting_closed'] or monitor['ridge_solves'] >= 5:
            raise ValueError('Supervised fit budget/stage exceeded')
        monitor['ridge_solves'] += 1
        return original_solve(*a, **k)
    def reference(*a, **k):
        if monitor['power_reference_passes'] or monitor['fitting_closed']:
            raise ValueError('Power reference can only be computed once before fitting closes')
        monitor['power_reference_passes'] += 1
        return original_ref(*a, **k)
    def score(*a, **k):
        monitor['formal_scores'] += 1
        if monitor['formal_scores'] > 84:
            raise ValueError('Fixed scoring budget exceeded')
        return original_score(*a, **k)
    guard = ForbiddenImports();sys.meta_path.insert(0, guard)
    inputs = None;started = time.perf_counter()
    try:
        inputs = VolumeInputs(config, args.data_root, {'v1': args.v1_root, 'truth': args.truth_root}, config_hash)
        with threadpool_limits(limits=2), patch.object(np.linalg, 'solve', solve), patch.object(core, 'historical_power', reference), patch.object(core, 'score_cell', score):
            result, selection_hash = run(inputs, args.output, config_hash, assert_identity, monitor)
        assert_identity()
        assert monitor['ridge_solves'] == 5 and monitor['power_reference_passes'] == 1 and guard.attempts == 0
        receipt = {'id': config['id'], 'status': result['status'], 'execution_commit': head,
                   'config_sha256_lf': config_hash, 'selection_sha256': selection_hash, 'monitor': monitor,
                   'read_ledger': inputs.ledger, 'cache_arrays_decoded': len(inputs.decoded),
                   'new_foundation_inference': 0, 'forbidden_import_attempts': guard.attempts,
                   'post_hoc_development': True, 'known_future_inputs': [], 'live_availability_established': False,
                   'elapsed_seconds': time.perf_counter()-started, 'threads': 2,
                   'timing_scope': 'bounded decoding/lineage checks, one reference, five fits, calibration and conditional evaluation, exports; excludes preflight/receipt',
                   'peak_memory_status': 'NOT_MEASURED', 'quota_reset_used': False}
        dump(args.output/'execution_receipt.json', receipt)
        print(json.dumps({'status': result['status'], 'fits': monitor['ridge_solves'], 'reference_passes': monitor['power_reference_passes'], 'scores': monitor['formal_scores'], 'seconds': receipt['elapsed_seconds']}))
    except Exception as exc:
        dump(args.output/'execution_receipt.json', {'status': 'VOLUME_INCREMENT_BLOCKED', 'reason_type': type(exc).__name__,
             'reason': str(exc), 'execution_commit': head, 'monitor': monitor, 'read_ledger': inputs.ledger if inputs else [], 'automatic_retry': False})
        raise
    finally:
        sys.meta_path.remove(guard)


if __name__ == '__main__':
    main()
