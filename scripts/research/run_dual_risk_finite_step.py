"""One new four-fit, fixed-grid, staged finite-step development experiment."""
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
import urbanev_forecast.dual_risk_finite_step as core

CONFIG = ROOT/'configs/research/DUAL_RISK_FINITE_STEP_V1.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_scores(path, selected, status='RUN'):
    fields = ['status', 'system', 'alpha', 'horizon', 'rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        if status == 'NOT_RUN':
            writer.writerow({'status': status})
        else:
            for system, point in selected.items():
                for cell in point['cells']:
                    writer.writerow({'status': status, 'system': system, 'alpha': point['alpha'],
                                     **{k: cell[k] for k in fields if k in cell}})


def write_grid(path, grid, selected, native):
    fields = ['system', 'alpha', 'scope', 'rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae', 'feasible', 'selected']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for system, candidates in grid.items():
            for point in candidates:
                for scope, row in [('H3', point['cells'][0]), ('H12', point['cells'][1]), ('macro', point['macro'])]:
                    writer.writerow({'system': system, 'alpha': point['alpha'], 'scope': scope,
                                     **{k: row[k] for k in ('rmse', 'mae', 'unclipped_move_rmse', 'unclipped_move_mae')},
                                     'feasible': core.step_feasible(point, native),
                                     'selected': point['alpha'] == selected[system]['alpha']})


class NoFoundationOrContinuous(importlib.abc.MetaPathFinder):
    attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'chronos', 'timesfm', 'transformers'}
                or fullname.startswith('urbanev_forecast.continuous')
                or fullname == 'urbanev_forecast.foundation'):
            self.attempts += 1
            raise RuntimeError('Foundation inference and continuous solver are forbidden')
        return None


def run_stages(cache, output, config_hash, assert_identity, monitor):
    fit_arrays = cache.fit_arrays()
    with threadpool_limits(limits=2):
        models = core.fit_only_720(fit_arrays)
    identities = {}
    for name, model in models.items():
        path = output/f'private_model_{name}.npz'
        np.savez(path, center=model['center'], scale=model['scale'], coefficient=model['coefficient'])
        identities[name] = {'file': path.name, 'sha256': digest(path),
                            'training_cut': 720, 'training_cells': 2, 'training_rows': model['training_rows']}
        with np.load(path, allow_pickle=False) as restored:
            for key in ('center', 'scale', 'coefficient'):
                if not np.array_equal(restored[key], model[key]):
                    raise ValueError('Saved model verification failed')
    cache.models_saved(identities)
    monitor['fitting_closed'] = True
    assert_identity()
    calibration = cache.calibration_arrays()
    p, v = core.fixed_directions(models, calibration, 1056)
    selected, grid = core.calibrate(calibration, p, v)
    cal_gate = core.compare_gate(selected)
    cal_decomp = core.locked_decompositions(calibration, 1056, p, v, selected)
    write_grid(output/'calibration_grid.csv', grid, selected, selected['native'])
    dump(output/'calibration_selected.json', selected)
    selection = {'config_sha256': config_hash, 'models': identities,
                 'alphas': {s: selected[s]['alpha'] for s in core.SYSTEMS},
                 'output_family': 'clip(clipped_native + alpha * frozen_direction,0,1)',
                 'calibration_information_gate': cal_gate['information_pass'],
                 'calibration_cut': 1056, 'fit_cut': 720, 'refit_after_selection': False}
    dump(output/'frozen_selection.json', selection)
    selection_hash = digest(output/'frozen_selection.json')
    if json.loads((output/'frozen_selection.json').read_text(encoding='utf-8')) != selection:
        raise ValueError('Selection write verification failed')
    decomposition = {'calibration': cal_decomp, 'evaluation': {'status': 'NOT_RUN'}}
    evaluation_gate = {'status': 'NOT_RUN', 'reason': 'CALIBRATION_INFORMATION_GATE_FAILED'}
    saved_predictions = []
    status = 'FINITE_STEP_CALIBRATION_NO_GO'
    if cal_gate['information_pass']:
        assert_identity()
        if any(digest(output/m['file']) != m['sha256'] for m in identities.values()):
            raise ValueError('Model changed after selection')
        evaluation_inputs = cache.evaluation_inputs((output/'frozen_selection.json').read_bytes(), selection_hash)
        ep, ev = core.fixed_directions(models, evaluation_inputs, 1392)
        saved_raw = {}
        for system in core.SYSTEMS:
            for h in (3, 12):
                raw = ep[h]+selection['alphas'][system]*ev[system][h]
                path = output/f'private_evaluation_{system}_h{h}.npy'
                np.save(path, raw.reshape(14, h, 275))
                restored = np.load(path, allow_pickle=False).ravel()
                if not np.array_equal(restored, raw):
                    raise ValueError('Saved evaluation prediction verification failed')
                saved_raw[system, h] = restored
                saved_predictions.append({'system': system, 'horizon': h, 'file': path.name, 'sha256': digest(path)})
        dump(output/'evaluation_predictions_frozen.json', {'selection_sha256': selection_hash, 'predictions': saved_predictions})
        assert_identity()
        if digest(output/'frozen_selection.json') != selection_hash:
            raise ValueError('Selection identity changed before truth release')
        truth = cache.evaluation_truth(saved_predictions)
        evaluation = core.evaluate_locked(truth, ep, ev, selection['alphas'], saved_raw)
        evaluation_gate = core.compare_gate(evaluation)
        decomposition['evaluation'] = core.locked_decompositions(truth, 1392, ep, ev, evaluation)
        write_scores(output/'evaluation_scores.csv', evaluation)
        dump(output/'evaluation_selected.json', evaluation)
        status = ('FINITE_STEP_DEV_PASS_REVIEW_REQUIRED' if evaluation_gate['information_pass']
                  else 'FINITE_STEP_EVALUATION_NO_GO')
    else:
        write_scores(output/'evaluation_scores.csv', {}, status='NOT_RUN')
    result = {'status': status, 'calibration': cal_gate, 'evaluation': evaluation_gate,
              'cross_window_confirmation': 'NOT_EVALUATED', 'full_four_horizon_confirmation': 'NOT_EVALUATED',
              'old_v1_v2_information_gate': False, 'old_v1_v2_structure_gate': False,
              'independent_confirmation': False, 'sota_claim': False, 'automatic_stage_advance': False}
    dump(output/'gate_decision.json', result)
    dump(output/'loss_decomposition.json', decomposition)
    return result, selection_hash, saved_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1-root', type=Path, required=True)
    parser.add_argument('--truth-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--claim', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh independent output directory required')
    config_bytes = CONFIG.read_bytes().replace(b'\r\n', b'\n')
    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if (config['id'] != 'DUAL_RISK_FINITE_STEP_V1_20260913' or config['grid'] != list(core.GRID)
            or config['systems'] != list(core.SYSTEMS) or config['max_new_multitarget_ridge_fits'] != 4):
        raise ValueError('Wrong frozen protocol')
    parent = ROOT/config['parent_manifest']
    if hashlib.sha256(parent.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != config['parent_manifest_sha256_lf']:
        raise ValueError('Parent identity changed')
    if config['files'] != json.loads(parent.read_text(encoding='utf-8'))['files']:
        raise ValueError('Source whitelist changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    def assert_identity():
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head:
            raise ValueError('Execution commit changed')
        if CONFIG.read_bytes().replace(b'\r\n', b'\n') != config_bytes:
            raise ValueError('Configuration changed')
        for name, expected in config['code_sha256_lf'].items():
            if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
                raise ValueError('Execution code identity changed')

    frozen = subprocess.check_output(['git', 'show', 'HEAD:configs/research/DUAL_RISK_FINITE_STEP_V1.json'], cwd=ROOT)
    if frozen.replace(b'\r\n', b'\n') != config_bytes:
        raise ValueError('Configuration not committed before execution')
    assert_identity()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *config['code_sha256_lf']], cwd=ROOT, check=True)
    args.claim.parent.mkdir(parents=True, exist_ok=True)
    with args.claim.open('x', encoding='utf-8') as f:
        json.dump({'id': config['id'], 'execution_commit': head, 'status': 'CLAIMED_ONCE'}, f)
    args.output.mkdir()
    dump(args.output/'preregistration.json', config)
    monitor = {'ridge_solves': 0, 'formal_cell_scores': 0, 'fitting_closed': False}
    guard = NoFoundationOrContinuous()
    sys.meta_path.insert(0, guard)
    original_solve, original_score = np.linalg.solve, core.score_cell
    started = time.perf_counter()
    cache = None

    def solve(*a, **kw):
        if monitor['fitting_closed'] or monitor['ridge_solves'] >= 4:
            raise ValueError('New fit after FIT or fit budget exceeded')
        monitor['ridge_solves'] += 1
        return original_solve(*a, **kw)

    def score(*a, **kw):
        monitor['formal_cell_scores'] += 1
        if monitor['formal_cell_scores'] > 116:
            raise ValueError('Grid/evaluation score budget exceeded')
        return original_score(*a, **kw)

    try:
        cache = core.StagedCache(config['files'], {'v1': args.v1_root, 'truth': args.truth_root}, config_hash)
        with threadpool_limits(limits=2), patch.object(np.linalg, 'solve', solve), patch.object(core, 'score_cell', score):
            result, selection_hash, prediction_hashes = run_stages(cache, args.output, config_hash, assert_identity, monitor)
        assert_identity()
        assert monitor['ridge_solves'] == 4 and guard.attempts == 0
        decoded = [x for x in cache.ledger if x['kind'] == 'numeric_decode']
        receipt = {'id': config['id'], 'status': result['status'], 'execution_commit': head,
                   'config_sha256_lf': config_hash, 'frozen_selection_sha256': selection_hash,
                   'monitor': monitor, 'forbidden_import_attempts': guard.attempts,
                   'read_ledger': cache.ledger, 'numeric_array_count': len(decoded),
                   'opaque_array_hash_count': 36, 'max_decoded_target_index': max(x['max_target_index'] for x in decoded),
                   'prediction_artifact_hashes': prediction_hashes,
                   'calibration_grid_combinations': 56, 'calibration_unique_horizon_entries': 112,
                   'alpha0_scores_reused': True, 'evaluation_grid_searches': 0,
                   'new_foundation_inference': 0, 'raw_data_files_read': 0, 'refit_after_selection': False,
                   'upstream_cache_parameters_vary_by_cut': True, 'post_hoc_development': True,
                   'elapsed_seconds': time.perf_counter()-started, 'threads': 2,
                   'timing_scope': 'cache hashes, staged decoding, four fits, grid, conditional evaluation, exports; excludes preflight and receipt write',
                   'peak_process_memory_bytes': None, 'peak_memory_status': 'NOT_MEASURED',
                   'quota_reset_used': False}
        dump(args.output/'execution_receipt.json', receipt)
        print(json.dumps({'status': result['status'], 'fits': 4, 'decoded_arrays': len(decoded),
                          'formal_scores': monitor['formal_cell_scores'], 'seconds': receipt['elapsed_seconds']}))
    except Exception as exc:
        dump(args.output/'execution_receipt.json', {'status': 'FINITE_STEP_BLOCKED', 'error_type': type(exc).__name__,
             'error': str(exc), 'execution_commit': head, 'monitor': monitor,
             'read_ledger': cache.ledger if cache else [], 'automatic_retry': False})
        raise
    finally:
        sys.meta_path.remove(guard)


if __name__ == '__main__':
    main()
