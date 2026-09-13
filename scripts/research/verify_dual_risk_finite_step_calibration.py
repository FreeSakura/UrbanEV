"""Recompute existing calibration points from saved models; never fit or select."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from urbanev_forecast.dual_risk_finite_step import SYSTEMS, GRID, fixed_directions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--v1-root', type=Path, required=True)
    parser.add_argument('--truth-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.archive.exists():
        raise FileExistsError('Fresh verification artifacts required')
    manifest = json.loads((args.run/'preregistration.json').read_text(encoding='utf-8'))
    selection = json.loads((args.run/'frozen_selection.json').read_text(encoding='utf-8'))
    receipt = json.loads((args.run/'execution_receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] == 'FINITE_STEP_BLOCKED' or digest(args.run/'frozen_selection.json') != receipt['frozen_selection_sha256']:
        raise ValueError('Incomplete or changed selected result')
    sources = [r for r in manifest['files'] if r['cut'] == 1056]
    assert len(sources) == 12
    arrays, ledger = {}, []
    for row in sources:
        root = (args.truth_root if row['system'] == 'truth' else args.v1_root).resolve()
        path = (root/row['file']).resolve()
        if path.parent != root or digest(path) != row['sha256']:
            raise ValueError('Source identity mismatch')
        a = np.load(path, allow_pickle=False).astype(float)
        assert a.shape == (14, row['horizon'], 275)
        arrays[1056, row['horizon'], row['system']] = a
        ledger.append({'file': row['file'], 'sha256': row['sha256'], 'cut': 1056})
    models = {}
    for rep, record in selection['models'].items():
        path = args.run/record['file']
        if digest(path) != record['sha256']:
            raise ValueError('Frozen model changed')
        with np.load(path, allow_pickle=False) as model:
            models[rep] = {k: model[k].copy() for k in ('center', 'scale', 'coefficient')}
    with threadpool_limits(limits=2):
        points, directions = fixed_directions(models, arrays, 1056)
    with (args.run/'calibration_grid.csv').open(encoding='utf-8') as f:
        grid = {(r['system'], float(r['alpha']), r['scope']): r for r in csv.DictReader(f)}
    expected = {(s, a, scope) for s in SYSTEMS[1:] for a in GRID for scope in ('H3', 'H12', 'macro')}
    assert set(grid) == expected
    maximum = 0.
    args.archive.mkdir()
    archival_hashes = []
    selected_scores = json.loads((args.run/'calibration_selected.json').read_text(encoding='utf-8'))
    for system in SYSTEMS:
        for alpha in ((0.,) if system == 'native' else GRID):
            actual = []
            for i, h in enumerate((3, 12)):
                raw = points[h]+alpha*directions[system][h]
                y = arrays[1056, h, 'truth'].ravel()
                q = np.clip(raw, 0., 1.)
                row = {'rmse': float(np.sqrt(np.mean((q-y)**2))), 'mae': float(np.mean(abs(q-y))),
                       'unclipped_move_rmse': float(np.sqrt(np.mean((raw-y)**2))),
                       'unclipped_move_mae': float(np.mean(abs(raw-y)))}
                expected_row = (selected_scores['native']['cells'][i] if system == 'native' else grid[system, alpha, f'H{h}'])
                maximum = max(maximum, max(abs(v-float(expected_row[k])) for k, v in row.items()))
                actual.append(row)
                if alpha == selection['alphas'][system]:
                    for role, value in [('direction', directions[system][h]), ('selected_unclipped_move', raw)]:
                        path = args.archive/f'private_calibration_{system}_h{h}_{role}.npy'
                        np.save(path, value.reshape(14, h, 275))
                        archival_hashes.append({'file': path.name, 'sha256': digest(path)})
                    expected_selected = selected_scores[system]['cells'][i]
                    maximum = max(maximum, max(abs(v-float(expected_selected[k])) for k, v in row.items()))
            if system != 'native':
                macro = {k: sum(r[k] for r in actual)/2 for k in actual[0]}
                maximum = max(maximum, max(abs(v-float(grid[system, alpha, 'macro'][k])) for k, v in macro.items()))
    assert maximum <= 1e-10
    assert not any(r.get('cut') == 1392 for r in receipt['read_ledger'] if r['kind'] == 'numeric_decode') if receipt['status'] == 'FINITE_STEP_CALIBRATION_NO_GO' else True
    result = {'status': 'PASS', 'scope': 'Independent NumPy reductions at already registered calibration points; no new fit or selection',
              'tunable_points_recomputed': 56, 'native_points_recomputed': 1,
              'maximum_metric_difference': maximum, 'source_numeric_arrays_read': 12,
              'source_ledger': ledger, 'max_source_target_index': 1223,
              'saved_models_loaded': 4, 'new_ridge_fits': 0, 'new_foundation_inference': 0,
              'evaluation_arrays_decoded': 0, 'original_selection_unchanged': True,
              'private_selected_artifact_hashes': archival_hashes}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'maximum_metric_difference': maximum, 'new_fits': 0}))


if __name__ == '__main__':
    main()
