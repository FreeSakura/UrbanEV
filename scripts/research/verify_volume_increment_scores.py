"""Independent reductions of already registered points from archived directions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SYSTEMS = ('native', 'OD', 'OD_STATIC', 'OD_DUPLICATE', 'OD_VOLUME', 'OD_MISALIGNED')
GRID = (0., .125, .25, .5, 1., 2., 4., 8.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'v1-root', 'truth-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh verification receipt required')
    manifest = json.loads((args.run/'preregistration.json').read_text(encoding='utf-8'))
    receipt = json.loads((args.run/'execution_receipt.json').read_text(encoding='utf-8'))
    selection = json.loads((args.run/'frozen_selection.json').read_text(encoding='utf-8'))
    assert receipt['status'] != 'VOLUME_INCREMENT_BLOCKED'
    assert sha(args.run/'frozen_selection.json') == receipt['selection_sha256']
    source_rows = [r for r in manifest['cache_files'] if r['cut'] == 1056]
    assert len(source_rows) == 4
    paths = []
    for r in source_rows:
        root = (args.truth_root if r['system'] == 'truth' else args.v1_root).resolve()
        path = (root/r['file']).resolve()
        assert path.parent == root and sha(path) == r['sha256']
        paths.append((r, path))
    cache = {(r['horizon'], r['system']): np.load(p, allow_pickle=False).astype(float).ravel() for r, p in paths}
    with (args.run/'calibration_grid.csv').open(encoding='utf-8') as f:
        grid = {(r['system'], float(r['alpha']), r['scope']): r for r in csv.DictReader(f)}
    assert set(grid) == {(s, a, t) for s in SYSTEMS[1:] for a in GRID for t in ('H3', 'H12', 'macro')}
    selected = json.loads((args.run/'calibration_selected.json').read_text(encoding='utf-8'))
    largest, hashes = 0., []
    for s in SYSTEMS:
        directions = {}
        for h in (3, 12):
            path = args.run/f'private_calibration_{s}_h{h}_direction.npy'
            directions[h] = np.load(path, allow_pickle=False).astype(float).ravel()
            hashes.append({'file': path.name, 'postrun_sha256': sha(path)})
        for a in ((0.,) if s == 'native' else GRID):
            actual = []
            for index, h in enumerate((3, 12)):
                p = np.clip(cache[h, 'native'], 0., 1.); y = cache[h, 'truth']
                raw = p+a*directions[h];q = np.clip(raw, 0., 1.)
                values = {'rmse': float(np.sqrt(np.mean((q-y)**2))), 'mae': float(np.mean(abs(q-y))),
                          'unclipped_move_rmse': float(np.sqrt(np.mean((raw-y)**2))),
                          'unclipped_move_mae': float(np.mean(abs(raw-y)))}
                expected = selected[s]['cells'][index] if s == 'native' else grid[s, a, f'H{h}']
                largest = max(largest, max(abs(v-float(expected[k])) for k, v in values.items()))
                if a == selection['alphas'][s]:
                    saved = args.run/f'private_calibration_{s}_h{h}_unclipped_move.npy'
                    assert np.array_equal(np.load(saved, allow_pickle=False).ravel(), raw)
                actual.append(values)
            if s != 'native':
                macro = {k: sum(r[k] for r in actual)/2 for k in actual[0]}
                largest = max(largest, max(abs(v-float(grid[s, a, 'macro'][k])) for k, v in macro.items()))
    assert largest <= 1e-10
    result = {'status': 'PASS', 'tunable_points_recomputed': 40, 'native_points_recomputed': 1,
              'maximum_metric_difference': largest, 'cached_source_arrays_read': 4,
              'dynamic_csv_rows_read': 0, 'new_supervised_fits': 0, 'new_reference_passes': 0,
              'new_alpha_selection': False, 'evaluation_arrays_decoded': 0,
              'scope': 'Independent NumPy score reductions from archived directions; not independent feature/model reconstruction',
              'archive_hash_timing': 'Hashes recorded at this post-run verification, not claimed preregistered input identities',
              'archived_direction_hashes': hashes}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'maximum_metric_difference': largest, 'new_fits': 0}))


if __name__ == '__main__': main()
