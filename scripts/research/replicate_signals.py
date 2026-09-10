"""Frozen fold-two replication; no test read or hyperparameter search."""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from urbanev_forecast.data import load_rate_prefix, scores
from urbanev_forecast.foundation import file_hash
from dual_region_pilot import GroupedChronos, affinity, partition, validate_groups

CONFIG = ROOT / 'configs/research/SIGNAL_REPLICATION_V1.json'
TRAIN, END, HS = 1171, 1318, (3, 12)


def derangement(n, seed=44):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def load_duration(path, info, csv):
    with path.open('rb') as stream:
        payload = b''.join(stream.readline() for _ in range(END + 1))
    frame = pd.read_csv(io.BytesIO(payload), index_col=0, parse_dates=True)
    cols = pd.read_csv(csv, nrows=0).columns[1:].astype(str).tolist()
    static = pd.read_csv(info, dtype={'TAZID': str})
    cap = static.groupby('TAZID')['charge_count'].sum().reindex(cols).to_numpy(float)
    if frame.shape != (END, 275) or list(frame.columns.astype(str)) != cols:
        raise ValueError('Duration columns or shape differ')
    if not frame.index.equals(pd.date_range('2022-09-01', periods=END, freq='h')):
        raise ValueError('Duration timestamps differ')
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise ValueError('Invalid capacity')
    d = frame.to_numpy(np.float32) / cap.astype(np.float32)[None, :]
    if not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
        raise ValueError('Duration numeric qualification failed')
    return d, {'prefix_sha256': hashlib.sha256(payload).hexdigest(),
               'info_sha256': file_hash(info), 'rows': END,
               'capacity_sum': float(cap.sum()), 'numeric_range': [float(d.min()), float(d.max())],
               'units_documented_hours_per_interval': True,
               'endpoint_and_live_availability_established': False}


def contexts(o, d, s, variant, perm):
    history = o[s-168:s]
    if variant == 'duplicate_occupancy':
        return np.concatenate([history, history], axis=1)
    if variant in ('lagged_duration', 'permuted_duration'):
        aux = d[s-169:s-1]
        if variant == 'permuted_duration':
            aux = aux[:, perm]
        return np.concatenate([history, aux], axis=1)
    return history


def predict_grouped(backend, context, groups, h, augmented=False):
    tasks = [g + [c+275 for c in g] for g in groups] if augmented else groups
    raw = backend.predict(context, tasks, h)
    out = np.empty((h, 275), dtype=np.float32)
    offset = 0
    for g, task in zip(groups, tasks):
        out[:, g] = raw[:, offset:offset+len(g)]
        offset += len(task)
    if offset != raw.shape[1] or not np.isfinite(out).all():
        raise ValueError('Incomplete output or mapping mismatch')
    return out


def ridge_design(o, d, origins, variant, perm):
    histories = np.stack([o[s-168:s].T for s in origins]).reshape(-1, 168).astype(float)
    anchor = histories[:, -1:].copy()
    x = histories - anchor
    if variant == 'duplicate_occupancy':
        x = np.concatenate([x, x], axis=1)
    elif variant in ('lagged_duration', 'permuted_duration'):
        aux = np.stack([d[s-169:s-1].T for s in origins])
        if variant == 'permuted_duration':
            aux = aux[:, perm, :]
        x = np.concatenate([x, aux.reshape(-1, 168)], axis=1)
    return x, anchor


def ridge_predict(o, d, h, origins, variant, perm):
    train = np.arange(169, TRAIN-h+1, 3)
    x, anchor = ridge_design(o, d, train, variant, perm)
    vx, va = ridge_design(o, d, origins, variant, perm)
    y = np.stack([o[s:s+h].T for s in train]).reshape(-1, h)-anchor
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1
    x, vx = (x-mean)/scale, (vx-mean)/scale
    x = np.column_stack([x, np.ones(len(x))])
    vx = np.column_stack([vx, np.ones(len(vx))])
    regularizer = np.eye(x.shape[1]) * .01
    regularizer[-1, -1] = 0
    beta = np.linalg.solve(x.T@x/len(x)+regularizer, x.T@y/len(x))
    pred = (vx@beta+va).reshape(len(origins), 275, h).transpose(0, 2, 1)
    return pred, {'train_origins': len(train), 'train_last_target': int(train[-1]+h-1),
                  'features_with_intercept': x.shape[1], 'ridge': .01,
                  'normal_equation_relative_residual': float(np.linalg.norm((x.T@x/len(x)+regularizer)@beta-x.T@y/len(x))/max(np.linalg.norm(x.T@y/len(x)), 1e-12))}


def comparison(predictions, truths, candidate, reference):
    cells = []
    losses = []
    for h in HS:
        y = truths[h]
        cp, rp = np.clip(predictions[candidate][h], 0, 1), np.clip(predictions[reference][h], 0, 1)
        cs, rs = scores(cp, y), scores(rp, y)
        cells.append({'horizon': h, 'candidate': cs, 'reference': rs,
                      'rmse_degradation_percent': 100*(cs['rmse']/rs['rmse']-1)})
        losses.append(((cp.astype(float)-y)**2, (rp.astype(float)-y)**2,
                       np.abs(cp.astype(float)-y), np.abs(rp.astype(float)-y)))
    cm = {m: float(np.mean([c['candidate'][m] for c in cells])) for m in ('rmse', 'mae')}
    rm = {m: float(np.mean([c['reference'][m] for c in cells])) for m in ('rmse', 'mae')}
    gain = 100*(1-cm['rmse']/rm['rmse'])
    # Resample origins jointly across zones and horizon steps. The same circular
    # block starts couple H3/H12; wrap each valid origin count separately.
    rng = np.random.default_rng(20260910)
    stats = []
    origin_losses = [[a.mean(axis=(1, 2)) for a in row] for row in losses]
    for _ in range(2000):
        uniform_starts = rng.random(size=7)
        cmse, rmse, cmae, rmae = [], [], [], []
        for a, b, c, r in origin_losses:
            starts = np.floor(uniform_starts*len(a)).astype(int)
            idx = ((starts[:, None]+np.arange(24)) % len(a)).ravel()[:len(a)]
            cmse.append(np.sqrt(a[idx].mean())); rmse.append(np.sqrt(b[idx].mean()))
            cmae.append(c[idx].mean()); rmae.append(r[idx].mean())
        stats.append((100*(1-np.mean(cmse)/np.mean(rmse)), np.mean(cmae)-np.mean(rmae)))
    passed = gain >= 1 and cm['mae'] <= rm['mae'] and max(c['rmse_degradation_percent'] for c in cells) <= 1
    return {'candidate': candidate, 'reference': reference, 'cells': cells,
            'macro_rmse_gain_percent': gain, 'macro_mae_change': cm['mae']-rm['mae'],
            'rmse_gain_95pct_block_interval': np.quantile(np.asarray(stats)[:, 0], [.025, .975]).tolist(),
            'mae_change_95pct_block_interval': np.quantile(np.asarray(stats)[:, 1], [.025, .975]).tolist(),
            'point_gate': 'GO_DEVELOPMENT_ONLY' if passed else 'NO_GO',
            'interval_scope': 'descriptive within-window paired circular blocks; selected best-control identity held fixed'}


def main():
    p = argparse.ArgumentParser()
    for arg in ('csv', 'duration', 'info', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--route', choices=['ridge', 'foundation'], required=True)
    p.add_argument('--model-dir', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new output directory')
    config = json.loads(CONFIG.read_text())
    if (config['train_end'], config['validation_end'], tuple(config['horizons'])) != (TRAIN, END, HS):
        raise ValueError('Frozen config/code mismatch')
    o, source = load_rate_prefix(a.csv, END)
    d, qualification = load_duration(a.duration, a.info, a.csv)
    a.output.mkdir(parents=True)
    perm = derangement(275)
    groups = partition(affinity(o[TRAIN-288:TRAIN]))
    validate_groups(groups)
    (a.output/'private_structure.json').write_text(json.dumps({'groups': groups, 'duration_permutation': perm.tolist()}))
    if a.route == 'foundation':
        backend = GroupedChronos(a.model_dir)
        systems = ['full275', 'random32_s42', 'random32_s43', 'random32_s44', 'raw_correlation32',
                   'duplicate_occupancy', 'lagged_duration', 'permuted_duration']
    else:
        backend = None
        systems = ['occupancy', 'duplicate_occupancy', 'lagged_duration', 'permuted_duration']
    predictions = {s: {} for s in systems}; truths = {}; records = []
    begun = time.perf_counter()
    with threadpool_limits(limits=2):
        for h in HS:
            origins = np.arange(TRAIN, END-h+1)
            truths[h] = o[origins[:, None]+np.arange(h)]
            np.save(a.output/f'private_origins_h{h}.npy', origins)
            for system in systems:
                start = time.perf_counter(); extra = {}
                if backend is None:
                    pred, extra = ridge_predict(o, d, h, origins, system, perm)
                else:
                    chosen = groups
                    if system == 'full275':
                        chosen = [list(range(275))]
                    elif system.startswith('random32_s'):
                        order = np.random.default_rng(int(system.split('_s')[-1])).permutation(275).tolist()
                        chosen = [sorted(order[i:i+32]) for i in range(0, 275, 32)]
                    augmented = system in ('duplicate_occupancy', 'lagged_duration', 'permuted_duration')
                    pred = np.stack([predict_grouped(backend, contexts(o, d, s, system, perm), chosen, h, augmented) for s in origins])
                    extra = {'origins': len(origins), 'channels_per_origin': 550 if augmented else 275,
                             'group_sizes': [len(g)*(2 if augmented else 1) for g in chosen]}
                predictions[system][h] = pred
                dest = a.output/f'private_prediction_{system}_h{h}.npy'
                np.save(dest, pred)
                row = {'system': system, 'horizon': h, 'raw': scores(pred, truths[h]),
                       'clipped': scores(np.clip(pred, 0, 1), truths[h]), 'seconds': time.perf_counter()-start,
                       'prediction_sha256': file_hash(dest), **extra}
                records.append(row)
                print(json.dumps(row), flush=True)
    macro = {s: {m: float(np.mean([r['clipped'][m] for r in records if r['system'] == s])) for m in ('rmse', 'mae')} for s in systems}
    comparisons = []
    if backend is not None:
        best = min(systems[:4], key=lambda s: macro[s]['rmse'])
        comparisons.append(comparison(predictions, truths, 'raw_correlation32', best))
    base = 'raw_correlation32' if backend is not None else 'occupancy'
    best = min([base, 'duplicate_occupancy'], key=lambda s: macro[s]['rmse'])
    comparisons.append(comparison(predictions, truths, 'lagged_duration', best))
    comparisons.append(comparison(predictions, truths, 'lagged_duration', 'permuted_duration'))
    report = {'protocol': config['id'], 'route': a.route, 'source': source, 'qualification': qualification,
              'records': records, 'macro': macro, 'comparisons': comparisons,
              'backend': backend.base.metadata if backend else 'fixed ridge, no stochastic training seeds',
              'config_sha256': file_hash(CONFIG), 'script_sha256': file_hash(Path(__file__)),
              'group_helper_sha256': file_hash(Path(__file__).with_name('dual_region_pilot.py')),
              'structure_sha256': file_hash(a.output/'private_structure.json'),
              'elapsed_seconds': time.perf_counter()-begun, 'controlled_latency_benchmark': False,
              'fold_two_test_loaded': False, 'previous_fold_test_enters_training': True,
              'independent_confirmation': False, 'sota_claim': False}
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'macro': macro, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
