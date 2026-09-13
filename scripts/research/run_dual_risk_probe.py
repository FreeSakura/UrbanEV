"""Run the registered diagnostic once on existing, hash-locked cache arrays."""
from __future__ import annotations
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
from urbanev_forecast.dual_risk_probe import execute, load_whitelist

CONFIG = ROOT/'configs/research/DUAL_RISK_INFORMATION_PROBE_V1.json'


class RejectHeavyImports(importlib.abc.MetaPathFinder):
    attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'transformers', 'chronos', 'timesfm'}
                or fullname.startswith('urbanev_forecast.continuous')
                or fullname == 'urbanev_forecast.foundation'):
            self.attempts += 1
            raise RuntimeError('PROBE_BLOCKED: model or solver import forbidden')
        return None


def dump(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1-root', type=Path, required=True)
    parser.add_argument('--truth-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--claim', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output directory required')
    config_bytes = CONFIG.read_bytes().replace(b'\r\n', b'\n')
    cfg = json.loads(config_bytes)
    if cfg['id'] != 'DUAL_RISK_INFORMATION_PROBE_V1_20260913' or cfg['ridge'] != 0.01:
        raise ValueError('PROBE_BLOCKED: configuration identity')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frozen = subprocess.check_output(['git', 'show', 'HEAD:configs/research/DUAL_RISK_INFORMATION_PROBE_V1.json'], cwd=ROOT)
    if frozen.replace(b'\r\n', b'\n') != config_bytes:
        raise ValueError('PROBE_BLOCKED: manifest not frozen in HEAD')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', str(CONFIG),
                    'src/urbanev_forecast/dual_risk_probe.py', 'scripts/research/run_dual_risk_probe.py'], cwd=ROOT, check=True)
    for name, expected in cfg['code_sha256_lf'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('PROBE_BLOCKED: frozen code changed')
    args.claim.parent.mkdir(parents=True, exist_ok=True)
    with args.claim.open('x', encoding='utf-8') as claim:
        json.dump({'id': cfg['id'], 'execution_commit': head, 'status': 'CLAIMED_ONCE'}, claim)
    args.output.mkdir()
    started = time.perf_counter()
    guard = RejectHeavyImports()
    sys.meta_path.insert(0, guard)
    calls = {'ridge_linear_solves': 0}
    original_solve = np.linalg.solve

    def counted_solve(*a, **kw):
        calls['ridge_linear_solves'] += 1
        if calls['ridge_linear_solves'] > 8:
            raise RuntimeError('PROBE_BLOCKED: excess fits')
        return original_solve(*a, **kw)

    try:
        arrays, ledger = load_whitelist(cfg, {'v1': args.v1_root, 'truth': args.truth_root})
        with threadpool_limits(limits=2), patch.object(np.linalg, 'solve', counted_solve):
            result = execute(arrays)
        assert calls['ridge_linear_solves'] == 8 and guard.attempts == 0
        dump(args.output/'probe_manifest.json', cfg)
        dump(args.output/'directional_risk_report.json', {k: v for k, v in result.items() if k != 'moment_scores'})
        with (args.output/'moment_prediction_scores.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(result['moment_scores'][0]))
            writer.writeheader()
            writer.writerows(result['moment_scores'])
        receipt = {'id': cfg['id'], 'status': result['gate']['status'], 'execution_commit': head,
                   'config_sha256_lf': hashlib.sha256(config_bytes).hexdigest(),
                   'read_ledger': ledger, 'numerical_array_count': len(arrays),
                   'fit_monitor': calls, 'forbidden_import_attempts': guard.attempts,
                   'new_foundation_inference': 0, 'alpha_searches': 0,
                   'original_data_files_read': 0, 'maximum_target_index': 1559,
                   'duration_latest_source_rule': 'origin-2, inherited from frozen V1 prediction lineage',
                   'post_hoc_development_diagnostic': True, 'blind_test': False,
                   'new_point_forecast_scores_generated': False, 'old_no_go_unchanged': True,
                   'elapsed_seconds': time.perf_counter()-started, 'threads': 2,
                   'timing_scope': 'load/verify arrays, fits, derivative summaries and result writing; excludes preflight and receipt write',
                   'peak_process_memory_bytes': None, 'peak_memory_status': 'NOT_MEASURED',
                   'quota_reset_used': False}
        dump(args.output/'probe_receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'fits': 8, 'arrays': len(arrays), 'elapsed_seconds': receipt['elapsed_seconds']}))
    except Exception as exc:
        dump(args.output/'probe_receipt.json', {'status': 'PROBE_BLOCKED', 'error_type': type(exc).__name__,
             'error': str(exc), 'execution_commit': head, 'fit_monitor': calls,
             'forbidden_import_attempts': guard.attempts, 'automatic_retry': False})
        raise
    finally:
        sys.meta_path.remove(guard)


if __name__ == '__main__':
    main()
