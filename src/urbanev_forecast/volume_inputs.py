"""Stage-gated covariate prefixes and native/truth cache for the volume pilot."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .bounded_predictor_csv import prefix_bytes, parse_prefix
from .volume_composition import SYSTEMS, GRID

LIMITS = {'FIT': {'occupancy.csv': 876, 'duration.csv': 875, 'volume.csv': 875},
          'CALIBRATE': {'occupancy.csv': 1212, 'duration.csv': 1211, 'volume.csv': 1211},
          'EVALUATE_INPUTS': {'occupancy.csv': 1548, 'duration.csv': 1547, 'volume.csv': 1547}}


class VolumeInputs:
    def __init__(self, config, data_root, cache_roots, config_hash):
        self.config, self.config_hash = config, config_hash
        self.root = Path(data_root).resolve()
        self.stage, self.ledger, self.decoded, self.values = 'PRECHECK', [], set(), {}
        records = config['cache_files']
        expected = {(c, h, s) for c in (720, 1056, 1392) for h in (3, 12) for s in ('native', 'truth')}
        if len(records) != 12 or {(r['cut'], r['horizon'], r['system']) for r in records} != expected:
            raise ValueError('Invalid native/truth whitelist')
        self.blobs, self.records = {}, {}
        for row in records:
            cut, h, s = row['cut'], row['horizon'], row['system']
            role = 'truth' if s == 'truth' else 'v1'
            name = f'private_truth_{cut}_h{h}.npy' if s == 'truth' else f'private_{cut}_h{h}_native_a1.npy'
            base = Path(cache_roots[role]).resolve()
            path = (base/name).resolve()
            if row['file'] != name or row['root_role'] != role or row['shape'] != [14, h, 275] or path.parent != base:
                raise ValueError('Invalid cache identity')
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row['sha256']:
                raise ValueError('Cache hash mismatch')
            self.blobs[cut, h, s], self.records[cut, h, s] = payload, row
            self.ledger.append({'kind': 'opaque_cache_hash', 'file': name, 'sha256': digest})
        info_path = (self.root/'inf.csv').resolve()
        if info_path.parent != self.root:
            raise ValueError('Static source outside registered root')
        info_bytes = info_path.read_bytes()
        if hashlib.sha256(info_bytes).hexdigest() != config['static_info_sha256']:
            raise ValueError('Static info identity mismatch')
        self.info_bytes = info_bytes
        self.ledger.append({'kind': 'opaque_static_hash', 'file': 'inf.csv', 'sha256': config['static_info_sha256']})

    def _cache(self, cut, systems):
        allowed = {'FIT': (720, {'native', 'truth'}), 'CALIBRATE': (1056, {'native', 'truth'}),
                   'EVALUATE_INPUTS': (1392, {'native'}), 'EVALUATE_TRUTH': (1392, {'truth'})}
        if self.stage not in allowed or (cut, set(systems)) != allowed[self.stage]:
            raise ValueError('Cache decode outside stage')
        out = {}
        for h in (3, 12):
            for s in systems:
                key = cut, h, s
                if key in self.decoded:
                    raise ValueError('Cache already decoded')
                a = np.load(io.BytesIO(self.blobs[key]), allow_pickle=False)
                if a.shape != (14, h, 275) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
                    raise ValueError('Invalid cache values')
                if s == 'truth' and ((a < 0) | (a > 1)).any():
                    raise ValueError('Invalid cached target')
                out[key] = a.astype(np.float64)
                self.values[key] = out[key]
                self.decoded.add(key)
                self.ledger.append({'kind': 'numeric_cache_decode', 'stage': self.stage, 'cut': cut,
                                    'horizon': h, 'system': s, 'file': self.records[key]['file'],
                                    'max_target_index': cut+156+h-1})
        return out

    def _csvs(self):
        if self.stage not in LIMITS:
            raise ValueError('CSV read outside current stage')
        payloads = {}
        for name, rows in LIMITS[self.stage].items():
            blob = prefix_bytes(self.root, name, rows, rows, set(LIMITS[self.stage]))
            expected = self.config['prefixes'][self.stage][name]
            if expected['rows'] != rows or hashlib.sha256(blob).hexdigest() != expected['sha256']:
                raise ValueError('Stage prefix hash mismatch')
            payloads[name] = blob
            self.ledger.append({'kind': 'opaque_prefix_hash', 'stage': self.stage, 'file': name,
                                'row_range': [0, rows], 'sha256': expected['sha256']})
        header = payloads['occupancy.csv'].splitlines(keepends=True)[0]
        columns = next(csv.reader([header.decode('utf-8-sig').rstrip('\r\n')]))[1:]
        if len(columns) != 275 or hashlib.sha256('\n'.join(columns).encode()).hexdigest() != self.config['column_order_sha256']:
            raise ValueError('Region order does not match native cache')
        if not hasattr(self, 'capacities'):
            info = pd.read_csv(io.BytesIO(self.info_bytes), usecols=['station_id', 'TAZID', 'charge_count'], dtype={'TAZID': str})
            capacities = info.groupby('TAZID', sort=False)['charge_count'].sum().reindex(columns).to_numpy(np.float64)
            if not np.isfinite(capacities).all() or (capacities <= 0).any():
                raise ValueError('Invalid zone capacity mapping')
            self.capacities = capacities
            self.ledger.append({'kind': 'static_numeric_columns', 'file': 'inf.csv',
                                'columns': ['station_id', 'TAZID', 'charge_count'], 'zone_count': 275,
                                'ordered_capacity_float32_sha256': hashlib.sha256(capacities.astype(np.float32).tobytes()).hexdigest()})
        raw = {}
        for name, payload in payloads.items():
            if payload.splitlines(keepends=True)[0] != header:
                raise ValueError('Time/region header mismatch')
            entry = self.config['prefixes'][self.stage][name]
            values, receipt = parse_prefix(payload, entry['rows'], columns, entry['sha256'], nonnegative=True)
            key = name.removesuffix('.csv')
            if key == 'occupancy':
                values = (values.astype(np.float32)/self.capacities.astype(np.float32)[None, :]).astype(np.float64)
                if ((values < 0) | (values > 1)).any():
                    raise ValueError('Occupancy outside[0,1] after original float32 normalization')
            raw[key] = values
            self.ledger.append({'kind': 'numeric_csv_prefix', 'stage': self.stage, 'file': name, **receipt})
        d = raw['duration']/self.capacities[None, :]
        if ((d < 0) | (d > 1)).any():
            raise ValueError('Duration/capacity outside[0,1]')
        self.raw = raw
        return raw

    def lineage_check(self):
        rows, largest = 0, 0.
        for (cut, h, system), y in self.values.items():
            if system != 'truth':
                continue
            indices = np.arange(cut, cut+168, 12)[:, None]+np.arange(h)[None, :]
            mask = indices < len(self.raw['occupancy'])
            actual = self.raw['occupancy'][indices[mask]]
            if actual.size:
                largest = max(largest, float(np.max(abs(actual-y[mask]))))
                rows += actual.size
        if largest > 1e-7:
            raise ValueError('Occupancy/cached-truth lineage mismatch')
        self.ledger.append({'kind': 'available_target_lineage_check', 'stage': self.stage,
                            'entries_checked': rows, 'maximum_difference': largest,
                            'no_extra_rows_read': True})

    def fit(self):
        if self.stage != 'PRECHECK':
            raise ValueError('Wrong FIT stage')
        self.stage = 'FIT'
        raw = self._csvs()
        cache = self._cache(720, ('native', 'truth'))
        self.lineage_check()
        return raw, cache

    def freeze_models(self, models, power):
        if self.stage != 'FIT' or set(models) != set(SYSTEMS[1:]):
            raise ValueError('Five saved models required')
        self.models, self.power = models, power
        self.stage = 'FIT_FROZEN'

    def calibration(self):
        if self.stage != 'FIT_FROZEN':
            raise ValueError('Wrong CALIBRATE stage')
        self.stage = 'CALIBRATE'
        raw = self._csvs()
        cache = self._cache(1056, ('native', 'truth'))
        self.lineage_check()
        return raw, cache

    def evaluation_inputs(self, blob, expected_hash):
        if self.stage != 'CALIBRATE' or hashlib.sha256(blob).hexdigest() != expected_hash:
            raise ValueError('Wrong selection identity/stage')
        selected = json.loads(blob)
        if (selected['calibration_information_gate'] is not True or selected['config_sha256'] != self.config_hash
                or selected['models'] != self.models or selected['power_reference'] != self.power
                or set(selected['alphas']) != set(SYSTEMS) or any(a not in GRID for a in selected['alphas'].values())):
            raise ValueError('Evaluation not admitted')
        self.stage = 'EVALUATE_INPUTS'
        raw = self._csvs()
        cache = self._cache(1392, ('native',))
        self.lineage_check()
        return raw, cache

    def evaluation_truth(self, saved):
        if (self.stage != 'EVALUATE_INPUTS' or len(saved) != 12
                or {(r['system'], r['horizon']) for r in saved} != {(s, h) for s in SYSTEMS for h in (3, 12)}
                or not all(len(r.get('sha256', '')) == 64 for r in saved)):
            raise ValueError('Twelve fixed predictions required before truth decode')
        self.stage = 'EVALUATE_TRUTH'
        cache = self._cache(1392, ('truth',))
        self.lineage_check()
        return cache
