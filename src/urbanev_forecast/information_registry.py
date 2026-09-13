"""Metadata-only registry contracts; deliberately independent of data/model loaders."""
import csv
import hashlib
from pathlib import Path

FAMILIES = {'occupancy_calendar', 'electricity_price', 'service_price', 'weather',
            'capacity_static', 'spatial', 'volume', 'duration'}
HEADER_FILES = {'occupancy.csv', 'duration.csv', 'volume.csv', 'volume-11kW.csv',
                'e_price.csv', 's_price.csv', 'weather_central.csv', 'weather_airport.csv',
                'inf.csv', 'poi.csv', 'adj.csv', 'distance.csv'}
REQUIRED_FIELD_KEYS = {'id', 'family', 'source_files', 'column_or_alias', 'unit',
                       'data_version', 'spatial_level', 'measurement_kind', 'generation',
                       'parents', 'interval_definition', 'observed_at', 'effective_at',
                       'published_at', 'revised_at', 'normalization', 'imputation',
                       'experimental_status', 'time_eligibility', 'evidence', 'uncertainties'}


def read_header_only(root, filename, max_bytes=65536):
    if filename not in HEADER_FILES or Path(filename).name != filename:
        raise ValueError('Unregistered source; only CSV headers are allowed')
    root = Path(root).resolve()
    path = (root/filename).resolve()
    if path.parent != root:
        raise ValueError('Source resolves outside registered root')
    # Unbuffered, bounded first line: never decode an observation row.
    with path.open('rb', buffering=0) as handle:
        header = handle.readline(max_bytes+1)
    if len(header) > max_bytes:
        raise ValueError('Header exceeds registered byte limit')
    columns = next(csv.reader([header.decode('utf-8-sig').rstrip('\r\n')]))
    if not columns or not any(columns):
        raise ValueError('Missing header')
    return {'file': filename, 'columns': columns,
            'header_sha256': hashlib.sha256(header).hexdigest(), 'temporal_rows_read': 0}


def validate_registry(registry):
    if registry['id'] != 'INFORMATION_SCOPE_REGISTRY_V1_20260913':
        raise ValueError('Unexpected registry identity')
    budget = registry['execution_budget']
    if any(budget[k] != 0 for k in ('fits', 'inference', 'alpha_searches', 'new_target_rows',
                                    'full_dynamic_file_hashes')):
        raise ValueError('Metadata stage cannot authorize numerical experiments')
    if registry['known_future_inputs'] != []:
        raise ValueError('Known-future input contract must remain unchanged')
    fields = registry['fields']
    if {f['family'] for f in fields} != FAMILIES:
        raise ValueError('All eight information families must be covered')
    if len({f['id'] for f in fields}) != len(fields):
        raise ValueError('Duplicate canonical field identity')
    for f in fields:
        if not REQUIRED_FIELD_KEYS <= f.keys():
            raise ValueError('Missing field metadata')
        if not f['experimental_status'] or not f['time_eligibility']:
            raise ValueError('Experiment coverage and time eligibility are separate required fields')
        if f.get('proven_ineffective', False):
            raise ValueError('Current evidence does not establish field-level ineffectiveness')
        if not set(f['source_files']) <= HEADER_FILES:
            raise ValueError('Unknown source file alias')
    return {'status': 'REGISTRY_COMPLETE_WITH_UNRESOLVED_ELIGIBILITY',
            'families': len(FAMILIES), 'fields': len(fields), 'new_experiment_authorized': False}
