import copy
from pathlib import Path
import pytest
from urbanev_forecast.information_registry import (
    FAMILIES, REQUIRED_FIELD_KEYS, read_header_only, validate_registry,
)


def registry():
    fields = []
    for family in sorted(FAMILIES):
        field = {key: 'SYNTHETIC_UNKNOWN' for key in REQUIRED_FIELD_KEYS}
        field.update(id=family, family=family, source_files=['occupancy.csv'])
        fields.append(field)
    return {'id': 'INFORMATION_SCOPE_REGISTRY_V1_20260913', 'known_future_inputs': [],
            'execution_budget': dict(fits=0, inference=0, alpha_searches=0, new_target_rows=0,
                                     full_dynamic_file_hashes=0), 'fields': fields}


def test_eight_families_and_unknown_eligibility_do_not_mean_failure():
    result = validate_registry(registry())
    assert result['families'] == 8
    assert result['status'] == 'REGISTRY_COMPLETE_WITH_UNRESOLVED_ELIGIBILITY'
    assert result['new_experiment_authorized'] is False


def test_state_separation_and_future_observation_rejected():
    r = registry()
    r['fields'][0]['time_eligibility'] = ''
    with pytest.raises(ValueError, match='separate'):
        validate_registry(r)
    r = registry()
    r['known_future_inputs'] = ['future_actual_weather']
    with pytest.raises(ValueError, match='Known-future'):
        validate_registry(r)


def test_field_identity_coverage_and_invalid_exclusion():
    r = registry()
    r['fields'].append(copy.deepcopy(r['fields'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        validate_registry(r)
    r = registry()
    r['fields'][0]['proven_ineffective'] = True
    with pytest.raises(ValueError, match='ineffectiveness'):
        validate_registry(r)
    r = registry()
    r['fields'] = [f for f in r['fields'] if f['family'] != 'weather']
    with pytest.raises(ValueError, match='eight'):
        validate_registry(r)


def test_header_only_ignores_invalid_binary_observation_rows(tmp_path):
    (tmp_path/'weather_central.csv').write_bytes(b'time,T,U\r\n\xff\xfeDO_NOT_DECODE_OBSERVATIONS')
    result = read_header_only(tmp_path, 'weather_central.csv')
    assert result['columns'] == ['time', 'T', 'U']
    assert result['temporal_rows_read'] == 0


def test_unregistered_paths_and_predictions_rejected_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Should reject before any file open'))
    for name in ('../weather_central.csv', 'private_truth_1392_h12.npy', 'station_information.csv'):
        with pytest.raises(ValueError, match='Unregistered'):
            read_header_only(tmp_path, name)


def test_bounded_header_and_zero_compute_budget(tmp_path):
    (tmp_path/'e_price.csv').write_bytes(b'x'*100+b'\n')
    with pytest.raises(ValueError, match='byte limit'):
        read_header_only(tmp_path, 'e_price.csv', max_bytes=20)
    r = registry()
    r['execution_budget']['fits'] = 1
    with pytest.raises(ValueError, match='numerical experiments'):
        validate_registry(r)


def test_resolved_escape_rejected_before_open(tmp_path, monkeypatch):
    original = Path.resolve

    def redirect(path, *args, **kwargs):
        if path.name == 'e_price.csv':
            return tmp_path.parent/'outside'/'e_price.csv'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', redirect)
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Escaped path must not be opened'))
    with pytest.raises(ValueError, match='outside'):
        read_header_only(tmp_path, 'e_price.csv')
