import hashlib
from pathlib import Path
import numpy as np
import pytest
from urbanev_forecast.bounded_predictor_csv import prefix_bytes, parse_prefix


def test_stops_before_later_invalid_rows(tmp_path):
    payload = b'time,z1,z2\n2022-09-01 00:00:00,1,2\n2022-09-01 01:00:00,3,4\n'
    (tmp_path/'volume.csv').write_bytes(payload+b'\xff\xfeMUST_NOT_PARSE_LATER_ROWS')
    read = prefix_bytes(tmp_path, 'volume.csv', 2, 2, {'volume.csv'})
    assert read == payload
    values, receipt = parse_prefix(read, 2, ['z1', 'z2'], hashlib.sha256(read).hexdigest(), True)
    np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
    assert receipt['numeric_row_range'] == [0, 2]


def test_wrong_stage_or_path_rejected_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Forbidden source must not open'))
    for name, rows, limit in [('volume.csv', 3, 2), ('../volume.csv', 2, 2),
                              ('occupancy.csv', 2, 2), ('volume.csv', 1561, 1561)]:
        with pytest.raises(ValueError, match='registered stage'):
            prefix_bytes(tmp_path, name, rows, limit, {'volume.csv'})


def test_hash_mismatch_precedes_parsing():
    with pytest.raises(ValueError, match='identity'):
        parse_prefix(b'not a CSV', 1, ['z1'], '0'*64)


@pytest.mark.parametrize('line', [b'2022-09-01 01:00:00,1\n', b'2022-09-01 00:00:00,nan\n',
                                 b'2022-09-01 00:00:00,-1\n'])
def test_time_and_value_failures_are_not_repaired(line):
    payload = b'time,z1\n'+line
    with pytest.raises(ValueError):
        parse_prefix(payload, 1, ['z1'], hashlib.sha256(payload).hexdigest(), True)


def test_column_order_must_match():
    payload = b'time,z2,z1\n2022-09-01 00:00:00,1,2\n'
    with pytest.raises(ValueError, match='columns'):
        parse_prefix(payload, 1, ['z1', 'z2'], hashlib.sha256(payload).hexdigest())
