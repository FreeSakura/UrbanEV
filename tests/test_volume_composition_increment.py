import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest
import urbanev_forecast.volume_composition as c
from urbanev_forecast.volume_inputs import VolumeInputs, LIMITS


def test_single_fixed_power_and_cancellation_threshold():
    rng = np.random.default_rng(4)
    d = rng.uniform(.01, .8, (875, 2))
    v = d*np.array([7., 60.])
    power, _ = c.historical_power(d, v)
    derived, _ = c.derive_series(d, v, np.ones(2), power)
    assert np.max(abs(derived['z'])) < 1e-10
    x = np.column_stack([np.ones(875), derived['z']])
    model = c.fit_scalar([(x, np.ones(875))])
    assert model['inactive'].all()


def test_two_power_compositions_and_fixed_reference_window():
    d = np.ones((875, 2))
    v = np.tile([7., 60.], (875, 1))
    b, _ = c.historical_power(d, v)
    changed_d, changed_v = d.copy(), v.copy()
    changed_d[719:] = .1; changed_v[719:] = 1000.
    np.testing.assert_array_equal(c.historical_power(changed_d, changed_v)[0], b)
    mix = v.copy(); mix[720] = [60., 7.]
    derived, _ = c.derive_series(d, mix, np.ones(2), b)
    np.testing.assert_allclose(derived['z'][720], [53., -53.])


def test_zero_reference_fallback_and_inconsistent_pairs_retained():
    d = np.column_stack([np.ones(875), np.zeros(875)])
    v = np.column_stack([np.full(875, 7.), np.ones(875)])
    b, summary = c.historical_power(d, v)
    assert summary['zero_denominator_fallback_regions'] == 1
    assert b[1] == 8.
    _, diagnostic = c.derive_series(d, v, np.ones(2), b)
    assert diagnostic['D_zero_V_positive'] == 875
    with pytest.raises(ValueError, match='DURATION_REFERENCE_EMPTY'):
        c.historical_power(np.zeros_like(d), v)


def synthetic_raw():
    o = np.tile(np.arange(876)[:, None]/10000, (1, 2))
    d = np.tile(np.arange(875)[:, None]/10000, (1, 2))
    return {'occupancy': o, 'duration': d, 'volume': d*np.array([7., 60.])}


def test_historical_indices_widths_and_own_capacity_before_permutation():
    raw = synthetic_raw(); capacities = np.array([1., 2.]); power = np.array([5., 50.])
    a, _ = c.derive_series(raw['duration'], raw['volume'], capacities, power)
    native = np.full((14, 3, 2), .2)
    for system, width in c.WIDTHS.items():
        x, p = c.design(raw['occupancy'], a, capacities, power, native, 720, 3, system)
        assert x.shape == (84, width)
        assert x[0, 1] == raw['occupancy'][719, 0]
        assert x[0, 25] == a['d'][718, 0]
        assert x[0, 33] == a['d'][551, 0]
        if system == 'OD_MISALIGNED':
            assert x[0, 48] == a['z'][718, 1]
    before, _ = c.design(raw['occupancy'], a, capacities, power, native, 720, 3, 'OD_VOLUME')
    raw['occupancy'][720:] = .9
    for value in a.values(): value[719:] = 999.
    after, _ = c.design(raw['occupancy'], a, capacities, power, native, 720, 3, 'OD_VOLUME')
    np.testing.assert_array_equal(before[:6], after[:6])


def test_training_only_five_fits_and_future_label_isolation(monkeypatch):
    raw = synthetic_raw(); capacities = np.ones(2)
    power, _ = c.historical_power(raw['duration'], raw['volume'])
    derived, _ = c.derive_series(raw['duration'], raw['volume'], capacities, power)
    cache = {(cut, h, s): np.full((14, h, 2), .2 if s == 'native' else .3)
             for cut in (720, 1056, 1392) for h in (3, 12) for s in ('native', 'truth')}
    original = c.fit_scalar; calls = []
    def fit(cells): calls.append(len(cells)); return original(cells)
    monkeypatch.setattr(c, 'fit_scalar', fit)
    first = c.fit_models(raw, derived, capacities, power, cache)
    for cut in (1056, 1392):
        for h in (3, 12): cache[cut, h, 'truth'][:] = 1.
    second = c.fit_models(raw, derived, capacities, power, cache)
    assert calls == [2]*10
    for system in c.SYSTEMS[1:]:
        np.testing.assert_array_equal(first[system]['coefficient'], second[system]['coefficient'])


def test_cell_weights_and_near_constant_rule_apply_to_all_columns():
    x1 = np.column_stack([np.ones(2), [1., 1.+1e-12]])
    x2 = np.repeat(x1, 4, axis=0)
    model = c.fit_scalar([(x1, np.zeros(2)), (x2, np.ones(8))])
    assert model['inactive'][0]
    assert model['coefficient'][0] == pytest.approx(.5)


def point(alpha=0., rmse=.1, mae=.05):
    cells = [dict(horizon=h, rmse=rmse, mae=mae, unclipped_move_rmse=rmse, unclipped_move_mae=mae,
                  baseline_rmse=.1, baseline_mae=.05) for h in (3, 12)]
    return {'alpha': alpha, 'cells': cells, 'macro': c.macro_score(cells)}


def test_complete_controls_and_strict_mae_and_ties():
    selected = {s: point() for s in c.SYSTEMS}
    selected['OD_VOLUME'] = point(.25, .098, .049)
    assert c.gate(selected)['passed']
    assert c.gate(selected)['Cstar'] == 'native'
    selected['OD_VOLUME']['macro']['mae'] = np.nextafter(.05, 1.)
    assert not c.gate(selected)['passed']
    del selected['OD_DUPLICATE']
    with pytest.raises(ValueError, match='six'): c.gate(selected)
    grid = [point(a, .1 if a == 0 else .09) for a in c.GRID]
    assert c.choose_step(grid, point())['alpha'] == .125


def test_evaluation_labels_do_not_enter_step_selection():
    cache = {(1056, h, 'truth'): np.full(h, .25) for h in (3, 12)}
    p = {h: np.full(h, .2) for h in (3, 12)}
    u = {s: {h: np.full(h, .01) if s != 'native' else np.zeros(h) for h in (3, 12)} for s in c.SYSTEMS}
    cache[1392, 3, 'truth'] = np.array([0.])
    first, _ = c.calibrate(cache, p, u)
    cache[1392, 3, 'truth'][:] = 1.
    second, _ = c.calibrate(cache, p, u)
    assert [first[s]['alpha'] for s in c.SYSTEMS] == [second[s]['alpha'] for s in c.SYSTEMS]


@pytest.fixture
def input_fixture(tmp_path):
    names = [f'z{i}' for i in range(275)]
    config = {'column_order_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(), 'prefixes': {}, 'cache_files': []}
    header = 'time,'+','.join(names)+'\n'
    import pandas as pd
    for file, value, count in [('occupancy.csv', .2, 1212), ('duration.csv', .1, 1211), ('volume.csv', .7, 1211)]:
        lines = [header]+[str(t)+','+','.join([str(value)]*275)+'\n' for t in pd.date_range('2022-09-01', periods=count, freq='h')]
        (tmp_path/file).write_bytes(''.join(lines).encode())
        for stage, limits in LIMITS.items():
            rows = limits[file]
            config['prefixes'].setdefault(stage, {})[file] = {'rows': rows, 'sha256': hashlib.sha256(''.join(lines[:rows+1]).encode()).hexdigest() if rows <= count else '0'*64}
    info = 'station_id,TAZID,charge_count\n'+''.join(f's{i},z{i},1\n' for i in range(275))
    (tmp_path/'inf.csv').write_bytes(info.encode());config['static_info_sha256'] = hashlib.sha256(info.encode()).hexdigest()
    for cut in (720, 1056, 1392):
        for h in (3, 12):
            for system in ('native', 'truth'):
                file = f'private_truth_{cut}_h{h}.npy' if system == 'truth' else f'private_{cut}_h{h}_native_a1.npy'
                data = np.full((14, h, 275), .2)
                if cut == 1392 and system == 'truth': data[:] = np.nan
                b = io.BytesIO();np.save(b, data);blob = b.getvalue();(tmp_path/file).write_bytes(blob)
                config['cache_files'].append(dict(cut=cut,horizon=h,system=system,file=file,root_role='truth' if system=='truth' else 'v1',shape=[14,h,275],sha256=hashlib.sha256(blob).hexdigest()))
    return config, tmp_path, {'truth': tmp_path, 'v1': tmp_path}


def test_stage_no_go_does_not_expand_csv_or_decode1392(input_fixture, tmp_path):
    config, root, cache_roots = input_fixture
    inputs = VolumeInputs(config, root, cache_roots, 'cfg')
    spec = importlib.util.spec_from_file_location('volume_runner_test', Path(__file__).resolve().parents[1]/'scripts/research/run_volume_composition_increment.py')
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    output = tmp_path/'result';output.mkdir()
    result, _ = runner.run(inputs, output, 'cfg', lambda: None, {'fitting_closed': False})
    assert result['status'] == 'VOLUME_INCREMENT_CALIBRATION_NO_GO'
    assert len(inputs.decoded) == 8 and not any(k[0] == 1392 for k in inputs.decoded)
    assert not any(r.get('stage') == 'EVALUATE_INPUTS' for r in inputs.ledger)


def test_hash_and_wrong_stage_rejected_before_numeric_decode(input_fixture, monkeypatch):
    config, root, roots = input_fixture
    bad = copy.deepcopy(config);bad['cache_files'][-1]['sha256'] = '0'*64
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Hash checks must precede decode'))
    with pytest.raises(ValueError, match='hash mismatch'): VolumeInputs(bad, root, roots, 'cfg')
    inputs = VolumeInputs(config, root, roots, 'cfg')
    with pytest.raises(ValueError, match='stage'): inputs._cache(1392, ('truth',))
    with pytest.raises(ValueError, match='CALIBRATE'): inputs.calibration()


def test_calibration_prefix_identity_failure_no_new_fit(input_fixture):
    config, root, roots = input_fixture
    inputs = VolumeInputs(config, root, roots, 'cfg')
    inputs.fit();inputs.freeze_models(dict.fromkeys(c.SYSTEMS[1:], 'hash'), 'power')
    config['prefixes']['CALIBRATE']['volume.csv']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='prefix hash'): inputs.calibration()
    assert not any(k[0] == 1056 for k in inputs.decoded)


def test_failed_admission_cannot_read_evaluation_prefix(monkeypatch):
    inputs = VolumeInputs.__new__(VolumeInputs)
    inputs.stage = 'CALIBRATE';inputs.config_hash = 'cfg'
    inputs.models = dict.fromkeys(c.SYSTEMS[1:], 'model');inputs.power = 'power'
    monkeypatch.setattr(inputs, '_csvs', lambda: pytest.fail('Must not expand source prefix'))
    blob = json.dumps({'calibration_information_gate':False}).encode()
    with pytest.raises(ValueError, match='not admitted'):
        inputs.evaluation_inputs(blob, hashlib.sha256(blob).hexdigest())


def test_all_prediction_hashes_before_evaluation_truth(monkeypatch):
    inputs = VolumeInputs.__new__(VolumeInputs);inputs.stage = 'EVALUATE_INPUTS'
    calls = []
    monkeypatch.setattr(inputs, '_cache', lambda cut, systems: calls.append((cut, systems)) or {})
    monkeypatch.setattr(inputs, 'lineage_check', lambda: None)
    saved = [dict(system=s,horizon=h) for s in c.SYSTEMS for h in (3,12)]
    with pytest.raises(ValueError, match='Twelve fixed'):
        inputs.evaluation_truth(saved)
    assert calls == []
    for row in saved: row['sha256'] = '1'*64
    inputs.evaluation_truth(saved)
    assert calls == [(1392, ('truth',))]
