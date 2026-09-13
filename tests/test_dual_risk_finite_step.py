import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest
import urbanev_forecast.dual_risk_finite_step as c
from urbanev_forecast.finite_step_metrics import score_cell, macro_score


def point(alpha, rmse=.1, mae=.05):
    cells = [dict(horizon=h, rmse=rmse, mae=mae, raw_rmse=rmse, raw_mae=mae,
                  baseline_rmse=.1, baseline_mae=.05) for h in (3, 12)]
    return {'alpha': alpha, 'cells': cells, 'macro': macro_score(cells)}


def test_tie_uses_global_min_and_smallest_feasible_alpha():
    native = point(0)
    rows = [point(a, .09 if a else .1) for a in c.GRID]
    rows[-1] = point(8., .09-5e-13)
    assert c.choose_step(rows, native)['alpha'] == .125
    rows[-1] = point(8., .09-2e-12)
    assert c.choose_step(rows, native)['alpha'] == 8.


def test_no_mae_tolerance_and_legal_zero_grid():
    native = point(0)
    rows = [point(a, .09, np.nextafter(.05, 1.)) if a else native for a in c.GRID]
    result = c.choose_step(rows, native)
    assert result['alpha'] == 0 and result['selection_status'] == 'GRID_ZERO_ONLY_FEASIBLE'
    zero = [point(a) for a in c.GRID]
    assert c.choose_step(zero, native)['alpha'] == 0


def test_whole_system_Cstar_exact_tie_and_required_controls():
    selected = {s: point(1, .1, .05) for s in c.SYSTEMS}
    selected['native']['alpha'] = 0
    selected['gate_duration'] = point(1, .098, .049)
    g = c.compare_gate(selected)
    assert g['Cstar'] == 'native' and g['information_pass'] and g['method_pass']
    del selected['hard_duration']
    with pytest.raises(ValueError, match='All8'):
        c.compare_gate(selected)


def test_reference_zero_has_no_improvement_space():
    selected = {s: point(0, 0., 0.) for s in c.SYSTEMS}
    result = c.compare_gate(selected)
    assert result['rmse_gain_percent_vs_Cstar'] is None
    assert not result['information_pass']


def test_shared_alpha_uses_both_horizons():
    native = point(0)
    rows = [point(a, .099, .04) if a else native for a in c.GRID]
    rows[1]['cells'][0]['rmse'] = .08
    rows[1]['cells'][1]['rmse'] = .101
    rows[1]['macro'] = macro_score(rows[1]['cells'])
    assert c.choose_step(rows, native)['alpha'] == .125


def test_hard_soft_and_mean_directions_with_inward_boundaries(monkeypatch):
    p = np.array([0., .5, 1.])
    monkeypatch.setattr(c, 'design', lambda *a: (np.ones((3, 2)), p))
    monkeypatch.setattr(c, 'predict_moments', lambda *a: (np.array([-.2, .3, .2]),
                        np.array([[.8, .1, .1], [.1, .1, .8], [.1, .1, .8]])))
    _, d = c.fixed_directions(dict.fromkeys(c.REPRESENTATIONS), {}, 1056)
    np.testing.assert_allclose(d['mean_duration'][3], [0, .3, 0])
    np.testing.assert_allclose(d['hard_duration'][3], [0, .3, 0])
    np.testing.assert_allclose(d['gate_duration'][3], [0, .18, 0])


def fit_arrays():
    arrays = {}
    for h in (3, 12):
        for i, s in enumerate(('truth',)+c.SOURCE_SYSTEMS):
            arrays[720, h, s] = np.full((14, h, 275), .2+.01*i)
    return arrays


def test_future_labels_never_enter_four_fit_model(monkeypatch):
    arrays = fit_arrays()
    arrays[1056, 3, 'truth'] = np.array([0.])
    arrays[1392, 3, 'truth'] = np.array([0.])
    original = c.fit_ridge
    calls = []
    def fit(cells, **kw):
        calls.append(len(cells))
        return original(cells, **kw)
    monkeypatch.setattr(c, 'fit_ridge', fit)
    a = c.fit_only_720(arrays)
    arrays[1056, 3, 'truth'][:] = 1.
    arrays[1392, 3, 'truth'][:] = 1.
    b = c.fit_only_720(arrays)
    assert calls == [2]*8  # Four per distinct synthetic construction.
    for rep in c.REPRESENTATIONS:
        for k in ('center', 'scale', 'coefficient'):
            np.testing.assert_array_equal(a[rep][k], b[rep][k])


def test_future_labels_cannot_change_calibration_alpha():
    arrays = {(1056, h, 'truth'): np.full(h, .25) for h in (3, 12)}
    p = {h: np.full(h, .2) for h in (3, 12)}
    d = {s: {h: np.full(h, .01) if s != 'native' else np.zeros(h) for h in (3, 12)} for s in c.SYSTEMS}
    arrays[1392, 3, 'truth'] = np.array([0.])
    a, _ = c.calibrate(arrays, p, d)
    arrays[1392, 3, 'truth'][:] = 1.
    b, _ = c.calibrate(arrays, p, d)
    assert {s: v['alpha'] for s, v in a.items()} == {s: v['alpha'] for s, v in b.items()}


@pytest.fixture
def cache_files(tmp_path):
    records = []
    for cut in (720, 1056, 1392):
        for h in (3, 12):
            for s in ('truth',)+c.SOURCE_SYSTEMS:
                name = f'private_truth_{cut}_h{h}.npy' if s == 'truth' else f'private_{cut}_h{h}_{s}_a1.npy'
                blob = io.BytesIO()
                np.save(blob, np.full((14, h, 275), .2))
                data = blob.getvalue()
                (tmp_path/name).write_bytes(data)
                records.append({'cut': cut, 'horizon': h, 'system': s, 'file': name,
                                'root_role': 'truth' if s == 'truth' else 'v1',
                                'shape': [14, h, 275], 'sha256': hashlib.sha256(data).hexdigest()})
    return records, {'v1': tmp_path, 'truth': tmp_path}


def test_all_hashes_before_decode_and_wrong_whitelist(cache_files, monkeypatch):
    records, roots = cache_files
    bad = copy.deepcopy(records)
    bad[-1]['sha256'] = '0'*64
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('No decode before complete hashes'))
    with pytest.raises(ValueError, match='hash mismatch'):
        c.StagedCache(bad, roots, 'cfg')
    with pytest.raises(ValueError, match='whitelist'):
        c.StagedCache(records[:-1], roots, 'cfg')


def test_calibration_failure_does_not_release1392(cache_files):
    records, roots = cache_files
    cache = c.StagedCache(records, roots, 'cfg')
    with pytest.raises(ValueError, match='CALIBRATE'):
        cache.calibration_arrays()
    with pytest.raises(ValueError, match='outside current stage'):
        cache._decode(1392, ('truth',))
    cache.fit_arrays()
    models = dict.fromkeys(c.REPRESENTATIONS, 'model-hash')
    cache.models_saved(models)
    cache.calibration_arrays()
    selection = {'calibration_information_gate': False, 'config_sha256': 'cfg',
                 'models': models, 'alphas': dict.fromkeys(c.SYSTEMS, 0.)}
    blob = json.dumps(selection).encode()
    with pytest.raises(ValueError, match='not admitted'):
        cache.evaluation_inputs(blob, hashlib.sha256(blob).hexdigest())
    assert len(cache.decoded) == 24
    assert not any(k[0] == 1392 for k in cache.decoded)


def test_evaluation_inputs_precede_truth_and_selection_identity(cache_files):
    records, roots = cache_files
    cache = c.StagedCache(records, roots, 'cfg')
    cache.fit_arrays()
    models = dict.fromkeys(c.REPRESENTATIONS, 'model-hash')
    cache.models_saved(models)
    cache.calibration_arrays()
    blob = json.dumps({'calibration_information_gate': True, 'config_sha256': 'cfg',
                      'models': models, 'alphas': dict.fromkeys(c.SYSTEMS, 0.)}).encode()
    with pytest.raises(ValueError, match='hash mismatch'):
        cache.evaluation_inputs(blob, '0'*64)
    arrays = cache.evaluation_inputs(blob, hashlib.sha256(blob).hexdigest())
    assert len(arrays) == 10 and not any(k[2] == 'truth' for k in arrays)
    with pytest.raises(ValueError, match='All16'):
        cache.evaluation_truth([])
    truth = cache.evaluation_truth([dict(system=s, horizon=h, sha256='1'*64) for s in c.SYSTEMS for h in (3, 12)])
    assert len(truth) == 2 and len(cache.decoded) == 36
    assert cache.ledger[-1]['stage'] == 'EVALUATE_TRUTH'


def test_evaluation_uses_only_one_locked_point_and_no_refit(monkeypatch):
    p = {h: np.full(h, .2) for h in (3, 12)}
    v = {s: {h: np.full(h, .01) for h in (3, 12)} for s in c.SYSTEMS}
    alphas = {s: c.GRID[i] for i, s in enumerate(c.SYSTEMS)}
    raw = {(s, h): p[h]+alphas[s]*v[s][h] for s in c.SYSTEMS for h in (3, 12)}
    truth = {(1392, h, 'truth'): np.full(h, .25) for h in (3, 12)}
    calls = []
    original = c.score_cell
    def score(y, p, d, alpha, **kw):
        calls.append(alpha)
        return original(y, p, d, alpha, **kw)
    monkeypatch.setattr(c, 'score_cell', score)
    monkeypatch.setattr(c, 'fit_ridge', lambda *a, **k: pytest.fail('No post-selection fit'))
    result = c.evaluate_locked(truth, p, v, alphas, raw)
    assert len(result) == 8 and calls == [alphas[s] for s in c.SYSTEMS for _ in (3, 12)]
    raw['gate_duration', 3] = raw['gate_duration', 3]+.01
    with pytest.raises(ValueError, match='prediction changed'):
        c.evaluate_locked(truth, p, v, alphas, raw)


def test_runner_no_go_never_calls_evaluation(cache_files, tmp_path):
    spec = importlib.util.spec_from_file_location('finite_runner_test', Path(__file__).resolve().parents[1]/'scripts/research/run_dual_risk_finite_step.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    records, roots = cache_files
    cache = c.StagedCache(records, roots, 'cfg')
    output = tmp_path/'output'
    output.mkdir()
    result, _, predictions = runner.run_stages(cache, output, 'cfg', lambda: None, {'fitting_closed': False})
    assert result['status'] == 'FINITE_STEP_CALIBRATION_NO_GO'
    assert result['evaluation']['status'] == 'NOT_RUN' and predictions == []
    assert not any(k[0] == 1392 for k in cache.decoded)
    assert (output/'evaluation_scores.csv').read_text().count('NOT_RUN') == 1


def test_mse_and_rmse_loss_identity_at_selected_clipped_point():
    result = score_cell(np.array([0., .4, 1.]), np.array([.1, .4, .9]), np.array([-1., .2, 1.]), 8.)
    assert abs(result['rmse_identity_error']) < 1e-10
    assert abs(result['clipped_mae_identity_error']) < 1e-10
