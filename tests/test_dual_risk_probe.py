import copy
import hashlib
import io

import numpy as np
import pytest

from urbanev_forecast.dual_risk_probe import (
    SYSTEMS, REPRESENTATIONS, common_direction, design, directional_scores,
    execute, fit_ridge, gate, load_whitelist, moment_targets, predict_moments, project_simplex,
)


def test_common_descent_population_identity_and_atoms():
    rng = np.random.default_rng(123)
    probabilities = rng.dirichlet([1, 1, 1], 100)
    mean = rng.uniform(-1, 1, 100)
    s = probabilities[:, 0]-probabilities[:, 2]
    atom = probabilities[:, 1]
    k = np.maximum(-s*np.sign(mean)-atom, 0)
    h = common_direction(mean, probabilities, np.full(100, .5))
    np.testing.assert_allclose(s*h+atom*np.abs(h), -np.abs(mean)*k*k, atol=2e-16)
    assert np.all(mean*h >= 0)
    assert common_direction(np.array([.1]), np.array([[0., .9, .1]]), np.array([0.]))[0] == 0


def test_median_counterexample_and_nonunique_exception():
    y = np.array([0.]*9+[1.])
    q = .05
    assert np.mean((y-q)**2) < np.mean(y*y)
    assert np.mean(np.abs(y-q)) > np.mean(np.abs(y))
    y = np.array([0., 1.])
    assert np.mean(abs(y-.5)) == np.mean(abs(y))
    assert np.mean((y-.5)**2) < np.mean(y*y)


def test_simplex_euclidean_projection_and_tangent_boundaries():
    p = project_simplex(np.array([[2., -1., 0.], [.2, .3, .5], [-1., -1., -1.]]))
    np.testing.assert_allclose(p, [[1, 0, 0], [.2, .3, .5], [1/3]*3], atol=1e-15)
    h = common_direction(np.array([-1., 1., .5]), np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]]), np.array([0., 1., .5]))
    np.testing.assert_allclose(h, [0, 0, .5])


def test_directional_derivatives_match_one_sided_finite_difference_with_atom():
    y, p, h = np.array([.2, .5, .9]), np.array([.1, .5, .8]), np.array([.02, -.01, .03])
    result = directional_scores(y, p, h)
    step = 1e-6
    da = (np.mean(abs(y-p-step*h))-np.mean(abs(y-p)))/step
    dr = (np.sqrt(np.mean((y-p-step*h)**2))-np.sqrt(np.mean((y-p)**2)))/step
    assert da == pytest.approx(result['mae_derivative'], abs=1e-9)
    assert dr == pytest.approx(result['rmse_derivative'], abs=1e-8)
    with pytest.raises(ValueError, match='zero native'):
        directional_scores(y, y, h)


def test_weighted_ridge_unit_cell_weights_and_unpenalized_intercept():
    a = np.column_stack((np.ones(2), np.zeros(2)))
    b = np.column_stack((np.ones(8), np.zeros(8)))
    m = fit_ridge([(a, np.tile([0, 1, 0, 0], (2, 1))), (b, np.tile([1, 0, 0, 1], (8, 1)))])
    pred, probs = predict_moments(m, a)
    np.testing.assert_allclose(pred, .5, atol=1e-14)
    np.testing.assert_allclose(probs, [[.5, 0, .5]]*2, atol=1e-14)
    assert m['scale'][0] == 0


def test_ridge_invariant_to_repeating_a_horizon_cell_and_eval_no_mutation():
    rng = np.random.default_rng(7)
    x = np.column_stack((np.ones(10), rng.normal(size=(10, 3))))
    t = rng.normal(size=(10, 4))
    m1 = fit_ridge([(x, t), (x+np.array([0, 1, 1, 1]), t)])
    m2 = fit_ridge([(np.repeat(x, 4, axis=0), np.repeat(t, 4, axis=0)), (x+np.array([0, 1, 1, 1]), t)])
    np.testing.assert_allclose(m1['coefficient'], m2['coefficient'], atol=1e-12)
    frozen = copy.deepcopy(m1)
    predict_moments(m1, x*100)
    for key in ('center', 'scale', 'coefficient'):
        np.testing.assert_array_equal(m1[key], frozen[key])


def test_features_raw_contrasts_calendar_and_atomic_targets():
    arrays = {(720, 3, s): np.full((14, 3, 275), i*.1-.1) for i, s in enumerate(SYSTEMS)}
    x, p = design(arrays, 720, 3, 'duration')
    assert x.shape == (11550, 11)
    assert np.all(x[:, 6] == 0)
    np.testing.assert_allclose(x[:, 10], .1)
    assert np.all(p == 0) and np.all(x[:, 8] == 1)
    np.testing.assert_array_equal(moment_targets(np.array([0., .5, 1.]), np.full(3, .5))[:, 1:], np.eye(3))


def test_gate_strictness_and_no_compatible_control():
    rows = [{'evaluation_cut': c, 'representation': rep, 'mae_derivative': -1.,
             'rmse_derivative': -1., 'direction_norm': 1., 'normalized_rmse_descent': 2. if rep == 'duration' else 1.}
            for c in (1056, 1392) for rep in REPRESENTATIONS]
    assert gate(rows)['status'].startswith('MECHANISM_SUPPORTED')
    rows[1]['mae_derivative'] = 0.
    assert gate(rows)['status'].startswith('MECHANISM_NOT_SUPPORTED')
    for row in rows:
        row['mae_derivative'] = -.1 if row['representation'] == 'duration' else .1
    assert gate(rows)['window_checks'][0]['reference_efficiency'] == 0


def test_whitelist_rejects_extra_or_wrong_identity_before_numeric_read(tmp_path, monkeypatch):
    roots = {'v1': tmp_path, 'truth': tmp_path}
    rows = []
    for c in (720, 1056, 1392):
        for h in (3, 12):
            for s in ('truth',)+SYSTEMS:
                name = f'private_truth_{c}_h{h}.npy' if s == 'truth' else f'private_{c}_h{h}_{s}_a1.npy'
                blob = io.BytesIO()
                np.save(blob, np.zeros((14, h, 275)))
                data = blob.getvalue()
                (tmp_path/name).write_bytes(data)
                rows.append(dict(cut=c, horizon=h, system=s, root_role='truth' if s == 'truth' else 'v1', file=name,
                                 shape=[14, h, 275], sha256=hashlib.sha256(data).hexdigest()))
    manifest = {'files': rows}
    arrays, _ = load_whitelist(manifest, roots)
    assert len(arrays) == 36
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Numerical read before complete hash checks'))
    bad = copy.deepcopy(manifest)
    bad['files'][-1]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash mismatch'):
        load_whitelist(bad, roots)
    bad['files'].append(bad['files'][0])
    with pytest.raises(ValueError, match='36 unique'):
        load_whitelist(bad, roots)


def test_forbidden_foundation_import_guard():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/research/run_dual_risk_probe.py'
    spec = importlib.util.spec_from_file_location('probe_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guard = module.RejectHeavyImports()
    with pytest.raises(RuntimeError, match='forbidden'):
        guard.find_spec('urbanev_forecast.foundation')
    assert guard.attempts == 1


def test_forward_probe_eight_fits_and_macro_aggregation(monkeypatch):
    import urbanev_forecast.dual_risk_probe as module
    arrays = {}
    for c in (720, 1056, 1392):
        for h in (3, 12):
            for i, s in enumerate(SYSTEMS):
                arrays[c, h, s] = np.full((14, h, 275), .3+.01*i)
            arrays[c, h, 'truth'] = np.full((14, h, 275), .4+(h == 12)*.1)
    sizes = []
    original = module.fit_ridge

    def record(cells):
        sizes.append(len(cells))
        return original(cells)

    monkeypatch.setattr(module, 'fit_ridge', record)
    result = execute(arrays)
    assert sizes == [2]*4+[4]*4
    assert len(result['moment_scores']) == 16
    assert len(result['fits']) == 8
    for window in result['windows']:
        cells = [r for r in result['cells'] if r['representation'] == window['representation']
                 and r['evaluation_cut'] == window['evaluation_cut']]
        assert window['rmse_derivative'] == pytest.approx(sum(r['rmse_derivative'] for r in cells)/2)
        assert window['mae_derivative'] == pytest.approx(sum(r['mae_derivative'] for r in cells)/2)
