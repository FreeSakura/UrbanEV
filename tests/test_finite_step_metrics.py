import numpy as np
import pytest
from urbanev_forecast.finite_step_metrics import macro_score, score_cell


def test_alpha_zero_is_exact_native_with_atoms():
    y, p, h = np.array([0., .3, 1.]), np.array([0., .4, 1.]), np.array([.1, -.2, -.1])
    result = score_cell(y, p, h, 0.)
    assert result['mae'] == result['baseline_mae']
    assert result['rmse'] == result['baseline_rmse']
    assert result['mae_linear_term'] == result['mae_crossing_term'] == 0


def test_crossing_and_atom_identity_without_clipping():
    r = score_cell(np.array([.3, .5, .6]), np.array([.2, .5, .8]), np.array([.2, -.1, -.3]), 1.)
    assert r['mae_crossing_term'] > 0
    assert abs(r['raw_mae_identity_error']) < 1e-15
    assert abs(r['clipped_mae_identity_error']) < 1e-15
    assert r['raw_outside_fraction'] == 0


def test_clipping_accounted_separately_from_raw_identity():
    r = score_cell(np.array([0.]), np.array([.2]), np.array([-1.]), 1.)
    assert r['mae_linear_term'] == -1.
    assert r['mae_crossing_term'] == pytest.approx(1.6)
    assert r['mae_clipping_adjustment'] == pytest.approx(-.8)
    assert r['mae'] == 0.
    assert abs(r['clipped_mae_identity_error']) < 1e-15


def test_macro_is_equal_cells_not_pooled_mse():
    a = score_cell(np.zeros(1), np.full(1, .1), np.zeros(1), 0.)
    b = score_cell(np.zeros(4), np.full(4, .9), np.zeros(4), 0.)
    m = macro_score([a, b])
    assert m['rmse'] == pytest.approx(.5)
    assert m['mae'] == pytest.approx(.5)


@pytest.mark.parametrize('alpha', [-1., float('nan'), float('inf')])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        score_cell(np.array([.2]), np.array([.3]), np.array([.1]), alpha)


def test_invalid_shapes_and_bounded_origin():
    with pytest.raises(ValueError):
        score_cell(np.array([.2]), np.array([.3, .4]), np.array([.1]), 1.)
    with pytest.raises(ValueError):
        score_cell(np.array([.2]), np.array([1.1]), np.array([.1]), 1.)
