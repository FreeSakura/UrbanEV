import numpy as np
import pytest

from urbanev_audit.metrics import equal_cell_macro, relative_gain, rmse, sha256_array


def test_metric_primitives_are_deterministic():
    target = np.array([1.0, 2.0, 4.0])
    prediction = np.array([1.0, 3.0, 2.0])
    assert rmse(prediction, target) == pytest.approx(np.sqrt(5.0 / 3.0))
    assert relative_gain(10.0, 9.0) == pytest.approx(10.0)
    assert equal_cell_macro([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert sha256_array(target) == sha256_array(target.copy())
