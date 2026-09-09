from itertools import product
import numpy as np
import pytest
from urbanev_audit.paired_events import event_bounds, paired_brier_bounds, joint_brier_bounds


def brute(observed, a, b, K, L):
    indices = np.flatnonzero(np.asarray(observed) == -1)
    scores = []
    for values in product((0, 1), repeat=len(indices)):
        z = np.array(observed)
        z[indices] = values
        event, _ = event_bounds(z, K, L)
        scores.append(float(np.sum((event-a)**2-(event-b)**2)))
    return min(scores), max(scores)


def test_overlap_counterexample():
    result = joint_brier_bounds([0, -1, 0], [.25, .75], [.75, .25], 2, 1)
    assert result["lower_sum"] == pytest.approx(0)
    assert result["upper_sum"] == pytest.approx(0)
    assert result["independent_lower_sum"] == pytest.approx(-1)
    assert result["independent_upper_sum"] == pytest.approx(1)


def test_joint_bounds_exhaustive_and_single_queries():
    rng = np.random.default_rng(909)
    for K in (1, 2, 3, 4):
        for L in range(1, K+1):
            for _ in range(12):
                T = 6
                obs = rng.choice([-1, 0, 1], T)
                a, b = rng.random((2, T-K+1))
                result = joint_brier_bounds(obs, a, b, K, L, query_planning=True)
                lo, hi = brute(obs, a, b, K, L)
                assert result["lower_sum"] == pytest.approx(lo)
                assert result["upper_sum"] == pytest.approx(hi)
                assert lo >= result["independent_lower_sum"]-1e-12
                assert hi <= result["independent_upper_sum"]+1e-12
                for query in result["query_planning"]:
                    widths = []
                    for branch in query["branches"]:
                        revealed = obs.copy()
                        revealed[query["index"]] = branch["revealed_bit"]
                        qlo, qhi = brute(revealed, a, b, K, L)
                        assert branch["lower_sum"] == pytest.approx(qlo)
                        assert branch["upper_sum"] == pytest.approx(qhi)
                        widths.append(qhi-qlo)
                    assert query["guaranteed_width_reduction_sum"] == pytest.approx(max(0, hi-lo-max(widths)))


def test_complete_labels_and_equal_predictions():
    obs = [1, 1, 0, 1, 1]
    r = joint_brier_bounds(obs, [.2, .3], [.7, .5], 4, 2)
    assert r["lower_sum"] == pytest.approx(r["upper_sum"])
    r = joint_brier_bounds([-1]*5, [.4, .6], [.4, .6], 4, 2, query_planning=True)
    assert r["lower_sum"] == pytest.approx(0)
    assert r["upper_sum"] == pytest.approx(0)
    assert all(q["guaranteed_width_reduction_sum"] == 0 for q in r["query_planning"])


def test_validation_and_binary_endpoints():
    with pytest.raises(ValueError):
        joint_brier_bounds([-1, 2], [.3], [.6], 2, 1)
    with pytest.raises(ValueError):
        joint_brier_bounds([-1, 1], [1.2], [.6], 2, 1)
    with pytest.raises(ValueError):
        paired_brier_bounds([.2], [.3], [1], [0])
