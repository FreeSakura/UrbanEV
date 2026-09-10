import importlib.util
import sys
from pathlib import Path

import numpy as np


def module():
    root = Path(__file__).resolve().parents[1]/'scripts/research'
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('replicate_signals', root/'replicate_signals.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_augmented_context_has_no_future_duration_and_deranges_zones():
    m = module(); o = np.arange(400*275).reshape(400, 275).astype(float)
    d = o+1e6; perm = m.derangement(275)
    assert np.all(perm != np.arange(275)) and sorted(perm) == list(range(275))
    ctx = m.contexts(o, d, 300, 'lagged_duration', perm)
    assert ctx.shape == (168, 550)
    np.testing.assert_array_equal(ctx[-1, :275], o[299])
    np.testing.assert_array_equal(ctx[-1, 275:], d[298])
    np.testing.assert_array_equal(m.contexts(o, d, 300, 'permuted_duration', perm)[:, 275:], d[131:299, perm])


def test_task_output_mapping_keeps_only_correct_occupancy_channels():
    m = module(); groups = [list(range(i, min(i+32, 275))) for i in range(0, 275, 32)]
    class Backend:
        def predict(self, x, tasks, h):
            assert sum(map(len, tasks)) == 550
            return np.tile(sum(tasks, []), (h, 1))
    pred = m.predict_grouped(Backend(), np.zeros((168, 550)), groups, 3, True)
    np.testing.assert_array_equal(pred, np.tile(np.arange(275), (3, 1)))


def test_duplicate_ridge_features_and_flattened_alignment():
    m = module(); o = np.arange(400*275).reshape(400, 275).astype(float)
    d = o+1e6; origins = [300, 301]; perm = m.derangement(275)
    x, anchor = m.ridge_design(o, d, origins, 'duplicate_occupancy', perm)
    np.testing.assert_array_equal(x[:, :168], x[:, 168:])
    np.testing.assert_array_equal(anchor[:275, 0], o[299])
    x, _ = m.ridge_design(o, d, origins, 'permuted_duration', perm)
    np.testing.assert_array_equal(x[:275, 168:], d[131:299, perm].T)
