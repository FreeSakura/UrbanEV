from pathlib import Path

import numpy as np
import pytest

from urbanev_audit.artifacts import validate_npz


def test_target_free_package_passes(tmp_path: Path):
    path = tmp_path / "public.npz"
    np.savez(path, prediction=np.zeros((2, 2)), target_sha256=np.asarray("0" * 64))
    assert validate_npz(path)["prediction_keys"] == ["prediction"]


def test_target_array_fails(tmp_path: Path):
    path = tmp_path / "private.npz"
    np.savez(path, prediction=np.zeros(2), target=np.zeros(2))
    with pytest.raises(ValueError, match="forbidden"):
        validate_npz(path)
