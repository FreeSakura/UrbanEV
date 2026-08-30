import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from urbanev_audit.artifacts import (
    FAMILY_SCHEMA_VERSIONS,
    MANIFEST_SCHEMA_VERSION,
    validate_npz,
    validate_release_zip,
)


def _distillation_arrays() -> dict[str, np.ndarray]:
    shape = (2, 2)
    return {
        "schema_version": np.asarray(FAMILY_SCHEMA_VERSIONS["distillation"]),
        "family": np.asarray("distillation"),
        "dataset": np.asarray("urbanev"),
        "source_label": np.asarray("fixture_distillation.npz"),
        "target_id": np.asarray("urbanev_" + "0" * 16),
        "target_sha256": np.asarray("0" * 64),
        "target_dtype": np.asarray("float32"),
        "target_shape": np.asarray(shape, dtype=np.int64),
        "target_construction": np.asarray("occupancy_float32"),
        "prediction": np.zeros(shape, dtype=np.float32),
        "student_residual": np.zeros(shape, dtype=np.float32),
        "target_index": np.arange(shape[0], dtype=np.int64),
    }


def test_target_free_family_package_passes(tmp_path: Path):
    path = tmp_path / "public.npz"
    np.savez(path, **_distillation_arrays())
    result = validate_npz(path)
    assert result["family"] == "distillation"
    assert result["prediction_keys"] == ["prediction", "student_residual"]


@pytest.mark.parametrize("key", ["target", "y_true", "winner_label"])
def test_sensitive_array_fails(tmp_path: Path, key: str):
    path = tmp_path / "private.npz"
    arrays = _distillation_arrays()
    arrays[key] = np.zeros((2, 2))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="forbidden|unknown"):
        validate_npz(path)


def test_unknown_key_fails(tmp_path: Path):
    path = tmp_path / "unknown.npz"
    arrays = _distillation_arrays()
    arrays["unregistered_diagnostic"] = np.zeros((2, 2))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="unknown keys"):
        validate_npz(path)


def test_object_array_fails(tmp_path: Path):
    path = tmp_path / "object.npz"
    arrays = _distillation_arrays()
    arrays["source_label"] = np.asarray([{"unsafe": True}], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Object arrays|object array"):
        validate_npz(path)


def test_release_manifest_is_bidirectional(tmp_path: Path):
    package_path = tmp_path / "fixture.npz"
    np.savez_compressed(package_path, **_distillation_arrays())
    package_bytes = package_path.read_bytes()
    result = validate_npz(package_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "urbanev",
        "target_values_included": False,
        "formal_or_protected_included": False,
        "packages": [{
            "file": "fixture.npz",
            "family": "distillation",
            "sha256": hashlib.sha256(package_bytes).hexdigest(),
            "keys": result["keys"],
        }],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    zip_path = tmp_path / "release.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("fixture.npz", package_bytes)
        archive.writestr("SCHEMA_MANIFEST.json", manifest_bytes)
    assert validate_release_zip(zip_path).packages == 1

    with zipfile.ZipFile(tmp_path / "extra.zip", "w") as archive:
        archive.writestr("fixture.npz", package_bytes)
        archive.writestr("extra.txt", b"not registered")
        archive.writestr("SCHEMA_MANIFEST.json", manifest_bytes)
    with pytest.raises(ValueError, match="member mismatch"):
        validate_release_zip(tmp_path / "extra.zip")
