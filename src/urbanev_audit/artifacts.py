"""Schema validation for target-free public prediction packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "urbanev-public-prediction/v1"
FORBIDDEN_ARRAY_KEYS = frozenset({"target", "validation_target", "oracle"})
PREDICTION_KEYS = frozenset(
    {
        "prediction",
        "predictions",
        "raw_prediction",
        "clipped_prediction",
        "student_residual",
        "caper",
        "timexer",
        "global_fixed",
        "horizon_fixed",
        "hard",
        "soft",
        "mlp",
        "base",
        "experts",
        "weights",
        "gates",
    }
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "urbanev-public-manifest/v1":
        raise ValueError("unsupported public manifest schema")
    return manifest


def validate_npz(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as package:
        keys = set(package.files)
        forbidden = sorted(keys & FORBIDDEN_ARRAY_KEYS)
        if forbidden:
            raise ValueError(f"forbidden arrays in {path.name}: {forbidden}")
        prediction_keys = sorted(keys & PREDICTION_KEYS)
        if not prediction_keys:
            raise ValueError(f"no recognized prediction array in {path.name}")
        for key in prediction_keys:
            array = np.asarray(package[key])
            if array.dtype.hasobject:
                raise ValueError(f"object array is not allowed: {path.name}:{key}")
        return {"path": str(path), "keys": sorted(keys), "prediction_keys": prediction_keys}
