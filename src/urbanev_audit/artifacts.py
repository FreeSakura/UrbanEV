"""Strict schemas for target-free public prediction packages and release ZIPs."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

MANIFEST_SCHEMA_VERSION = "urbanev-public-release/v2"
REPOSITORY_MANIFEST_VERSION = "urbanev-full-release-manifest/v1"

COMMON_KEYS = frozenset(
    {
        "schema_version",
        "family",
        "dataset",
        "source_label",
        "target_id",
        "target_sha256",
        "target_dtype",
        "target_shape",
        "target_construction",
    }
)
FAMILY_KEYS: dict[str, frozenset[str]] = {
    "router": COMMON_KEYS
    | {
        "target_index", "caper", "timexer", "global_fixed", "horizon_fixed",
        "hard", "soft", "mlp", "gates_hard", "gates_soft", "gates_mlp", "zone_ids",
    },
    "chronos": COMMON_KEYS
    | {
        "raw_prediction", "clipped_prediction", "target_index", "forecast_origin",
        "first_history_index", "last_history_index", "target_time", "zone_ids",
    },
    "distillation": COMMON_KEYS | {"prediction", "student_residual", "target_index"},
    "paris_teacher": COMMON_KEYS
    | {
        "raw_prediction", "clipped_prediction", "mask", "target_index", "target_time",
        "target_time_ns", "origin_index", "origin_time", "origin_time_ns", "station_ids",
        "validation_raw_prediction", "validation_clipped_prediction", "validation_mask",
        "validation_target_index", "validation_target_time_ns", "validation_origin_time_ns",
        "eval_target_value_hash", "eval_mask_hash", "fill_hash", "scaler_hash", "fold_capacity_hash",
    },
}
FAMILY_SCHEMA_VERSIONS = {
    family: f"urbanev-public-prediction/{family.replace('_', '-')}/v1"
    for family in FAMILY_KEYS
}
FAMILY_DATASETS = {
    "router": "urbanev",
    "chronos": "urbanev",
    "distillation": "urbanev",
    "paris_teacher": "paris-development",
}

# These exact source-only arrays may exist during export but are never copied.
# Any other unregistered source key makes export fail closed.
PRIVATE_SOURCE_KEYS = frozenset({"target", "validation_target", "oracle"})
FORBIDDEN_ARRAY_KEYS = PRIVATE_SOURCE_KEYS | frozenset(
    {"y_true", "labels", "ground_truth", "future_values", "winner_label"}
)
PREDICTION_KEYS = frozenset(
    {
        "prediction", "raw_prediction", "clipped_prediction", "student_residual", "caper",
        "timexer", "global_fixed", "horizon_fixed", "hard", "soft", "mlp",
        "validation_raw_prediction", "validation_clipped_prediction",
    }
)
HASH_KEYS = frozenset(
    {
        "target_sha256", "eval_target_value_hash", "eval_mask_hash", "fill_hash",
        "scaler_hash", "fold_capacity_hash",
    }
)
SAFE_SOURCE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReleaseValidation:
    dataset: str
    packages: int
    members: tuple[str, ...]
    manifest_sha256: str


def _text_scalar(value: np.ndarray, key: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{key} must be a scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise ValueError(f"{key} must be a string")
    return scalar


def infer_family(keys: set[str], dataset: str) -> str:
    if dataset == "paris-development":
        family = "paris_teacher"
    elif {"caper", "timexer", "global_fixed", "gates_hard"} <= keys:
        family = "router"
    elif {"raw_prediction", "clipped_prediction", "forecast_origin"} <= keys:
        family = "chronos"
    elif {"prediction", "student_residual"} <= keys:
        family = "distillation"
    else:
        raise ValueError("cannot infer a registered package family")
    if dataset != FAMILY_DATASETS[family]:
        raise ValueError(f"dataset/family mismatch: {dataset}/{family}")
    return family


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in {
        "urbanev-public-manifest/v1",
        "urbanev-evidence-artifact-manifest/v1",
        REPOSITORY_MANIFEST_VERSION,
    }:
        raise ValueError("unsupported public manifest schema")
    return manifest


def _require_numeric(array: np.ndarray, key: str) -> None:
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{key} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains a non-finite value")


def _require_shape(array: np.ndarray, expected: tuple[int, ...], key: str) -> None:
    if array.shape != expected:
        raise ValueError(f"{key} shape {array.shape} != target_shape {expected}")


def _validate_loaded(package: Any, display_name: str) -> dict[str, Any]:
    keys = set(package.files)
    forbidden = sorted(keys & FORBIDDEN_ARRAY_KEYS)
    if forbidden:
        raise ValueError(f"forbidden arrays in {display_name}: {forbidden}")
    if "family" not in keys or "dataset" not in keys or "schema_version" not in keys:
        raise ValueError(f"missing family/dataset/schema metadata: {display_name}")
    family = _text_scalar(package["family"], "family")
    dataset = _text_scalar(package["dataset"], "dataset")
    if family not in FAMILY_KEYS:
        raise ValueError(f"unregistered family: {family}")
    allowed = FAMILY_KEYS[family]
    unknown = sorted(keys - allowed)
    missing = sorted(allowed - keys)
    if unknown:
        raise ValueError(f"unknown keys in {display_name}: {unknown}")
    if missing:
        raise ValueError(f"missing keys in {display_name}: {missing}")
    if dataset != FAMILY_DATASETS[family]:
        raise ValueError(f"dataset/family mismatch: {dataset}/{family}")
    if _text_scalar(package["schema_version"], "schema_version") != FAMILY_SCHEMA_VERSIONS[family]:
        raise ValueError(f"schema version mismatch for {family}")

    for key in keys:
        if np.asarray(package[key]).dtype.hasobject:
            raise ValueError(f"object array is not allowed: {display_name}:{key}")

    target_shape_array = np.asarray(package["target_shape"])
    if target_shape_array.ndim != 1 or target_shape_array.dtype.kind not in "iu":
        raise ValueError("target_shape must be a one-dimensional integer array")
    target_shape = tuple(int(item) for item in target_shape_array)
    if len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError(f"invalid target_shape: {target_shape}")

    source_label = _text_scalar(package["source_label"], "source_label")
    if not SAFE_SOURCE_LABEL.fullmatch(source_label) or any(
        part in source_label.lower() for part in ("users", "codex", "vault")
    ):
        raise ValueError("source_label is not a safe repository-relative label")
    for key in HASH_KEYS & keys:
        if not HEX_SHA256.fullmatch(_text_scalar(package[key], key)):
            raise ValueError(f"invalid SHA-256 metadata: {key}")

    prediction_keys = sorted(PREDICTION_KEYS & keys)
    for key in prediction_keys:
        array = np.asarray(package[key])
        _require_numeric(array, key)
        if key != "student_residual" and not key.startswith("validation_"):
            _require_shape(array, target_shape, key)

    if family == "router":
        for key in (
            "caper", "timexer", "global_fixed", "horizon_fixed", "hard", "soft",
            "mlp", "gates_hard", "gates_soft", "gates_mlp",
        ):
            array = np.asarray(package[key])
            _require_numeric(array, key)
            _require_shape(array, target_shape, key)
            if np.any((array < 0) | (array > 1)):
                raise ValueError(f"{key} must be within [0,1]")
        if np.asarray(package["target_index"]).shape != (target_shape[0],):
            raise ValueError("target_index length mismatch")
        if np.asarray(package["zone_ids"]).shape != (target_shape[1],):
            raise ValueError("zone_ids length mismatch")
    elif family == "chronos":
        for key in ("raw_prediction", "clipped_prediction"):
            _require_shape(np.asarray(package[key]), target_shape, key)
        clipped = np.asarray(package["clipped_prediction"])
        if np.any((clipped < 0) | (clipped > 1)):
            raise ValueError("clipped_prediction must be within [0,1]")
        for key in (
            "target_index", "forecast_origin", "first_history_index", "last_history_index", "target_time",
        ):
            if np.asarray(package[key]).shape != (target_shape[0],):
                raise ValueError(f"{key} length mismatch")
        if np.asarray(package["zone_ids"]).shape != (target_shape[1],):
            raise ValueError("zone_ids length mismatch")
    elif family == "distillation":
        for key in ("prediction", "student_residual"):
            _require_shape(np.asarray(package[key]), target_shape, key)
        prediction = np.asarray(package["prediction"])
        if np.any((prediction < 0) | (prediction > 1)):
            raise ValueError("prediction must be within [0,1]")
        if np.asarray(package["target_index"]).shape != (target_shape[0],):
            raise ValueError("target_index length mismatch")
    else:
        for key in ("raw_prediction", "clipped_prediction", "mask"):
            _require_shape(np.asarray(package[key]), target_shape, key)
        if np.asarray(package["mask"]).dtype.kind != "b":
            raise ValueError("mask must be boolean")
        clipped = np.asarray(package["clipped_prediction"])
        if np.any((clipped < 0) | (clipped > 1)):
            raise ValueError("clipped_prediction must be within [0,1]")
        for key in (
            "target_index", "target_time", "target_time_ns", "origin_index", "origin_time", "origin_time_ns",
        ):
            if np.asarray(package[key]).shape != (target_shape[0],):
                raise ValueError(f"{key} length mismatch")
        if np.asarray(package["station_ids"]).shape != (target_shape[1],):
            raise ValueError("station_ids length mismatch")
        validation_shape = np.asarray(package["validation_clipped_prediction"]).shape
        if len(validation_shape) != 2 or validation_shape[1] != target_shape[1]:
            raise ValueError("validation prediction/entity shape mismatch")
        for key in ("validation_raw_prediction", "validation_clipped_prediction", "validation_mask"):
            if np.asarray(package[key]).shape != validation_shape:
                raise ValueError(f"{key} shape mismatch")
        if np.asarray(package["validation_mask"]).dtype.kind != "b":
            raise ValueError("validation_mask must be boolean")
        for key in (
            "validation_target_index", "validation_target_time_ns", "validation_origin_time_ns",
        ):
            if np.asarray(package[key]).shape != (validation_shape[0],):
                raise ValueError(f"{key} length mismatch")
        validation_clipped = np.asarray(package["validation_clipped_prediction"])
        if np.any((validation_clipped < 0) | (validation_clipped > 1)):
            raise ValueError("validation_clipped_prediction must be within [0,1]")

    return {
        "path": display_name,
        "family": family,
        "dataset": dataset,
        "keys": sorted(keys),
        "prediction_keys": prediction_keys,
        "target_shape": list(target_shape),
    }


def validate_npz(path: str | Path | BinaryIO, display_name: str | None = None) -> dict[str, Any]:
    if hasattr(path, "read"):
        source = path
        name = display_name or "<memory>.npz"
    else:
        source = Path(path)
        name = display_name or str(source)
    with np.load(source, allow_pickle=False) as package:
        return _validate_loaded(package, name)


def _safe_member(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or name.endswith("/"):
        raise ValueError(f"unsafe ZIP member: {name}")


def validate_release_zip(
    zip_path: str | Path, external_manifest: str | Path | None = None
) -> ReleaseValidation:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        for name in names:
            _safe_member(name)
        if "SCHEMA_MANIFEST.json" not in names:
            raise ValueError("release ZIP lacks SCHEMA_MANIFEST.json")
        manifest_bytes = archive.read("SCHEMA_MANIFEST.json")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported release manifest schema")
        records = manifest.get("packages")
        if not isinstance(records, list) or not records:
            raise ValueError("release manifest has no package records")
        registered = [record.get("file") for record in records]
        if len(registered) != len(set(registered)) or not all(isinstance(item, str) for item in registered):
            raise ValueError("invalid or duplicate manifest package names")
        actual = sorted(name for name in names if name != "SCHEMA_MANIFEST.json")
        if sorted(registered) != actual:
            missing = sorted(set(registered) - set(actual))
            extra = sorted(set(actual) - set(registered))
            raise ValueError(f"ZIP/manifest member mismatch; missing={missing}, extra={extra}")
        for record in records:
            name = record["file"]
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                raise ValueError(f"package checksum mismatch: {name}")
            result = validate_npz(io.BytesIO(payload), display_name=f"{zip_path.name}:{name}")
            if result["family"] != record.get("family") or result["keys"] != record.get("keys"):
                raise ValueError(f"package manifest metadata mismatch: {name}")
    if external_manifest is not None and Path(external_manifest).read_bytes() != manifest_bytes:
        raise ValueError("external schema manifest differs from ZIP manifest")
    return ReleaseValidation(
        dataset=str(manifest.get("dataset")),
        packages=len(records),
        members=tuple(sorted(names)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
