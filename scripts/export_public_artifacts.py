#!/usr/bin/env python3
"""Export deterministic target-free packages using registered family whitelists."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanev_audit.artifacts import (  # noqa: E402
    FAMILY_KEYS,
    FAMILY_SCHEMA_VERSIONS,
    MANIFEST_SCHEMA_VERSION,
    PRIVATE_SOURCE_KEYS,
    infer_family,
    validate_npz,
)

NORMALIZED_KEYS = {
    "hard_router": "hard",
    "soft_router": "soft",
    "direct_loss_mlp": "mlp",
    "hard_gate": "gates_hard",
    "soft_gate": "gates_soft",
    "mlp_gate": "gates_mlp",
}


def array_hash(values: np.ndarray) -> str:
    values = np.asarray(values)
    canonical = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def export_one(source: Path, destination: Path, dataset: str) -> dict[str, object]:
    source_label = "__".join(source.parts[-3:])
    with np.load(source, allow_pickle=False) as package:
        source_keys = set(package.files)
        normalized_keys = {NORMALIZED_KEYS.get(key, key) for key in source_keys}
        source_dataset = _scalar_text(package["dataset"]) if "dataset" in source_keys else dataset
        if source_dataset != dataset:
            raise ValueError(f"dataset mismatch in {source}")
        family = infer_family(normalized_keys, dataset)
        allowed = FAMILY_KEYS[family]
        unknown = normalized_keys - allowed - PRIVATE_SOURCE_KEYS
        if unknown:
            raise ValueError(f"unregistered source keys in {source.name}: {sorted(unknown)}")
        arrays = {
            NORMALIZED_KEYS.get(key, key): np.asarray(package[key])
            for key in package.files
            if NORMALIZED_KEYS.get(key, key) in allowed
            and NORMALIZED_KEYS.get(key, key) not in {"schema_version", "family"}
        }
        target = np.asarray(package["target"]) if "target" in package else None
        if target is None and "validation_target" in package:
            target = np.asarray(package["validation_target"])
        if target is not None:
            target_hash = array_hash(target)
            arrays["target_sha256"] = np.asarray(target_hash)
            arrays["target_id"] = np.asarray(f"{dataset}_{target_hash[:16]}")
            arrays["target_dtype"] = np.asarray(str(target.dtype))
            arrays["target_shape"] = np.asarray(target.shape, dtype=np.int64)
        for required in (
            "target_id", "target_sha256", "target_dtype", "target_shape", "target_construction",
        ):
            if required not in arrays:
                raise ValueError(f"source lacks required irreversible target metadata: {required}")
        arrays["schema_version"] = np.asarray(FAMILY_SCHEMA_VERSIONS[family])
        arrays["family"] = np.asarray(family)
        arrays["dataset"] = np.asarray(dataset)
        if "source_label" not in arrays:
            arrays["source_label"] = np.asarray(source_label)
        for key, value in arrays.items():
            if np.asarray(value).dtype.hasobject:
                raise ValueError(f"object array is not exportable: {source.name}:{key}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    result = validate_npz(destination)
    return {
        "file": destination.name,
        "family": family,
        "dataset": dataset,
        "source_label": _scalar_text(arrays["source_label"]),
        "target_id": _scalar_text(arrays["target_id"]),
        "target_sha256": _scalar_text(arrays["target_sha256"]),
        "target_dtype": _scalar_text(arrays["target_dtype"]),
        "target_shape": result["target_shape"],
        "target_construction": _scalar_text(arrays["target_construction"]),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "keys": result["keys"],
    }


def deterministic_zip(source_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, (2026, 8, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["urbanev", "paris-development"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty: {args.output_dir}")
    records = []
    for index, source in enumerate(args.inputs, start=1):
        name = f"{args.dataset}_{index:04d}.npz"
        records.append(export_one(source, args.output_dir / name, args.dataset))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": args.dataset,
        "target_values_included": False,
        "formal_or_protected_included": False,
        "packages": records,
    }
    (args.output_dir / "SCHEMA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    deterministic_zip(args.output_dir, args.zip_path)
    print(f"wrote {args.zip_path} with {len(records)} package(s)")


if __name__ == "__main__":
    main()
