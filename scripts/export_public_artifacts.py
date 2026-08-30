#!/usr/bin/env python3
"""Create deterministic, target-free NPZ packages from allowlisted prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

FORBIDDEN = {"target", "validation_target", "oracle"}
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


def export_one(source: Path, destination: Path, dataset: str) -> dict[str, object]:
    source_label = "__".join(source.parts[-3:])
    with np.load(source, allow_pickle=True) as package:
        arrays = {
            NORMALIZED_KEYS.get(key, key): np.asarray(package[key])
            for key in package.files
            if key not in FORBIDDEN
        }
        target = None
        for key in ("target", "validation_target"):
            if key in package:
                target = np.asarray(package[key])
                break
        companion = source.with_name("targets.npz")
        if target is None and companion.is_file():
            with np.load(companion, allow_pickle=False) as target_package:
                if "target" in target_package:
                    target = np.asarray(target_package["target"])
                if "target_index" in target_package and "target_index" not in arrays:
                    arrays["target_index"] = np.asarray(target_package["target_index"])
        if target is None:
            target_hash = "not-contained-in-source"
            target_id = f"{dataset}_unavailable_{hashlib.sha256(source.name.encode()).hexdigest()[:12]}"
        else:
            target_hash = array_hash(target)
            target_id = f"{dataset}_{target_hash[:16]}"
        arrays["schema_version"] = np.asarray("urbanev-public-prediction/v1")
        arrays["dataset"] = np.asarray(dataset)
        arrays["source_label"] = np.asarray(source_label)
        arrays["target_id"] = np.asarray(target_id)
        arrays["target_sha256"] = np.asarray(target_hash)
        arrays["target_dtype"] = np.asarray(str(target.dtype) if target is not None else "unknown")
        arrays["target_shape"] = np.asarray(target.shape if target is not None else (), dtype=np.int64)
        if dataset == "urbanev" and "caper" in arrays:
            target_construction = "occupancy_float32_then_storage_dtype"
        elif dataset == "urbanev" and target is not None and target.dtype == np.float64:
            target_construction = "occupancy_float64"
        elif dataset == "urbanev":
            target_construction = "occupancy_float32"
        else:
            target_construction = "paris_state_fraction_float32"
        arrays["target_construction"] = np.asarray(target_construction)
        for key, value in list(arrays.items()):
            if value.dtype.hasobject:
                arrays[key] = value.astype(str)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {
        "file": destination.name,
        "dataset": dataset,
        "source_label": source_label,
        "target_id": target_id,
        "target_sha256": target_hash,
        "target_dtype": str(target.dtype) if target is not None else "unknown",
        "target_shape": list(target.shape) if target is not None else [],
        "target_construction": target_construction,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "keys": sorted(arrays),
    }


def deterministic_zip(source_dir: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(source_dir).as_posix(), (2026, 8, 30, 0, 0, 0))
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
    records = []
    for index, source in enumerate(args.inputs, start=1):
        name = f"{args.dataset}_{index:04d}.npz"
        records.append(export_one(source, args.output_dir / name, args.dataset))
    manifest = {
        "schema_version": "urbanev-public-release/v1",
        "dataset": args.dataset,
        "target_values_included": False,
        "formal_or_protected_included": False,
        "packages": records,
    }
    (args.output_dir / "SCHEMA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    deterministic_zip(args.output_dir, args.zip_path)
    print(f"wrote {args.zip_path} with {len(records)} package(s)")


if __name__ == "__main__":
    main()
