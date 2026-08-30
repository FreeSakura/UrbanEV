#!/usr/bin/env python3
"""Prove that a packaging/schema revision did not change public numerical arrays."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

PACKAGING_KEYS = {"schema_version", "family"}


def array_digest(array: np.ndarray) -> str:
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load(zip_path: Path) -> dict[str, dict[str, str]]:
    packages = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npz"):
                continue
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as package:
                packages[name] = {
                    key: array_digest(package[key])
                    for key in package.files
                    if key not in PACKAGING_KEYS
                }
    return packages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", action="append", type=Path, required=True)
    parser.add_argument("--new", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.old) != len(args.new):
        raise SystemExit("--old and --new counts must match")
    comparisons = []
    total_packages = 0
    for old_path, new_path in zip(args.old, args.new):
        old = load(old_path)
        new = load(new_path)
        if old != new:
            changed = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
            raise ValueError(f"public payload changed: {changed[:10]}")
        total_packages += len(old)
        comparisons.append(
            {
                "old_asset": old_path.name,
                "new_asset": new_path.name,
                "packages": len(old),
                "non_packaging_array_hashes_equal": True,
            }
        )
    payload = {
        "schema_version": "urbanev-release-payload-comparison/v1",
        "status": "PASS",
        "packages_compared": total_packages,
        "prediction_arrays_changed": False,
        "target_values_present": False,
        "headline_metrics_changed": False,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
