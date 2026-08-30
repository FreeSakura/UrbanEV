#!/usr/bin/env python3
"""Audit downloaded Release assets, strict schemas, checksums, and privacy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanev_audit.artifacts import validate_release_zip  # noqa: E402
from urbanev_audit.privacy import audit_file  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT / "release-assets")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    checksums = asset_root / "SHA256SUMS"
    if not checksums.is_file():
        raise SystemExit("missing SHA256SUMS")
    expected: dict[str, str] = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[name.strip()] = digest.lower()
    actual_assets = sorted(
        path.name for path in asset_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    if sorted(expected) != actual_assets:
        raise ValueError(
            f"SHA256SUMS membership mismatch; missing={sorted(set(expected)-set(actual_assets))}, "
            f"extra={sorted(set(actual_assets)-set(expected))}"
        )
    for name, digest in expected.items():
        if sha256_file(asset_root / name) != digest:
            raise ValueError(f"release checksum mismatch: {name}")

    validations = []
    for zip_path in sorted(asset_root.glob("*.zip")):
        prefix = "paris-development" if zip_path.name.startswith("paris-development") else "urbanev"
        manifests = sorted(asset_root.glob(f"{prefix}-schema-manifest-*.json"))
        if len(manifests) != 1:
            raise ValueError(f"expected one external schema manifest for {zip_path.name}")
        result = validate_release_zip(zip_path, manifests[0])
        validations.append(
            {
                "asset": zip_path.name,
                "dataset": result.dataset,
                "packages": result.packages,
                "members": len(result.members),
                "manifest_sha256": result.manifest_sha256,
            }
        )

    findings = []
    for name in actual_assets + ["SHA256SUMS"]:
        findings.extend(audit_file(asset_root / name, f"release:{name}"))
    if findings:
        raise ValueError("release privacy audit failed:\n" + "\n".join(sorted(set(findings))))

    report = {
        "schema_version": "urbanev-release-audit-report/v1",
        "status": "PASS",
        "checksums": len(expected),
        "release_zips": validations,
        "privacy_findings": 0,
        "target_values_included": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
