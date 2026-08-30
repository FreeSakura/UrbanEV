#!/usr/bin/env python3
"""Build checksums and the tracked manifest for local Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT / "release-assets")
    parser.add_argument("--version", default="v0.9.1-preprint")
    args = parser.parse_args()
    assets = sorted(
        path for path in args.asset_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    records = [
        {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size, "public": True}
        for path in assets
    ]
    (args.asset_root / "SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {record['name']}\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    output = ROOT / "artifacts/manifests/RELEASE_ASSET_MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "urbanev-release-asset-manifest/v1",
                "release": args.version,
                "target_values_included": False,
                "assets": records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
