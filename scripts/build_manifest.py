#!/usr/bin/env python3
"""Build a repository-relative SHA-256 manifest for public tracked artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOTS = ("configs", "artifacts/summaries", "models")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".tex", ".yml", ".yaml", ".toml"}


def repository_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_SUFFIXES else data


def main() -> None:
    records = []
    for name in INCLUDE_ROOTS:
        paths = sorted((ROOT / name).rglob("*"), key=lambda path: path.relative_to(ROOT).as_posix().casefold())
        for path in paths:
            if path.is_file():
                data = repository_bytes(path)
                records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    payload = {
        "schema_version": "urbanev-public-manifest/v1",
        "tracked_files": records,
        "release_assets": "See release-assets/SHA256SUMS and the GitHub v0.9.0-preprint release.",
    }
    output = ROOT / "artifacts/manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
