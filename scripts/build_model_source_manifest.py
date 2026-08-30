#!/usr/bin/env python3
"""Create the file-level executed model-source manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml", ".toml"}


def repository_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_SUFFIXES else data


def metadata(relative: Path) -> tuple[str, str, str]:
    family = relative.parts[0]
    if family == "timexer":
        return "Time-Series-Library executed local snapshot", "MIT", "executed local adaptation"
    if family == "caper":
        return "project-authored local CAPER-family snapshot", "MIT", "paper-to-code identity unknown/not claimed"
    if family == "paris":
        return "project-authored Paris development adapter", "MIT", "executed project source; provenance documented"
    return "UrbanEV project implementation", "MIT", "executed project source"


def main() -> None:
    rows = []
    paths = sorted(MODELS.rglob("*"), key=lambda path: path.relative_to(MODELS).as_posix().casefold())
    for path in paths:
        if not path.is_file() or path.name == "MODEL_SOURCE_MANIFEST.csv":
            continue
        relative = path.relative_to(MODELS)
        origin, license_name, state = metadata(relative)
        data = repository_bytes(path)
        rows.append(
            {
                "repository_path": f"models/{relative.as_posix()}",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "origin": origin,
                "license": license_name,
                "local_modification_status": state,
            }
        )
    output = MODELS / "MODEL_SOURCE_MANIFEST.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
