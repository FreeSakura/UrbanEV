#!/usr/bin/env python3
"""Rewrite private absolute identifiers and record original-to-public hashes."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = (ROOT / "configs", ROOT / "artifacts/summaries", ROOT / "models", ROOT / "paper")
TEXT_SUFFIXES = {".json", ".csv", ".md", ".py", ".tex", ".yml", ".yaml", ".txt"}
REPLACEMENTS = (
    (re.compile(r"D:\\\\Codex\\\\2026-08-23\\\\gen\\\\urbanev-forecast\\\\", re.IGNORECASE), "private-workspace://urbanev-forecast/"),
    (re.compile(r"D:\\\\Codex\\\\2026-08-23\\\\gen\\\\work\\\\UrbanEV-reproduction\\\\", re.IGNORECASE), "private-workspace://UrbanEV-reproduction/"),
    (re.compile(r"D:\\Codex\\2026-08-23\\gen\\urbanev-forecast\\", re.IGNORECASE), "private-workspace://urbanev-forecast/"),
    (re.compile(r"D:\\Codex\\2026-08-23\\gen\\work\\UrbanEV-reproduction\\", re.IGNORECASE), "private-workspace://UrbanEV-reproduction/"),
    (re.compile(r"C:\\\\Users\\\\[^\\\"']+\\\\", re.IGNORECASE), "private-local://redacted/"),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return digest(data)


def main() -> None:
    output = ROOT / "artifacts/manifests/PUBLIC_HASH_MAP.csv"
    mappings = []
    if output.is_file():
        with output.open("r", encoding="utf-8", newline="") as handle:
            mappings.extend(csv.DictReader(handle))
    for scope in SCOPES:
        for path in sorted(scope.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            original = path.read_bytes()
            text = original.decode("utf-8", errors="strict")
            public = text
            for pattern, replacement in REPLACEMENTS:
                public = pattern.sub(replacement, public)
            private_store_marker = "Paris" + "Evidence" + "Vault"
            public = public.replace(private_store_marker, "restricted-evidence-store")
            public_bytes = public.encode("utf-8")
            if public_bytes != original:
                path.write_bytes(public_bytes)
                record = {
                        "public_path": path.relative_to(ROOT).as_posix(),
                        "private_original_sha256": digest(original),
                        "public_sha256": digest(public_bytes),
                        "rewrite": "absolute/private identifier redaction",
                    }
                mappings = [item for item in mappings if item["public_path"] != record["public_path"]]
                mappings.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    for item in mappings:
        public_path = ROOT / item["public_path"]
        if public_path.is_file():
            item["public_sha256"] = public_digest(public_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("public_path", "private_original_sha256", "public_sha256", "rewrite")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(mappings, key=lambda item: item["public_path"]))
    print(f"sanitized {len(mappings)} file(s); wrote {output}")


if __name__ == "__main__":
    main()
