#!/usr/bin/env python3
"""Build evidence and full-tree repository-relative SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = (
    ".github", "artifacts", "configs", "docs", "licenses", "models", "paper", "release-assets", "scripts", "src", "tests",
)
ROOT_FILES = (
    ".gitattributes", ".gitignore", "AUTHORS.yml", "CITATION.cff", "CONTRIBUTING.md",
    "environment-gpu-cu121.yml", "LICENSE", "pyproject.toml", "README.md",
    "requirements-cpu.txt", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
)
EVIDENCE_ROOTS = ("configs", "artifacts/summaries", "models")
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".txt", ".tex", ".yml", ".yaml", ".toml", ".cff", ".bib",
}
EXCLUDED_PARTS = {"editable", "__pycache__", ".pytest_cache", "build", "tmp"}
EXCLUDED_FILES = {
    "artifacts/manifests/FULL_RELEASE_MANIFEST.json",
    "paper/archive/main_archive.pdf",
    "paper/main/main.pdf",
    "paper/supplement/supplement.pdf",
    "scripts/build_editable_papers.py",
}
BUILD_SUFFIXES = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc"}


def repository_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_SUFFIXES else data


def metadata(relative: str) -> tuple[str, str, str, str]:
    if relative.startswith("models/timexer/"):
        return "model_source", "Time-Series-Library local snapshot", "MIT", "public"
    if relative.startswith("paper/") or relative.startswith("docs/"):
        return "paper_or_documentation", "UrbanEV evidence-audit project", "CC-BY-4.0", "public"
    if relative.startswith("licenses/"):
        return "license_text", "upstream license authority", "as-named", "public"
    if relative.startswith("artifacts/summaries/"):
        return "evidence_summary", "frozen project artifact; publicly sanitized", "CC-BY-4.0", "public"
    role = relative.split("/", 1)[0].lstrip(".") or "repository"
    return role, "UrbanEV evidence-audit project", "MIT", "public"


def _eligible(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return (
        path.is_file()
        and not any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
        and relative not in EXCLUDED_FILES
        and path.suffix.lower() not in BUILD_SUFFIXES
        and path.name not in {"build_report.json", "compile.log"}
    )


def _records(paths: list[Path]) -> list[dict[str, object]]:
    records = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix().casefold()):
        if not _eligible(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        data = repository_bytes(path)
        role, origin, license_name, visibility = metadata(relative)
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "role": role,
                "origin": origin,
                "license": license_name,
                "public_or_external": visibility,
            }
        )
    return records


def main() -> None:
    evidence_paths = []
    for name in EVIDENCE_ROOTS:
        evidence_paths.extend((ROOT / name).rglob("*"))
    evidence = {
        "schema_version": "urbanev-evidence-artifact-manifest/v1",
        "tracked_files": _records(evidence_paths),
        "release_assets": "See artifacts/manifests/RELEASE_ASSET_MANIFEST.json.",
    }

    manifest_dir = ROOT / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    evidence_bytes = (json.dumps(evidence, indent=2) + "\n").encode()
    (manifest_dir / "EVIDENCE_ARTIFACT_MANIFEST.json").write_bytes(evidence_bytes)
    (ROOT / "artifacts/manifest.json").write_bytes(evidence_bytes)

    full_paths = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for name in PUBLIC_ROOTS:
        full_paths.extend((ROOT / name).rglob("*"))
    full = {
        "schema_version": "urbanev-full-release-manifest/v1",
        "tracked_files": _records(full_paths),
        "release_assets": "See artifacts/manifests/RELEASE_ASSET_MANIFEST.json.",
    }
    output = manifest_dir / "FULL_RELEASE_MANIFEST.json"
    output.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
