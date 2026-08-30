"""Privacy and secret checks for public repository and release artifacts."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from .artifacts import FORBIDDEN_ARRAY_KEYS

TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".tex", ".py", ".cff"}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?:Users|Codex)[\\/]", re.IGNORECASE),
    re.compile("/" + "home" + r"/[^/\s]+/"),
)
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)
FORBIDDEN_PRIVATE_MARKERS = ("paris" + "evidence" + "vault",)


def audit_npz(path: Path) -> list[str]:
    findings: list[str] = []
    with np.load(path, allow_pickle=False) as package:
        forbidden = sorted(set(package.files) & FORBIDDEN_ARRAY_KEYS)
        if forbidden:
            findings.append(f"{path}: forbidden array keys {forbidden}")
    return findings


def audit_text(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in ABSOLUTE_PATH_PATTERNS + SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"{path}: matches {pattern.pattern}")
    lowered = text.lower()
    for marker in FORBIDDEN_PRIVATE_MARKERS:
        if marker in lowered:
            findings.append(f"{path}: contains private marker {marker}")
    return findings


def audit_root(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            findings.extend(audit_text(path))
        elif path.suffix.lower() == ".npz":
            findings.extend(audit_npz(path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".npz"):
                        with archive.open(name) as src:
                            temp = root / "tmp" / "privacy" / Path(name).name
                            temp.parent.mkdir(parents=True, exist_ok=True)
                            temp.write_bytes(src.read())
                            findings.extend(audit_npz(temp))
                            temp.unlink()
    return findings
