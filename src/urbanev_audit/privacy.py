"""Structured privacy and secret checks for repositories and release artifacts."""

from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import tomli
import yaml
from pypdf import PdfReader

from .artifacts import FORBIDDEN_ARRAY_KEYS

TEXT_SUFFIXES = {
    ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".tex", ".py",
    ".cff", ".xml", ".ini", ".cfg", ".bib", ".sh", ".ps1", ".bat",
}
ZIP_SUFFIXES = {".zip", ".docx"}
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "tmp", "editable"}
SKIP_FILES = {"scripts/build_editable_papers.py"}
ALLOWED_PUBLIC_URIS = ("private-evidence://", "artifact://", "repository://")


def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    slash = r"[\\/]"
    drive = r"\b[A-Za-z]:" + slash
    escaped_drive = r"\b[A-Za-z]:(?:\\\\){2}"
    unc = r"(?:\\\\|(?<!:)//)[A-Za-z0-9._-]+" + slash + r"[A-Za-z0-9$_.-]+"
    posix_home = "/" + "home" + r"/[^/\s\"']+/"
    mac_home = "/" + "Users" + r"/[^/\s\"']+/"
    file_uri = "file" + r":/{2,3}"
    private_store = "restricted" + "-evidence" + "-store"
    private_vault = "paris" + "evidence" + "vault"
    private_uri = "private" + r"-(?:workspace|local)://"
    return (
        ("windows_absolute_path", re.compile(drive, re.IGNORECASE)),
        ("json_escaped_windows_path", re.compile(escaped_drive, re.IGNORECASE)),
        ("unc_path", re.compile(unc, re.IGNORECASE)),
        ("linux_home_path", re.compile(posix_home)),
        ("user_directory", re.compile(mac_home)),
        ("file_uri", re.compile(file_uri, re.IGNORECASE)),
        ("private_physical_marker", re.compile(private_store, re.IGNORECASE)),
        ("private_vault_marker", re.compile(private_vault, re.IGNORECASE)),
        ("deprecated_private_uri", re.compile(private_uri, re.IGNORECASE)),
        ("github_token", re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")),
        ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
        ("generic_bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{24,}")),
    )


PATTERNS = _patterns()


def _scan_text(text: str, location: str) -> list[str]:
    findings: list[str] = []
    scrubbed = text
    for uri in ALLOWED_PUBLIC_URIS:
        scrubbed = scrubbed.replace(uri, "public-alias://")
    for label, pattern in PATTERNS:
        matches = list(pattern.finditer(scrubbed))
        if label == "unc_path":
            matches = [
                match
                for match in matches
                if not re.sub(r"\s+", "", scrubbed[max(0, match.start() - 16):match.start()]).lower().endswith(
                    ("http:", "https:", "public-alias:")
                )
            ]
        if matches:
            findings.append(f"{location}: {label}")
    return findings


def _scan_value(value: Any, location: str) -> list[str]:
    findings: list[str] = []
    if isinstance(value, str):
        findings.extend(_scan_text(value, location))
    elif isinstance(value, dict):
        for key, child in value.items():
            findings.extend(_scan_text(str(key), f"{location}:<key>"))
            findings.extend(_scan_value(child, f"{location}:{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_scan_value(child, f"{location}[{index}]"))
    return findings


def audit_text_bytes(data: bytes, location: str, suffix: str = "") -> list[str]:
    text = data.decode("utf-8", errors="replace")
    findings = _scan_text(text, location)
    try:
        if suffix == ".json":
            findings.extend(_scan_value(json.loads(text), location))
        elif suffix in {".yaml", ".yml", ".cff"}:
            findings.extend(_scan_value(yaml.safe_load(text), location))
        elif suffix == ".toml":
            findings.extend(_scan_value(tomli.loads(text), location))
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        findings.append(f"{location}: structured_parse_error:{type(exc).__name__}")
    return sorted(set(findings))


def audit_pdf_bytes(data: bytes, location: str) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # malformed public PDFs must fail closed
        return [f"{location}: pdf_parse_error:{type(exc).__name__}"]
    return _scan_text(text, location)


def audit_npz_bytes(data: bytes, location: str) -> list[str]:
    findings: list[str] = []
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as package:
            keys = set(package.files)
            forbidden = sorted(keys & FORBIDDEN_ARRAY_KEYS)
            if forbidden:
                findings.append(f"{location}: forbidden array keys {forbidden}")
            for key in package.files:
                findings.extend(_scan_text(key, f"{location}:key:{key}"))
                array = np.asarray(package[key])
                if array.dtype.hasobject:
                    findings.append(f"{location}:{key}: object_array")
                elif array.dtype.kind in "SU":
                    for index, value in enumerate(array.reshape(-1)):
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        findings.extend(_scan_text(str(value), f"{location}:{key}[{index}]"))
    except ValueError as exc:
        findings.append(f"{location}: npz_parse_error:{exc}")
    return sorted(set(findings))


def _safe_zip_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return not pure.is_absolute() and ".." not in pure.parts and "\\" not in name


def audit_zip_bytes(data: bytes, location: str) -> list[str]:
    findings: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                findings.append(f"{location}: duplicate_zip_members")
            for name in names:
                findings.extend(_scan_text(name, f"{location}:{name}:member_name"))
                if not _safe_zip_name(name):
                    findings.append(f"{location}:{name}: unsafe_zip_member")
                    continue
                if name.endswith("/"):
                    continue
                payload = archive.read(name)
                suffix = Path(name).suffix.lower()
                member_location = f"{location}:{name}"
                if suffix in TEXT_SUFFIXES:
                    findings.extend(audit_text_bytes(payload, member_location, suffix))
                elif suffix == ".npz":
                    findings.extend(audit_npz_bytes(payload, member_location))
                elif suffix == ".pdf":
                    findings.extend(audit_pdf_bytes(payload, member_location))
    except (OSError, zipfile.BadZipFile) as exc:
        findings.append(f"{location}: zip_parse_error:{type(exc).__name__}")
    return sorted(set(findings))


def audit_file(path: Path, display_name: str | None = None) -> list[str]:
    location = display_name or str(path)
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix in TEXT_SUFFIXES:
        return audit_text_bytes(data, location, suffix)
    if suffix == ".npz":
        return audit_npz_bytes(data, location)
    if suffix == ".pdf":
        return audit_pdf_bytes(data, location)
    if suffix in ZIP_SUFFIXES:
        return audit_zip_bytes(data, location)
    return []


def audit_root(root: Path) -> list[str]:
    root = root.resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or relative.as_posix() in SKIP_FILES
            or any(part in SKIP_PARTS for part in relative.parts)
        ):
            continue
        findings.extend(audit_file(path, relative.as_posix()))
    return sorted(set(findings))


def audit_git_history(root: Path) -> list[str]:
    root = root.resolve()
    exception_path = root / "artifacts/manifests/HISTORICAL_PRIVACY_EXCEPTIONS.json"
    allowed_blobs: set[str] = set()
    if exception_path.is_file():
        payload = json.loads(exception_path.read_text(encoding="utf-8"))
        allowed_blobs = {item["git_blob"] for item in payload.get("exceptions", [])}
    result = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    objects: dict[str, str] = {}
    for line in result.stdout.splitlines():
        object_id, separator, name = line.partition(" ")
        if separator and name and object_id not in objects:
            objects[object_id] = name
    findings: list[str] = []
    for object_id, name in objects.items():
        if object_id in allowed_blobs:
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in TEXT_SUFFIXES | {".npz", ".pdf"} | ZIP_SUFFIXES:
            continue
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        location = f"git:{object_id[:12]}:{name}"
        if suffix in TEXT_SUFFIXES:
            blob_findings = audit_text_bytes(blob, location, suffix)
            # Superseded releases used non-physical role aliases. Current-tree
            # policy rejects them, while history auditing focuses on physical
            # paths and credentials unless a blob is explicitly excepted.
            findings.extend(item for item in blob_findings if "deprecated_private_uri" not in item)
        elif suffix == ".npz":
            findings.extend(audit_npz_bytes(blob, location))
        elif suffix == ".pdf":
            findings.extend(audit_pdf_bytes(blob, location))
        else:
            findings.extend(audit_zip_bytes(blob, location))
    return sorted(set(findings))
