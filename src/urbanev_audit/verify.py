"""Verify public manifests, assets, local registrations, and target-free schemas."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .artifacts import load_manifest, validate_npz

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".tex", ".yml", ".yaml", ".toml"}


def sha256_file(path: Path, normalize_repository_text: bool = False) -> str:
    digest = hashlib.sha256()
    if normalize_repository_text and path.suffix.lower() in TEXT_SUFFIXES:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path: Path, data_root: Path | None = None) -> list[str]:
    manifest = load_manifest(manifest_path)
    repo_root = manifest_path.resolve().parent
    while repo_root.parent != repo_root and not (repo_root / "pyproject.toml").is_file():
        repo_root = repo_root.parent
    if not (repo_root / "pyproject.toml").is_file():
        raise ValueError("could not locate repository root from manifest path")
    messages: list[str] = []
    for item in manifest.get("tracked_files", []):
        path = repo_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path, normalize_repository_text=True)
        if actual != item["sha256"]:
            raise ValueError(f"hash mismatch: {item['path']}")
        messages.append(f"verified {item['path']}")
    release_dir = repo_root / "release-assets"
    checksums = release_dir / "SHA256SUMS"
    if checksums.is_file():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(None, 1)
            asset = release_dir / name.strip()
            if not asset.is_file():
                raise FileNotFoundError(f"release asset missing: {asset.name}; run scripts/download_release_assets.py")
            if sha256_file(asset) != expected.lower():
                raise ValueError(f"release checksum mismatch: {asset.name}")
            messages.append(f"verified release checksum {asset.name}")
    for path in sorted(release_dir.rglob("*.npz")):
        validate_npz(path)
        messages.append(f"validated target-free schema {path.name}")
    if data_root is not None:
        registrations = list(data_root.glob(".urbanev_*_registration.json"))
        if not registrations:
            raise ValueError("no local dataset registration; run urbanev_audit register-data first")
        messages.append(f"found {len(registrations)} local dataset registration(s)")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    for message in verify(args.manifest, args.data_root):
        print(message)


if __name__ == "__main__":
    main()
