#!/usr/bin/env python3
"""Download and checksum the target-free GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.9.1-preprint")
    parser.add_argument("--repository", default="FreeSakura/UrbanEV")
    parser.add_argument("--output", type=Path, default=Path("release-assets"))
    args = parser.parse_args()
    api = f"https://api.github.com/repos/{args.repository}/releases/tags/{args.tag}"
    request = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json", "User-Agent": "urbanev-audit/0.9.1"})
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    args.output.mkdir(parents=True, exist_ok=True)
    for asset in release["assets"]:
        destination = args.output / asset["name"]
        urllib.request.urlretrieve(asset["browser_download_url"], destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(f"{digest}  {destination}")
        if destination.suffix.lower() == ".zip":
            extract_root = args.output / destination.stem
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                for member in archive.infolist():
                    target = (extract_root / member.filename).resolve()
                    if extract_root.resolve() not in target.parents and target != extract_root.resolve():
                        raise ValueError(f"unsafe release member: {member.filename}")
                archive.extractall(extract_root)


if __name__ == "__main__":
    main()
