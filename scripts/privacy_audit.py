#!/usr/bin/env python3
"""Run the public repository privacy/secret gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanev_audit.privacy import audit_file, audit_git_history, audit_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--git-history", action="store_true")
    parser.add_argument("--extra", action="append", type=Path, default=[])
    args = parser.parse_args()
    findings = audit_root(args.root.resolve())
    if args.git_history:
        findings.extend(audit_git_history(args.root.resolve()))
    for path in args.extra:
        findings.extend(audit_file(path.resolve(), f"extra:{path.name}"))
    findings = sorted(set(findings))
    if findings:
        print("privacy audit FAILED")
        print("\n".join(findings))
        raise SystemExit(1)
    print("privacy audit PASS")


if __name__ == "__main__":
    main()
