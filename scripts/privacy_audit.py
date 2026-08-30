#!/usr/bin/env python3
"""Run the public repository privacy/secret gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbanev_audit.privacy import audit_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = audit_root(args.root.resolve())
    if findings:
        print("privacy audit FAILED")
        print("\n".join(findings))
        raise SystemExit(1)
    print("privacy audit PASS")


if __name__ == "__main__":
    main()
