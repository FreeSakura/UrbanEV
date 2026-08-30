"""Convenience command dispatcher."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m urbanev_audit")
    parser.add_argument("command", choices=["fetch", "verify", "recompute"])
    args, remaining = parser.parse_known_args()
    if args.command == "fetch":
        from . import fetch
        import sys
        sys.argv = ["urbanev_audit.fetch", *remaining]
        fetch.main()
    elif args.command == "verify":
        from . import verify
        import sys
        sys.argv = ["urbanev_audit.verify", *remaining]
        verify.main()
    else:
        from . import recompute
        import sys
        sys.argv = ["urbanev_audit.recompute", *remaining]
        recompute.main()


if __name__ == "__main__":
    main()
