"""License-aware local dataset registration.

This command never redistributes or silently downloads raw data. It records the
user-selected local root and upstream terms after basic existence checks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "urbanev": {
        "license": "CC0-1.0 (as distributed with the obtained dataset)",
        "source": "See docs/DATA_LICENSES.md and the paper bibliography.",
        "requires_acceptance": False,
    },
    "paris": {
        "license": "ODbL source terms / CC BY 4.0 derived release; verify the selected upstream source",
        "source": "See docs/DATA_LICENSES.md and the paper bibliography.",
        "requires_acceptance": True,
    },
}


def register(dataset: str, data_root: Path, accept_license: bool) -> Path:
    spec = DATASETS[dataset]
    if spec["requires_acceptance"] and not accept_license:
        raise SystemExit("Paris registration requires --accept-license after reviewing docs/DATA_LICENSES.md")
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist or is not a directory: {data_root}")
    marker = data_root / f".urbanev_{dataset}_registration.json"
    payload = {
        "schema_version": "urbanev-local-data-registration/v1",
        "dataset": dataset,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_notice": spec["license"],
        "source_notice": spec["source"],
        "license_accepted": bool(accept_license or not spec["requires_acceptance"]),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()
    marker = register(args.dataset, args.data_root, args.accept_license)
    print(f"registered local {args.dataset} data: {marker}")


if __name__ == "__main__":
    main()
