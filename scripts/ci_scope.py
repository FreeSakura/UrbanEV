"""Select documentation, event, current Word, and historical forecasting/TeX checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


SCOPES = ("tests", "forecasting", "papers", "manuscript")
EVENT_SCRIPT_PREFIXES = (
    "run_shared_missing", "verify_shared_missing", "explain_shared_missing",
    "benchmark_shared_missing", "summarize_shared_missing", "analyze_shared_missing",
    "fetch_shared_missing", "replay_shared_missing", "check_shared_missing",
    "verify_persistent_event", "analyze_persistent_event", "validate_persistent_event",
)
EVENT_TESTS = {
    "test_event_cover_geometry.py", "test_event_witnesses.py", "test_persistent_events.py",
    "test_shared_missing_pilot.py", "test_ap1_baselines.py",
}
MANUSCRIPT_BUILDERS = {
    "scripts/build_persistent_event_figures.py", "scripts/build_persistent_event_manuscript.py",
}


def classify(paths):
    scope = dict.fromkeys(SCOPES, False)
    environment = {"pyproject.toml", "requirements-cpu.txt", "environment-gpu-cu121.yml", "scripts/ci_scope.py"}
    for path in paths:
        path = path.replace("\\", "/")
        if path in environment or path.startswith(".github/workflows/"):
            return dict.fromkeys(scope, True)
        if path.startswith("paper/persistent_events/") or path in MANUSCRIPT_BUILDERS:
            scope["manuscript"] = True
            scope["tests"] |= path in MANUSCRIPT_BUILDERS
            continue
        if path.startswith(("artifacts/summaries/persistent_event_", "artifacts/summaries/shared_missing_events_")):
            if not path.endswith(".md"):
                scope["tests"] = scope["manuscript"] = True
            continue
        if ((path.startswith("scripts/research/") and Path(path).stem.startswith(EVENT_SCRIPT_PREFIXES))
                or (path.startswith("tests/") and Path(path).name in EVENT_TESTS)
                or path.startswith(("configs/research/SHARED_MISSING_EVENTS_", "configs/research/PERSISTENT_EVENT_"))):
            if not path.endswith(".md"):
                scope["tests"] = True
            continue
        if path.startswith("paper/") and not path.endswith(".md"):
            scope["papers"] = scope["tests"] = True
        if path.startswith(("scripts/build_paper", "scripts/build_historical_archive", "scripts/verify_paper_manifest")):
            scope["papers"] = scope["tests"] = True
        if path.startswith(("src/", "scripts/", "tests/", "models/", "configs/", "artifacts/summaries/")) and not path.endswith(".md"):
            scope["tests"] = True
        if ((path.startswith(("src/urbanev_forecast/", "scripts/research/", "models/", "configs/research/"))
             and not path.endswith(".md")) or (path.startswith("tests/") and path.endswith(".py")
             and path not in {"tests/test_repository_maintenance.py", "tests/test_ci_scope.py"})):
            scope["forecasting"] = scope["tests"] = True
    return scope


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    base = event.get("pull_request", {}).get("base", {}).get("sha") or event.get("before")
    # Initial pushes have no comparison base. Run the suite instead of blocking.
    if not base or set(base) == {"0"}:
        scope = dict.fromkeys(SCOPES, True)
    else:
        result = subprocess.run(["git", "diff", "--name-only", "-z", base, "HEAD"], capture_output=True)
        if result.returncode:
            scope = dict.fromkeys(SCOPES, True)
        else:
            scope = classify(result.stdout.decode("utf-8").strip("\0").split("\0"))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        for key, value in scope.items():
            handle.write(f"{key}={str(value).lower()}\n")
    print(json.dumps(scope, sort_keys=True))


if __name__ == "__main__":
    main()
