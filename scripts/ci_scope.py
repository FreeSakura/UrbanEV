"""Select CI work from changed paths; docs do not need training or LaTeX."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def classify(paths):
    scope = {"tests": False, "forecasting": False, "papers": False}
    environment = {"pyproject.toml", "requirements-cpu.txt", "environment-gpu-cu121.yml", "scripts/ci_scope.py"}
    for path in paths:
        if path in environment or path.startswith(".github/workflows/"):
            return dict.fromkeys(scope, True)
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
        scope = dict.fromkeys(("tests", "forecasting", "papers"), True)
    else:
        result = subprocess.run(["git", "diff", "--name-only", "-z", base, "HEAD"], capture_output=True)
        if result.returncode:
            scope = dict.fromkeys(("tests", "forecasting", "papers"), True)
        else:
            scope = classify(result.stdout.decode("utf-8").strip("\0").split("\0"))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        for key, value in scope.items():
            handle.write(f"{key}={str(value).lower()}\n")
    print(json.dumps(scope, sort_keys=True))


if __name__ == "__main__":
    main()
