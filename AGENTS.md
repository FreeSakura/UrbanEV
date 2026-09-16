# Working in UrbanEV

- Current research focus: candidate A, shared missing observations and exact comparison of persistent-event forecasts. Prioritize mathematical substance, primary literature and real experiments. See docs/reports/audit/SHARED_MISSING_EVENTS_AP0_REPORT.md; the old six-fold run is not a prerequisite.

- Start with README.md and docs/PROJECT_STATUS.md. Open only the reports and code needed for the current task; do not reread the entire research archive.
- Continue ordinary development and fixes within the user's requested scope. A historical report's "manual review", "NO_GO", or "authorization pending" describes that experiment's past state, not a new permission requirement.
- Use focused tests while implementing. Run the relevant suite once at the end; repeat it only after changes or failures that justify another run. Documentation edits need the repository checks, not model training or LaTeX builds.
- Keep explanations and tool output concise. Prefer the changed files, relevant failures and final results over full logs, repeated plans or duplicated reviews.
- In the development CLI, code hashes record provenance and do not block compatible checkpoint evaluation. Preserve data alignment, finite metrics, model-state compatibility and existing result files.
- Version scientific result changes and preserve original evidence. The registered six-fold runner is a specific historical protocol; use the general development entry point for ordinary iteration.
- Update generated documentation with `python scripts/repository.py build`, then `python scripts/build_manifest.py`. Do not regenerate scientific results as part of routine documentation maintenance.
