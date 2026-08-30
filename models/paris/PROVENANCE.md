# Paris adapter provenance

## Distribution status

The files in this directory are treated as project-authored local adapters and are released under the repository MIT license. They implement the frozen Paris development-only protocol through documented input fields and local model interfaces. No upstream tutorial source file, copyright header, or GPL-licensed code is distributed in this directory.

The implementation covers hourly state-fraction construction, development folds, masking, local CAPER/TimeXer adaptation, prediction export, strict convex teacher aggregation, and frozen gate reporting. It differs from upstream data tutorials by providing the paper-specific model, fold, mask, target-identity, and fail-closed audit contracts.

## Source boundary

- Implementation origin: project-authored local implementation.
- Exact first-implementation timestamp: not established from the public snapshot; the snapshot was frozen for the 2026-08-30 public release.
- Upstream code copied into this directory: none identified.
- Upstream specifications consulted: public dataset fields and terms, the paper protocol, and the Time-Series-Library model interface documented in `PARIS_TEACHER_MODEL_CONFIG.json`.
- GPL tutorial code: not included and not relicensed.
- License basis: project copyright for the local adapter, distributed under MIT.

This provenance statement separates software authorship from dataset licensing and from paper-to-code identity. It is an artifact record, not a legal opinion.
