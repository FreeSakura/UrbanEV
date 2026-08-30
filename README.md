# UrbanEV evidence audit

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/FreeSakura/UrbanEV?include_prereleases)](https://github.com/FreeSakura/UrbanEV/releases)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Paper: CC BY 4.0](https://img.shields.io/badge/paper-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Unofficial research artifact.**
> This repository is not the official UrbanEV dataset repository and is not maintained by the UrbanEV dataset authors.

This is the public, target-free companion to **From Configurations to Claims: An Evidence Audit of Urban EV Charging Forecasting**. It preserves the paper sources, frozen protocol and result summaries, executed model-source snapshots, and deterministic checks needed to inspect the reported evidence chain without publishing licensed targets or protected material.

- [Main manuscript](paper/main/UrbanEV_Evidence_Audit_Main.pdf)
- [Supplementary Material](paper/supplement/UrbanEV_Evidence_Audit_Supplement.pdf)
- [Current preprint release](https://github.com/FreeSakura/UrbanEV/releases/tag/v0.9.1-preprint)

## Evidence boundary

The current public tree and current Release contain only allowlisted source, configuration, summary, hash, and target-free prediction artifacts. They do **not** contain UrbanEV or Paris raw data, Paris formal/protected data or predictions, target values, model checkpoints, access tokens, physical local paths, or recoverable private Git objects. The public Paris receipt records only non-sensitive role metadata and an analytical-access count of zero.

The immutable `v0.9.0-preprint` tag is retained as a superseded historical record. Its known non-secret path-metadata defect is documented without reproducing those paths in `artifacts/manifests/HISTORICAL_PRIVACY_EXCEPTIONS.json`; the defect is removed from the current tree and `v0.9.1-preprint` assets.

The paper is an audit/evaluation study. It does not claim state-of-the-art performance, protected-test performance, production readiness, or cross-dataset superiority.

## What can be reproduced?

| Level | Raw targets required? | Scope |
|---|---:|---|
| Repository-only replay | No | Schemas, summaries, citations, manifests, paper build, privacy gates |
| Licensed-data recomputation | Yes, obtained by the user | Target hashes and headline metrics from target-free predictions |
| Optional model re-execution | Yes; GPU environment for relevant models | Frozen source/configuration identity, not required for stored-prediction claims |

## Clean-room route

Python 3.10 is the reference CPU environment. CUDA 12.1 is documented separately for optional model execution.

```bash
git clone https://github.com/FreeSakura/UrbanEV.git
cd UrbanEV
python -m venv .venv
python -m pip install -e .[test]
python -m urbanev_audit register-data --dataset urbanev --data-root /path/to/urbanev
python -m urbanev_audit register-data --dataset paris --accept-license --data-root /path/to/paris
python scripts/download_release_assets.py --tag v0.9.1-preprint
python scripts/audit_release.py --asset-root release-assets
python -m urbanev_audit.verify --manifest artifacts/manifests/FULL_RELEASE_MANIFEST.json
python -m urbanev_audit.recompute --scope headline --data-root /path/to/data --development-shard /path/to/paris/development_state_shard.csv
python scripts/build_paper.py --variant main
python scripts/build_paper.py --variant supplement
```

`register-data` records data already obtained by the user under upstream terms; it does not download or redistribute data. The deprecated `fetch` command remains as a compatibility alias. UrbanEV reconstruction reads `occupancy.csv` and `inf.csv`. Paris reconstruction defaults to an explicit development shard and reads a complete `train.csv` only with `--allow-full-source`; requested timestamps after the frozen development end fail closed.

## Repository map

- `paper/`: current Main and Supplement, historical archive, shared bibliography, figures, and PDFs.
- `src/urbanev_audit/`: metrics, strict schemas, verification, recomputation, and privacy checks.
- `models/`: executed source snapshots and file-level provenance/licensing manifest.
- `configs/`: frozen model, fold, gate, and environment specifications.
- `artifacts/summaries/`: public result tables and decision receipts.
- `artifacts/manifests/`: evidence, full-tree, model, paper-build, Release, and privacy records.
- `scripts/`: paper build, asset export/audit, manifest, and privacy utilities.
- `tests/`: clean-room metric, schema, citation, release, and privacy checks.
- `docs/`: licenses, identity limits, disclosure, and reproduction notes.

## Stable commands

```bash
python -m urbanev_audit register-data --help
python -m urbanev_audit verify --help
python -m urbanev_audit recompute --help
python scripts/build_paper.py --variant main  # or: supplement, archive
python scripts/privacy_audit.py --root . --git-history
python scripts/audit_release.py --asset-root release-assets
```

## Releases and licenses

Large target-free prediction packages are attached to GitHub Releases rather than committed to Git. Verify each asset against `SHA256SUMS`, then run the strict Release audit.

Original code is MIT licensed. Paper and original documentation are CC BY 4.0. Third-party snapshots retain their upstream licenses. Dataset licenses remain separate; see `docs/DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`.

## Citation

Use `CITATION.cff`. This remains a preprint artifact until the remaining submission metadata and archival record are locked.
