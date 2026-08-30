# UrbanEV evidence audit

This repository is the public, target-free companion to **From Configurations to Claims: An Evidence Audit of Urban EV Charging Forecasting**. It preserves the paper sources, frozen protocol and result summaries, executed model-source snapshots, and deterministic checks needed to inspect the reported evidence chain without publishing licensed targets or protected material.

## Evidence boundary

The public repository contains only allowlisted source, configuration, summary, hash, and target-free prediction artifacts. It does **not** contain UrbanEV or Paris raw data, Paris formal/protected data or predictions, private evidence-vault metadata, model checkpoints, access tokens, local paths, or recoverable private Git objects. The public Paris boundary receipt records only non-sensitive role metadata and the analytical-access count of zero.

The paper is an audit/evaluation study. It does not claim state-of-the-art performance, protected-test performance, production readiness, or cross-dataset superiority.

## Clean-room reproduction

Python 3.10 is the reference CPU environment. CUDA 12.1 is documented separately for model execution.

```bash
git clone https://github.com/FreeSakura/UrbanEV.git
cd UrbanEV
python -m venv .venv
python -m pip install -e .[test]
python -m urbanev_audit.fetch --dataset urbanev --data-root /path/to/licensed-data
python -m urbanev_audit.fetch --dataset paris --accept-license --data-root /path/to/licensed-data
python scripts/download_release_assets.py --tag v0.9.0-preprint
python -m urbanev_audit.verify --manifest artifacts/manifest.json --data-root /path/to/licensed-data
python -m urbanev_audit.recompute --scope headline --data-root /path/to/licensed-data
python scripts/build_paper.py --variant main
python scripts/build_paper.py --variant supplement
```

The fetch command records data the user obtained under the upstream terms; it does not redistribute raw datasets. UrbanEV reconstruction reads `occupancy.csv` and `inf.csv`; Paris development reconstruction reads the upstream state-count CSV described in `docs/REPRODUCIBILITY.md`. `recompute` aligns target indices/times and entity IDs, then requires the target hash recorded in each public package before calculating RMSE and gains.

## Repository map

- `paper/`: submission main text, Supplementary Material, preserved archive, shared bibliography, figures, and PDFs.
- `src/urbanev_audit/`: metric, schema, verification, recomputation, and privacy-check interfaces.
- `models/`: source snapshots actually used by the audited runs; see `MODEL_SOURCE_MANIFEST.csv`.
- `configs/`: frozen model, fold, gate, and environment specifications.
- `artifacts/summaries/`: public result tables and decision receipts.
- `scripts/`: paper build, asset download/export, manifest, and privacy utilities.
- `tests/`: clean-room metric, schema, citation, and privacy checks.
- `docs/`: licenses, identity limits, disclosure, and reproduction notes.

## Stable commands

```bash
python -m urbanev_audit.fetch --help
python -m urbanev_audit.verify --help
python -m urbanev_audit.recompute --help
python scripts/build_paper.py --variant main  # or: supplement, archive
python scripts/privacy_audit.py --root .
```

## Releases and licenses

Large target-free prediction packages are attached to the `v0.9.0-preprint` GitHub Release rather than committed to Git. Verify every asset against `SHA256SUMS` before use.

Original code is MIT licensed. Paper and original documentation are CC BY 4.0. Third-party snapshots retain their upstream licenses. Dataset licenses remain separate and are summarized in `docs/DATA_LICENSES.md`.

## Citation

Use `CITATION.cff`. The preprint citation is provisional until the full authorship is finalized.
