# Persistent event manuscript and reproduction

The manuscript studies a sharp structural regime for overlapping persistent-event labels under partial observations. Its main theorem is the uniform K ≥ 2L − 1 binary-image characterization. The private-run criterion is a geometric corollary with classical atom/union-family attribution. The cover-LP counterexample uses known min-up/min-down polyhedral structure. The final natural-cache audit includes 2,784 panels and 8,352 fixed model comparisons.

Files:

- [Editable Word full manuscript](WORD_MANUSCRIPT.docx), including the English text, full proof appendices, and Chinese abstract.
- [Editable manuscript source](manuscript.md), with English text, complete proofs, and Chinese abstract.
- [Evidence map](EVIDENCE_MAP.md) and [frozen package versions](requirements-frozen.txt).
- [Complete cover summaries](../../artifacts/summaries/persistent_event_complete_covers_20260917).
- [Core observation geometry](../../src/urbanev_audit/event_cover_geometry.py), [trajectory optimizer](../../src/urbanev_audit/persistent_events.py), and [portable cover validator](../../scripts/research/validate_persistent_event_covers.py).

## Small verification without data

From the repository root, install the scientific requirements and run:

```bash
python -m pip install -e ".[test,evidence]"
python scripts/research/verify_persistent_event_structure.py --output local-data/check-structure
python scripts/research/validate_persistent_event_covers.py --selfcheck --output local-data/check-all-covers
```

Use new output directories. These scripts check constructed cases; they do not train a predictor or download data. No GPU is required.

## Rebuild the two fixed evaluations from public data

The following route rebuilds forecast caches. For the closest numerical reproduction, use Python 3.10 and the versions recorded above. Original archives are public at UCI; their downloaded files and resulting hashes are recorded locally. First download them:

```bash
python scripts/research/fetch_shared_missing_data.py --output local-data/uci-events
```

Then fit/select Evaluation I and reuse its models for Evaluation II:

```bash
python scripts/research/run_shared_missing_ap1.py --phase new --data local-data/uci-events --ap0 local-data/ap0-unused --private local-data/ap1-run --output local-data/ap1-results
python scripts/research/run_shared_missing_ap2.py --data local-data/uci-events --ap1 local-data/ap1-run --private local-data/ap2-run --output local-data/ap2-results
python scripts/research/validate_persistent_event_covers.py --data local-data/uci-events --ap1-private local-data/ap1-run --ap2-private local-data/ap2-run --ap1-results local-data/ap1-results --ap2-results local-data/ap2-results --output local-data/complete-cover-results
```

The AP1 legacy CLI requires `--ap0`, but the `new` phase does not read that directory. AP0 is not needed to rebuild the two evaluations used by this paper. The new validator compares new predictions to the matching newly generated hierarchy, verifies the same observed geometry, and checks complete cover endpoints against a fresh trajectory calculation. It does not assume that a retrained model's binary file hash equals an archived model hash.

To use existing **original** AP1/AP2 prediction caches with the committed hierarchy tables, omit the two results flags and add `--strict-frozen-hashes`. That additionally checks original observation/prediction hashes against the published endpoint witness ledger. Without that flag, generated input hashes still record provenance but are not falsely described as matches to the original private arrays.

The source distribution does not include raw UCI archives or fitted model bundles. Regenerating them is an executable route, not a guarantee of bitwise-identical optimizer/training behavior on every OS. Preserve newly produced results separately from the committed frozen summaries.

## Regenerate manuscript figures and Word

```bash
python -m pip install -e ".[manuscript]"
python scripts/build_persistent_event_figures.py
python scripts/build_persistent_event_manuscript.py --output local-data/Persistent_Event_Manuscript.docx
```

The figure builder needs Poppler (`pdftoppm`) on PATH for PDF-to-PNG conversion. The Word builder needs Pandoc on PATH, or pass `--pandoc /path/to/pandoc`. It uses native Word Office Math equations, editable text/tables, and a restrained manuscript style. Source figures and their data remain editable independently of the Word image objects. Manuscript formatting does not start data downloading or scientific experiments.

For a preview that preserves all committed figures and the delivered Word file:

```bash
python scripts/build_persistent_event_figures.py --output local-data/manuscript-preview/figures
python scripts/build_persistent_event_manuscript.py --resource-root local-data/manuscript-preview --output local-data/manuscript-preview/WORD_MANUSCRIPT.docx
```

The **Current Word manuscript** CI job uses the same temporary-output route, checks `figure_data.json` against the frozen copy, and makes the rebuilt Word document available as a workflow artifact. It does not invoke the archived LaTeX papers or neural forecasting smoke test unless those paths also change.

## Scope of claims

The final natural cover-LP results match exact trajectory endpoints for every examined frozen objective. This is not a claim that the cover polytope is always integral. Panels overlap through label support and repeated model comparisons; counts are descriptive, not independent-trial confidence estimates. The code, proofs, and raw evidence roles are kept distinct. Source history contains earlier labels such as “higher-order required”; the current manuscript interprets the Changping case as insufficiency of unary implications resolved by the complete three-label cover family.
