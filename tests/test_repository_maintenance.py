"""Ensure documentation maintenance cannot silently distort published evidence."""
import csv
import importlib.util
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repository_maintenance", ROOT / "scripts/repository.py")
repository = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repository)


@pytest.fixture
def core_rows():
    path = ROOT / repository.CORE / "matched_core_scores.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main_rows(rows):
    return [r for r in rows if r["cohort"] == "NEW_MATCHED_CORE"
            and r["target_scope"] == "terminal_H" and r["postprocess"] == "raw"]


def test_public_result_aggregation_matches_frozen_summary():
    result = repository.development_table(ROOT)
    assert len(result) == 7
    assert sum(row["instances"] for row in result) == 17
    ridge = next(row for row in result if row["model"] == "RIDGE_OD")
    assert ridge["rmse"] == pytest.approx(0.11114679530313792, abs=1e-12)
    assert ridge["mae"] == pytest.approx(0.06849459515196628, abs=1e-12)


@pytest.fixture
def persistent_inputs():
    folder = ROOT / repository.PERSISTENT
    with (folder / "comparison_results.csv").open(newline="", encoding="utf-8-sig") as f:
        comparisons = list(csv.DictReader(f))
    with (folder / "panel_results.csv").open(newline="", encoding="utf-8-sig") as f:
        panels = list(csv.DictReader(f))
    return comparisons, panels, json.loads((folder / "STRUCTURE_VALIDATION.json").read_text())


def test_current_structure_table_matches_frozen_full_coverage(persistent_inputs):
    result = repository.aggregate_persistent_events(*persistent_inputs)
    assert sum(x["panels"] for x in result) == 470
    assert sum(x["comparisons"] for x in result) == 1410
    assert sum(x["endpoint_changes"] for x in result) == 110
    assert sum(x["direction_changes"] for x in result) == 1
    assert sum(x["cover_lp_gaps"] for x in result) == 0
    exact = [x for x in result if x["structure"] == "PAIRWISE_EXACT"]
    assert sum(x["panels"] for x in exact) == 323
    assert sum(x["comparisons"] for x in exact) == 969


def test_persistent_duplicate_cannot_inflate_headline(persistent_inputs):
    rows, panels, receipt = persistent_inputs
    with pytest.raises(ValueError, match="Duplicate"):
        repository.aggregate_persistent_events(rows + [rows[0]], panels, receipt)


def test_persistent_missing_pair_cannot_shrink_denominator(persistent_inputs):
    rows, panels, receipt = persistent_inputs
    with pytest.raises(ValueError, match="pair coverage"):
        repository.aggregate_persistent_events(rows[:-1], panels, receipt)


def test_persistent_results_must_match_complete_receipt(persistent_inputs):
    rows, panels, receipt = persistent_inputs
    receipt["comparisons"] -= 1
    with pytest.raises(ValueError, match="counts differ"):
        repository.aggregate_persistent_events(rows, panels, receipt)


def test_persistent_nonfinite_risks_are_rejected(persistent_inputs):
    rows, panels, receipt = persistent_inputs
    rows[0]["cover_LP_upper"] = "nan"
    with pytest.raises(ValueError, match="Non-finite"):
        repository.aggregate_persistent_events(rows, panels, receipt)


def test_catalog_reads_manuscript_title_and_word_metadata(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text("---\ntitle: Feasible Label Images\n---\n\n# Abstract\nBody", encoding="utf-8")
    assert repository.title(source) == "Feasible Label Images"
    word = ROOT / "paper/persistent_events/WORD_MANUSCRIPT.docx"
    assert repository.title(word) == "Persistent Event Evaluation under Partial Observations"
    assert "persistent_events/WORD_MANUSCRIPT.docx" in repository.generate(ROOT)["docs/catalog.md"]


def test_partial_horizons_cannot_enter_macro(core_rows):
    rows = main_rows(core_rows)
    rows.pop(0)
    with pytest.raises(ValueError, match="horizon coverage"):
        repository.aggregate_core(rows)


def test_duplicate_score_cannot_gain_extra_weight(core_rows):
    rows = main_rows(core_rows)
    with pytest.raises(ValueError, match="Duplicate"):
        repository.aggregate_core(rows + [rows[0]])


def test_missing_seed_cannot_improve_result_by_omission(core_rows):
    rows = [r for r in main_rows(core_rows) if not (r["model"] == "OD_PRODUCT" and r["seed"] == "20260916")]
    with pytest.raises(ValueError, match="seed coverage"):
        repository.aggregate_core(rows)


@pytest.mark.parametrize("field,value,reason", [
    ("cohort", "OLD_EXPERIMENT", "cohort"),
    ("information_track", "GLOBAL_O_168", "information track"),
    ("scored_values", "100", "support"),
    ("rmse", "nan", "Non-finite"),
    ("mae", "-0.1", "negative"),
])
def test_incompatible_or_invalid_scores_are_rejected(core_rows, field, value, reason):
    rows = main_rows(core_rows)
    rows[0][field] = value
    with pytest.raises(ValueError, match=reason):
        repository.aggregate_core(rows)


def test_link_parser_handles_images_references_and_code():
    text = '''[page](docs/a(b).md#section)
![image](<docs/image name.png>)
[guide][ref]
[ref]: docs/guide.md
`[literal](missing-inline.md)`
```md
[example](missing-fenced.md)
```
'''
    assert repository.markdown_links(text) == ["docs/a(b).md#section", "docs/image name.png", "docs/guide.md"]


def test_broken_links_and_unicode_headings(tmp_path):
    (tmp_path / "docs").mkdir()
    page = tmp_path / "docs/guide.md"
    page.write_text("# 中文标题\n\n## 用法\n\n## 用法\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("[ok](docs/guide.md#用法-1)\n[bad](docs/guide.md#不存在)\n[missing](absent.md)\n", encoding="utf-8")
    errors = repository.link_errors(tmp_path)
    assert len(errors) == 2
    assert any("missing heading" in e for e in errors)
    assert any("missing/unsafe link" in e for e in errors)


def test_all_studies_and_generated_documents_are_current():
    assert repository.check(ROOT) == []


def test_report_builder_resolves_each_relocated_source():
    spec = importlib.util.spec_from_file_location("v3_report", ROOT / "scripts/research/build_report.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    sources = [ROOT / "docs/reports/audit/PAIRED_AUDIT_V3_REPORT.md", ROOT / "docs/theory/DERIVATION_V3.md"]
    prefix = "https://github.com/FreeSakura/UrbanEV/blob/main/"
    for source in sources:
        rendered = renderer.github_links(source.read_text(encoding="utf-8"), source)
        for url in repository.markdown_links(rendered):
            if url.startswith(prefix):
                assert (ROOT / url[len(prefix):].split("#")[0]).is_file(), url


def manifest_fixture(root):
    folder = root / "artifacts/manifests"
    folder.mkdir(parents=True)
    (root / "README.md").write_bytes(b"# Example\r\n")
    record = {"path": "README.md", "bytes": len(b"# Example\n"),
              "sha256": hashlib.sha256(b"# Example\n").hexdigest()}
    (folder / "FULL_RELEASE_MANIFEST.json").write_text(json.dumps({
        "schema_version": "urbanev-full-release-manifest/v1", "tracked_files": [record]}), encoding="utf-8")


def test_manifest_check_needs_no_external_release_and_normalizes_text(tmp_path):
    manifest_fixture(tmp_path)
    assert repository.manifest_errors(tmp_path) == []


def test_manifest_check_detects_changed_and_missing_content(tmp_path):
    manifest_fixture(tmp_path)
    (tmp_path / "README.md").write_text("Changed", encoding="utf-8")
    assert any("content mismatch" in error for error in repository.manifest_errors(tmp_path))
    (tmp_path / "README.md").unlink()
    assert any("missing" in error for error in repository.manifest_errors(tmp_path))


def test_mixed_case_document_order_is_identical_across_platforms():
    names = ["SIGNAL_PLAN.md", "next_protocol.md", "CONTINUOUS_PLAN.md", "README.md"]
    windows = repository.sorted_paths(PureWindowsPath(name) for name in names)
    posix = repository.sorted_paths(PurePosixPath(name) for name in names)
    assert [path.as_posix() for path in windows] == [path.as_posix() for path in posix] == sorted(names)
