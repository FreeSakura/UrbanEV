"""Keep routine documentation changes out of expensive CI work."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("ci_scope", Path(__file__).resolve().parents[1] / "scripts/ci_scope.py")
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


def test_documentation_and_generated_manifest_changes_stay_lightweight():
    assert scope.classify(["README.md", "docs/research/roadmap.md", "results/inventory.csv",
                           "artifacts/manifests/FULL_RELEASE_MANIFEST.json"]) == {
        "tests": False, "forecasting": False, "papers": False, "manuscript": False}


def test_forecasting_changes_run_tests_without_building_papers():
    assert scope.classify(["src/urbanev_forecast/models.py"]) == {
        "tests": True, "forecasting": True, "papers": False, "manuscript": False}


def test_paper_sources_build_papers_without_training():
    assert scope.classify(["paper/shared/results.tex"]) == {
        "tests": True, "forecasting": False, "papers": True, "manuscript": False}


def test_ci_or_dependency_changes_run_all_paths():
    assert all(scope.classify([".github/workflows/ci.yml"]).values())
    assert all(scope.classify(["pyproject.toml"]).values())


def test_maintenance_tool_changes_use_base_tests_only():
    assert scope.classify(["scripts/repository.py", "tests/test_ci_scope.py"]) == {
        "tests": True, "forecasting": False, "papers": False, "manuscript": False}


def test_current_event_scripts_configs_and_tests_do_not_trigger_neural_training():
    for path in ["scripts/research/run_shared_missing_ap2.py",
                 "scripts/research/validate_persistent_event_covers.py",
                 "configs/research/PERSISTENT_EVENT_STRUCTURE_20260917.json",
                 "configs/research/SHARED_MISSING_EVENTS_AP1_20260916.json",
                 "tests/test_event_cover_geometry.py", "src/urbanev_audit/event_witnesses.py"]:
        assert scope.classify([path]) == {
            "tests": True, "forecasting": False, "papers": False, "manuscript": False}


def test_word_sources_and_evidence_select_the_current_manuscript():
    for path in ["paper/persistent_events/manuscript.md", "paper/persistent_events/WORD_MANUSCRIPT.docx"]:
        assert scope.classify([path]) == {
            "tests": False, "forecasting": False, "papers": False, "manuscript": True}
    assert scope.classify(["artifacts/summaries/persistent_event_complete_covers_20260917/comparison_results.csv"]) == {
        "tests": True, "forecasting": False, "papers": False, "manuscript": True}
    assert scope.classify(["scripts/build_persistent_event_figures.py"]) == {
        "tests": True, "forecasting": False, "papers": False, "manuscript": True}


def test_mixed_changes_preserve_all_required_routes():
    assert scope.classify(["paper/persistent_events/manuscript.md", "paper/shared/results.tex",
                           "src/urbanev_forecast/models.py"]) == dict.fromkeys(scope.SCOPES, True)
    assert scope.classify(["scripts/research/run_full_benchmark.py"])["forecasting"]
