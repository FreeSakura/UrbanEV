"""Keep routine documentation changes out of expensive CI work."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("ci_scope", Path(__file__).resolve().parents[1] / "scripts/ci_scope.py")
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


def test_documentation_and_generated_manifest_changes_stay_lightweight():
    assert scope.classify(["README.md", "docs/research/roadmap.md", "results/inventory.csv",
                           "artifacts/manifests/FULL_RELEASE_MANIFEST.json"]) == {
        "tests": False, "forecasting": False, "papers": False}


def test_forecasting_changes_run_tests_without_building_papers():
    assert scope.classify(["src/urbanev_forecast/models.py"]) == {
        "tests": True, "forecasting": True, "papers": False}


def test_paper_sources_build_papers_without_training():
    assert scope.classify(["paper/shared/results.tex"]) == {
        "tests": True, "forecasting": False, "papers": True}


def test_ci_or_dependency_changes_run_all_paths():
    assert all(scope.classify([".github/workflows/ci.yml"]).values())
    assert all(scope.classify(["pyproject.toml"]).values())


def test_maintenance_tool_changes_use_base_tests_only():
    assert scope.classify(["scripts/repository.py", "tests/test_ci_scope.py"]) == {
        "tests": True, "forecasting": False, "papers": False}
