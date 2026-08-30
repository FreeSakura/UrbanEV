import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "artifacts/summaries" / name).read_text(encoding="utf-8"))["headline"]


def test_urbanev_cleanroom_headlines_match_paper_precision():
    headline = load("URBANEV_CLEANROOM_RECOMPUTE.json")
    fusion = headline["urbanev_router_and_fixed_fusion"]
    assert fusion["global_fixed"] == pytest.approx(0.07136675688300531)
    assert fusion["global_fixed_gain_vs_timexer_pct"] == pytest.approx(3.1775965191758897)
    assert headline["urbanev_chronos2"]["macro_rmse"] == pytest.approx(0.07354229522535828)
    distillation = headline["urbanev_residual_distillation"]
    assert distillation["aligned_gain_vs_GT_only_pct"] == pytest.approx(0.11893143253937777)
    assert distillation["aligned_gain_vs_shuffled_pct"] == pytest.approx(0.2585256259249757)


def test_paris_cleanroom_headlines_match_paper_precision():
    teacher = load("PARIS_DEVELOPMENT_CLEANROOM_RECOMPUTE.json")["paris_development_teacher"]
    assert teacher["teacher"] == pytest.approx(0.2404877475991446)
    assert teacher["teacher_gain_vs_best_single_pct"] == pytest.approx(0.8468586786397444)
