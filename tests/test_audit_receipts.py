import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_citation_audit_receipt_is_complete():
    receipt = json.loads((ROOT / "artifacts/summaries/paper_audits/CITATION_AUDIT.json").read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["counts"] == {
        "actually_cited_entries": 26,
        "citation_occurrences": 58,
        "KEEP": 26,
        "FIX": 0,
        "REPLACE": 0,
        "REMOVE": 0,
    }


def test_claim_audit_has_no_numeric_failure():
    receipt = json.loads((ROOT / "artifacts/summaries/paper_audits/PAPER_CLAIM_AUDIT.json").read_text(encoding="utf-8"))
    assert receipt["counts"]["FAIL"] == 0
    assert receipt["body_numeric_mismatch_found"] is False
