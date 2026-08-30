from pathlib import Path

from urbanev_audit.privacy import audit_root

ROOT = Path(__file__).resolve().parents[1]


def test_repository_privacy_gate():
    assert audit_root(ROOT) == []
