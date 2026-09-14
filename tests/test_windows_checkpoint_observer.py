"""Regression evidence for checkpoint-observer sharing, without model/data access."""
import os

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
def test_open_resume_reader_can_block_atomic_metadata_replace(tmp_path):
    target = tmp_path / "resume.json"
    replacement = tmp_path / "resume.json.tmp"
    target.write_text('{"epoch":18}', encoding="utf-8")
    replacement.write_text('{"epoch":19}', encoding="utf-8")
    with target.open("r", encoding="utf-8"):
        with pytest.raises(PermissionError):
            os.replace(replacement, target)
    # Nothing is lost: replacement succeeds once the incompatible reader closes.
    os.replace(replacement, target)
    assert target.read_text(encoding="utf-8") == '{"epoch":19}'


def test_observing_separate_append_log_does_not_lock_checkpoint_target(tmp_path):
    log = tmp_path / "train.log"
    target = tmp_path / "resume.json"
    replacement = tmp_path / "resume.json.tmp"
    log.write_text('{"event":"checkpoint","epoch":18}\n', encoding="utf-8")
    target.write_text('{"epoch":18}', encoding="utf-8")
    replacement.write_text('{"epoch":19}', encoding="utf-8")
    with log.open("r", encoding="utf-8") as observer:
        assert '"checkpoint"' in observer.read()
        os.replace(replacement, target)
    assert target.read_text(encoding="utf-8") == '{"epoch":19}'
