"""Exercise current-code evaluation and practical output-directory handling."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from urbanev_forecast import __main__ as cli


@pytest.fixture
def trained(tmp_path, monkeypatch):
    data = tmp_path / "rates.csv"
    values = np.full((720, 275), 0.4, dtype=np.float32)
    pd.DataFrame(values, index=pd.date_range("2022-09-01", periods=720, freq="h"),
                 columns=[str(i) for i in range(275)]).to_csv(data, index_label="date")
    output = tmp_path / "train"
    output.mkdir()
    (output / "notes.txt").write_text("Existing notes", encoding="utf-8")
    monkeypatch.setattr(cli, "code_fingerprint", lambda: "training-code")
    monkeypatch.setattr(sys, "argv", ["forecast", "train", "--csv", str(data), "--epochs", "1",
                                      "--batch-size", "128", "--output", str(output)])
    cli.main()
    return data, output


def test_existing_notes_and_code_edits_do_not_block_checkpoint_evaluation(trained, tmp_path, monkeypatch):
    data, training = trained
    output = tmp_path / "evaluation"
    output.mkdir()
    monkeypatch.setattr(cli, "code_fingerprint", lambda: "edited-development-code")
    monkeypatch.setattr(sys, "argv", ["forecast", "test", "--csv", str(data), "--checkpoint",
                                      str(training / "checkpoint.pt"), "--output", str(output)])
    cli.main()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["evaluation_code_matches_training"] is False
    assert result["checkpoint_code_sha256"] == "training-code"
    assert result["code_sha256"] == "edited-development-code"
    assert np.isfinite(result["raw"]["rmse"])
    assert (training / "notes.txt").read_text(encoding="utf-8") == "Existing notes"


def test_changed_training_data_is_still_rejected(trained, tmp_path, monkeypatch):
    data, training = trained
    frame = pd.read_csv(data, index_col=0)
    frame.iloc[0, 0] = 0.8
    frame.to_csv(data, index_label="date")
    monkeypatch.setattr(sys, "argv", ["forecast", "test", "--csv", str(data), "--checkpoint",
                                      str(training / "checkpoint.pt"), "--output", str(tmp_path / "evaluation")])
    with pytest.raises(ValueError, match="Training/validation data differs"):
        cli.main()


def test_existing_results_are_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    result = output / "result.json"
    result.write_text('{"saved": true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["forecast", "smoke", "--output", str(output)])
    with pytest.raises(SystemExit):
        cli.main()
    assert result.read_text(encoding="utf-8") == '{"saved": true}'
