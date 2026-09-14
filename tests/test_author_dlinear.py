from pathlib import Path
import pytest

pytest.importorskip("torch")
from urbanev_forecast.author_dlinear import DLinearAdapter, load_author_model


def test_unverified_source_fails_before_execution(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    marker = tmp_path / "executed"
    (model_dir / "DLinear.py").write_text(f"open({str(marker)!r}, 'w').write('bad')")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_author_model(tmp_path)
    assert not marker.exists()


@pytest.mark.parametrize("field,bad", [("history", True), ("horizon", 0), ("channels", 2.5)])
def test_invalid_configuration_fails_before_import(field, bad):
    kwargs = dict(history=168, horizon=3, channels=275)
    kwargs[field] = bad
    with pytest.raises(ValueError, match="positive integer"):
        DLinearAdapter(Path("nonexistent"), **kwargs)
