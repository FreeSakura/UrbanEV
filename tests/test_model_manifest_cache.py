"""Importing/training a model must not change its source manifest."""
import csv
import importlib.util
from pathlib import Path


def test_source_manifest_is_unchanged_after_bytecode_cache(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/"scripts/build_model_source_manifest.py"
    spec=importlib.util.spec_from_file_location("model_manifest_builder",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    models=tmp_path/"models";family=models/"example";family.mkdir(parents=True)
    (family/"model.py").write_text("VALUE = 1\n",encoding="utf-8")
    monkeypatch.setattr(module,"MODELS",models)
    module.main();before=(models/"MODEL_SOURCE_MANIFEST.csv").read_bytes()
    cache=family/"__pycache__";cache.mkdir()
    (cache/"model.cpython-test.pyc").write_bytes(b"synthetic bytecode cache")
    (family/"legacy.pyc").write_bytes(b"synthetic old-style cache")
    module.main()
    assert (models/"MODEL_SOURCE_MANIFEST.csv").read_bytes()==before
    rows=list(csv.DictReader(before.decode().splitlines()))
    assert [r["repository_path"] for r in rows]==["models/example/model.py"]
