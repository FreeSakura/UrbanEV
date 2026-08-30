import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas

from urbanev_audit.privacy import (
    audit_npz_bytes,
    audit_pdf_bytes,
    audit_git_history,
    audit_root,
    audit_text_bytes,
    audit_zip_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


def _windows_path() -> str:
    backslash = chr(92)
    return "D:" + backslash + "private" + backslash + "artifact.json"


def test_detects_json_escaped_windows_path():
    data = json.dumps({"path": _windows_path()}).encode()
    findings = audit_text_bytes(data, "fixture.json", ".json")
    assert any("windows_absolute_path" in item or "json_escaped" in item for item in findings)


def test_detects_unc_path():
    backslash = chr(92)
    text = backslash * 2 + "server" + backslash + "share" + backslash + "file"
    assert any("unc_path" in item for item in audit_text_bytes(text.encode(), "fixture.txt", ".txt"))


def test_detects_linux_home_path():
    text = "/".join(("", "home", "person", "private", "file"))
    assert any("linux_home_path" in item for item in audit_text_bytes(text.encode(), "fixture.txt", ".txt"))


def test_detects_generic_user_directory():
    text = "/".join(("", "Users", "person", "private", "file"))
    assert any("user_directory" in item for item in audit_text_bytes(text.encode(), "fixture.txt", ".txt"))


def test_scans_pdf_text(tmp_path: Path):
    path = tmp_path / "fixture.pdf"
    doc = canvas.Canvas(str(path))
    doc.drawString(72, 720, _windows_path())
    doc.save()
    assert any("windows_absolute_path" in item for item in audit_pdf_bytes(path.read_bytes(), "fixture.pdf"))


def test_scans_zip_text_members():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", json.dumps({"path": _windows_path()}))
    assert audit_zip_bytes(buffer.getvalue(), "release.zip")


def test_scans_npz_string_arrays():
    buffer = io.BytesIO()
    np.savez(buffer, safe_metadata=np.asarray([_windows_path()]))
    assert audit_npz_bytes(buffer.getvalue(), "package.npz")


def test_scans_release_assets():
    buffer = io.BytesIO()
    package = io.BytesIO()
    np.savez(package, safe_metadata=np.asarray([_windows_path()]))
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("package.npz", package.getvalue())
    assert audit_zip_bytes(buffer.getvalue(), "release.zip")


def test_scans_git_history_or_exported_tree(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    fixture = tmp_path / "metadata.json"
    fixture.write_text(json.dumps({"path": _windows_path()}), encoding="utf-8")
    subprocess.run(["git", "add", "metadata.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    fixture.unlink()
    subprocess.run(["git", "commit", "-qam", "remove fixture"], cwd=tmp_path, check=True)
    assert any("windows_absolute_path" in item for item in audit_git_history(tmp_path))


def test_repository_privacy_gate():
    assert audit_root(ROOT) == []
