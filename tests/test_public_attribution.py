"""Keep project-owned attribution pseudonymous without storing removed PII."""
from pathlib import Path
import io
import tomli
import yaml
from pypdf import PdfReader, PdfWriter
from urbanev_audit.privacy import audit_pdf_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_project_author_records_are_pseudonymous():
    for name in ("AUTHORS.yml", "CITATION.cff"):
        records = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))["authors"]
        assert len(records) == 1
        assert records[0]["name"] == "FreeSakura"
        assert not set(records[0]) & {"email", "orcid", "affiliation", "given-names", "family-names"}
    project = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["authors"] == [{"name": "FreeSakura"}]
    macro = (ROOT / "paper/shared/author_metadata.tex").read_text(encoding="utf-8")
    assert "FreeSakura" in macro
    assert not any(token in macro.casefold() for token in ("mailto:", "orcid", "university", "@"))
    for name in ("main/UrbanEV_Evidence_Audit_Main.pdf", "supplement/UrbanEV_Evidence_Audit_Supplement.pdf", "archive/UrbanEV_Evidence_Audit_Historical_Archive_v0.9.0.pdf"):
        reader = PdfReader(ROOT / "paper" / name)
        assert reader.metadata.author == "FreeSakura"


def test_pdf_metadata_is_scanned_for_private_paths():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Author": "C:" + chr(92) + "synthetic-person" + chr(92) + "private"})
    buffer = io.BytesIO()
    writer.write(buffer)
    assert any("metadata" in finding and "windows_absolute_path" in finding for finding in audit_pdf_bytes(buffer.getvalue(), "synthetic.pdf"))
