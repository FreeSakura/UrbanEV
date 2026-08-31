#!/usr/bin/env python3
"""Record deterministic text, metadata, page, font, and source hashes for paper PDFs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDFS = (
    ROOT / "paper/main/UrbanEV_Evidence_Audit_Main.pdf",
    ROOT / "paper/supplement/UrbanEV_Evidence_Audit_Supplement.pdf",
    ROOT / "paper/archive/UrbanEV_Evidence_Audit_Archive.pdf",
    ROOT / "paper/archive/UrbanEV_Evidence_Audit_Archive_Rebuilt.pdf",
    ROOT / "paper/archive/UrbanEV_Evidence_Audit_Historical_Archive_v0.9.0.pdf",
)
SOURCE_SUFFIXES = {".tex", ".bib", ".py", ".csv", ".json"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_pdf_text(reader: PdfReader) -> str:
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def canonical_character_multiset(text: str) -> str:
    """Ignore float/page order while retaining every extracted alphanumeric character."""
    return "".join(sorted(text))


def _pypdf_fonts_embedded(reader: PdfReader) -> bool:
    for page in reader.pages:
        fonts = ((page.get("/Resources") or {}).get("/Font") or {})
        for reference in fonts.values():
            font = reference.get_object()
            if str(font.get("/Subtype", "")) == "/Type3":
                continue
            descriptor = font.get("/FontDescriptor")
            if descriptor is None and font.get("/DescendantFonts"):
                descriptor = font["/DescendantFonts"][0].get_object().get("/FontDescriptor")
            if descriptor is None:
                return False
            descriptor = descriptor.get_object()
            if not any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                return False
    return True


def fonts_embedded(path: Path, reader: PdfReader) -> bool:
    executable = shutil.which("pdffonts")
    if not executable:
        return _pypdf_fonts_embedded(reader)
    output = subprocess.run([executable, str(path)], check=True, capture_output=True, text=True).stdout
    rows = output.splitlines()[2:]
    return all(len(row.split()) >= 6 and row.split()[5].lower() == "yes" for row in rows if row.strip())


def main() -> None:
    pdf_records = []
    for path in PDFS:
        if not path.is_file():
            raise FileNotFoundError(path)
        reader = PdfReader(str(path))
        text = canonical_pdf_text(reader)
        metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items()}
        pdf_records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
                "pages": len(reader.pages),
                "normalized_text_sha256": sha256(text.encode("utf-8")),
                "normalized_character_multiset_sha256": sha256(
                    canonical_character_multiset(text).encode("utf-8")
                ),
                "fonts_embedded": fonts_embedded(path, reader),
                "metadata": metadata,
            }
        )
    source_paths = sorted(
        (
            path for path in (ROOT / "paper").rglob("*")
            if path.is_file()
            and path.suffix.lower() in SOURCE_SUFFIXES
            and "editable" not in path.relative_to(ROOT).parts
            and "build" not in path.relative_to(ROOT).parts
        ),
        key=lambda path: path.relative_to(ROOT).as_posix().casefold(),
    )
    sources = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path.read_bytes().replace(b"\r\n", b"\n")),
            "bytes": len(path.read_bytes().replace(b"\r\n", b"\n")),
        }
        for path in source_paths
    ]
    payload = {
        "schema_version": "urbanev-paper-build-manifest/v2",
        "source_date_epoch": 1788134400,
        "paper_version": "post-v0.9.1-language-draft",
        "pdfs": pdf_records,
        "sources": sources,
    }
    output = ROOT / "artifacts/manifests/PAPER_BUILD_MANIFEST.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
