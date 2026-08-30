#!/usr/bin/env python3
"""Verify rebuilt PDFs against normalized text, page, font, and metadata gates."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/manifests/PAPER_BUILD_MANIFEST.json"


def normalized_text_hash(reader: PdfReader) -> str:
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(character for character in text if character.isalnum())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    return all(
        len(row.split()) >= 6 and row.split()[5].lower() == "yes"
        for row in output.splitlines()[2:]
        if row.strip()
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["pdfs"]:
        path = ROOT / record["path"]
        reader = PdfReader(str(path))
        if len(reader.pages) != record["pages"]:
            raise ValueError(f"page-count drift: {record['path']}")
        if normalized_text_hash(reader) != record["normalized_text_sha256"]:
            raise ValueError(f"normalized-text drift: {record['path']}")
        if not fonts_embedded(path, reader):
            raise ValueError(f"unembedded font: {record['path']}")
        metadata = reader.metadata or {}
        for key in ("/Title", "/Author"):
            if key in record["metadata"] and str(metadata.get(key)) != record["metadata"][key]:
                raise ValueError(f"PDF metadata drift ({key}): {record['path']}")
    print(f"paper manifest PASS: {len(manifest['pdfs'])} PDFs")


if __name__ == "__main__":
    main()
