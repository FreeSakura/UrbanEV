#!/usr/bin/env python3
"""Add a non-normative cover to the immutable 42-page historical archive."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/archive/UrbanEV_Evidence_Audit_Archive.pdf"
OUTPUT = ROOT / "paper/archive/UrbanEV_Evidence_Audit_Historical_Archive_v0.9.0.pdf"
TEMP = ROOT / "tmp/pdfs/historical_archive_cover.pdf"


def main() -> None:
    source = PdfReader(str(SOURCE))
    if len(source.pages) != 42:
        raise ValueError(f"expected the immutable archive to have 42 pages, found {len(source.pages)}")
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    font_root = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("ArchiveSans", font_root / "Vera.ttf"))
    pdfmetrics.registerFont(TTFont("ArchiveSansBold", font_root / "VeraBd.ttf"))
    page = canvas.Canvas(str(TEMP), pagesize=A4, invariant=1)
    width, height = A4
    page.setTitle("Historical archive notice")
    page.setAuthor("Hongwei Chi")
    page.setFont("ArchiveSansBold", 20)
    page.drawCentredString(width / 2, height - 170, "HISTORICAL ARCHIVE")
    page.setFont("ArchiveSansBold", 14)
    page.drawCentredString(width / 2, height - 210, "NOT THE CURRENT SUBMISSION VERSION")
    page.setFont("ArchiveSans", 11)
    lines = [
        "This cover precedes the preserved 42-page v0.9.0 archive.",
        "The current normative preprint consists of Main v0.9.1",
        "and Supplementary Material v0.9.1.",
        "The original archive is retained for audit history only.",
    ]
    for index, line in enumerate(lines):
        page.drawCentredString(width / 2, height - 280 - index * 24, line)
    page.setStrokeColorRGB(0.2, 0.2, 0.2)
    page.rect(60, 90, width - 120, height - 180, stroke=1, fill=0)
    page.showPage()
    page.save()

    writer = PdfWriter()
    cover = PdfReader(str(TEMP))
    cover_page = cover.pages[0]
    content = cover_page.get_contents().get_data().replace(b"BT /F1 12 Tf 14.4 TL ET\n", b"")
    stream = DecodedStreamObject()
    stream.set_data(content)
    cover_page[NameObject("/Contents")] = stream
    cover_page["/Resources"]["/Font"].pop(NameObject("/F1"), None)
    writer.add_page(cover_page)
    writer.append(str(SOURCE))
    writer.add_metadata(
        {
            "/Title": "UrbanEV Evidence Audit - Historical Archive v0.9.0",
            "/Author": "Hongwei Chi",
            "/Subject": "Historical, non-normative archive",
            "/Keywords": "UrbanEV, evidence audit, historical archive",
        }
    )
    with OUTPUT.open("wb") as handle:
        writer.write(handle)
    reopened = PdfReader(str(OUTPUT))
    if len(reopened.pages) != 43:
        raise ValueError("historical wrapper must contain one cover plus 42 archived pages")
    print(OUTPUT)


if __name__ == "__main__":
    main()
