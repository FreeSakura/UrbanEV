#!/usr/bin/env python3
"""Render every PDF page and create labeled 2x2 QA contact sheets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def render(pdf: Path, output: Path) -> list[Path]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise SystemExit("pdftoppm is required")
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "page"
    subprocess.run([pdftoppm, "-png", "-r", "110", str(pdf), str(prefix)], check=True)
    return sorted(output.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))


def contact_sheets(pages: list[Path], output: Path) -> list[Path]:
    sheets = []
    for offset in range(0, len(pages), 4):
        group = pages[offset : offset + 4]
        thumbs = []
        for index, page in enumerate(group, start=offset + 1):
            image = Image.open(page).convert("RGB")
            image.thumbnail((650, 920))
            canvas = Image.new("RGB", (670, 960), "white")
            canvas.paste(image, ((670 - image.width) // 2, 25))
            ImageDraw.Draw(canvas).text((12, 8), f"page {index}", fill="black")
            thumbs.append(canvas)
        sheet = Image.new("RGB", (1340, 1920), "#d9d9d9")
        for index, image in enumerate(thumbs):
            sheet.paste(image, ((index % 2) * 670, (index // 2) * 960))
        path = output / f"contact-{offset + 1:03d}-{offset + len(group):03d}.jpg"
        sheet.save(path, quality=88)
        sheets.append(path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pages = render(args.pdf, args.output / "pages")
    sheets = contact_sheets(pages, args.output)
    print(f"rendered {len(pages)} pages into {len(sheets)} contact sheets")


if __name__ == "__main__":
    main()
