#!/usr/bin/env python3
"""Build one public paper variant and fail on unresolved references or overflow."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SOURCE_DATE_EPOCH", "1788134400")
VARIANTS = {
    "main": (ROOT / "paper/main", "main.tex", "UrbanEV_Evidence_Audit_Main.pdf"),
    "supplement": (ROOT / "paper/supplement", "supplement.tex", "UrbanEV_Evidence_Audit_Supplement.pdf"),
    "archive": (ROOT / "paper/archive", "main_archive.tex", "UrbanEV_Evidence_Audit_Archive_Rebuilt.pdf"),
}


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build(variant: str) -> Path:
    directory, source, output_name = VARIANTS[variant]
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    stem = Path(source).stem
    for suffix in ("aux", "bbl", "blg", "fdb_latexmk", "fls", "log", "out", "toc", "pdf"):
        candidate = directory / f"{stem}.{suffix}"
        if candidate.is_file():
            candidate.unlink()
    prefer_direct = platform.system() == "Windows" or os.environ.get("URBANEV_DIRECT_LATEX") == "1"
    if latexmk and not prefer_direct:
        run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", source], directory)
    elif pdflatex and bibtex:
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", source], directory)
        run([bibtex, stem], directory)
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", source], directory)
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", source], directory)
    else:
        raise SystemExit("A LaTeX installation with latexmk or pdflatex+bibtex is required")
    log = (directory / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    blockers = ("undefined references", "Citation `", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox")
    hits = [item for item in blockers if item in log]
    if hits:
        raise SystemExit(f"paper gate failed for {variant}: {hits}")
    built = directory / f"{stem}.pdf"
    final = directory / output_name
    shutil.copy2(built, final)
    print(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    build(args.variant)


if __name__ == "__main__":
    main()
