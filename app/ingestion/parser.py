"""
Day 1 — Person A
PDF Parser + ToC Extractor
==========================
Produces two things:
  - list[TextBlock]  : flat, ordered list of every text span in the PDF
  - list[TocEntry]   : chapter/section hierarchy with start pages

Both are consumed by chunker.py. Keep this file focused on extraction only —
no chunking logic belongs here.

Quick start (run this file directly to inspect your PDF):
    python -m app.ingestion.parser data/pdfs/hr_handbook.pdf
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ── Data classes (Person A defines, Person B mirrors in models.py) ────────────

@dataclass
class TextBlock:
    page_number: int      # 1-indexed (matches what humans see in the PDF)
    text: str
    font_size: float
    is_bold: bool
    bbox: tuple           # (x0, y0, x1, y1)
    block_type: str = "text"   # "text" | "table"


@dataclass
class TocEntry:
    level: int            # 1 = chapter, 2 = section, 3 = subsection
    title: str
    start_page: int       # 1-indexed


# ── PDF text extraction ───────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> list[TextBlock]:
    """
    Extract all text spans from the PDF in reading order.

    Implementation guide:
    1.  doc = fitz.open(pdf_path)
    2.  For each page (0-indexed internally, convert to 1-indexed for page_number):
        a.  page.get_text("dict") returns a dict with a "blocks" list.
        b.  Each block has a "lines" list; each line has a "spans" list.
        c.  Each span has: "text", "size" (font_size), "flags" (bit 4 = bold),
            "bbox" (x0, y0, x1, y1).
        d.  Skip spans whose text is whitespace-only.
        e.  Filter out running headers/footers: spans where
            bbox[1] < 50  (top 50px of page)  OR
            bbox[1] > page.rect.height - 50  (bottom 50px).
            Tune these thresholds after inspecting the real PDFs.
    3.  Return flat list sorted by (page_number, bbox[1], bbox[0])
        so text flows top-to-bottom, left-to-right.

    NOTE: Do NOT merge spans into paragraphs here — chunker.py does that.
    """
    raise NotImplementedError


# ── Table of Contents extraction ──────────────────────────────────────────────

def extract_toc(pdf_path: Path) -> list[TocEntry]:
    """
    Extract the document's chapter/section hierarchy.

    Strategy A — PDF bookmarks (try this first):
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()   # returns [[level, title, page], ...]
        If non-empty, convert each row to TocEntry and return.
        Most Word-exported PDFs have bookmarks — check with fitz first.

    Strategy B — Font-size heuristic (fallback if A returns empty):
        Scan TextBlocks for spans whose font_size is the largest seen on that
        page AND the span text looks like a heading (not all-digits, len > 3).
        Use the two largest distinct font sizes as level-1 and level-2 headings.
        Build TocEntry list from these detections.

    Run Strategy A first on both company PDFs on Day 1 morning.
    If doc.get_toc() returns a non-empty list, Strategy B is unnecessary.
    """
    raise NotImplementedError


# ── Quick inspection script ───────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python -m app.ingestion.parser data/pdfs/hr_handbook.pdf

    Prints the first 30 TextBlocks and the full ToC so you can verify
    font sizes, page numbers, and heading detection before building chunker.py.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.parser <path_to_pdf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    print(f"\n=== ToC ({path.name}) ===")
    toc = extract_toc(path)
    for e in toc:
        indent = "  " * (e.level - 1)
        print(f"{indent}[p{e.start_page}] {e.title}")

    print(f"\n=== First 30 TextBlocks ===")
    blocks = parse_pdf(path)
    for b in blocks[:30]:
        bold = "B" if b.is_bold else " "
        print(f"p{b.page_number:3d} | {b.font_size:5.1f}pt {bold} | {b.text[:80]!r}")
