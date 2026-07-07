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

import re
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
    is_underline: bool = False  # extension beyond the original docstring — see
                                 # parse_pdf() note on the bold+underline heading style


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

    JUDGMENT CALLS (not covered by the docstring above — flagging both):

    1. block_type="table": the docstring never says how a TextBlock becomes
       "table". PyMuPDF 1.24.3 (the pinned version) ships page.find_tables(),
       so each page is scanned for tables first; every detected table is
       collapsed into a single TextBlock (block_type="table", text = rows
       joined with " | " / "\\n") instead of one TextBlock per cell span.
       Any loose span whose center falls inside a detected table's bbox is
       skipped, so table content isn't double-counted as prose.

    2. is_underline: the real handbook uses bold+underlined text (no number
       prefix) as a secondary heading style (e.g. "Access Registration"),
       per the user's PDF formatting notes. Font "flags" in PyMuPDF do not
       carry an underline bit (underlines are drawn vector lines, not a font
       attribute), so this adds a real underline check: page.get_drawings()
       is scanned for near-horizontal line segments, and a span is flagged
       is_underline=True if a line sits just under its baseline and overlaps
       its x-range by >=50%. chunker.py's heading detection relies on this.
    """
    doc = fitz.open(pdf_path)
    blocks: list[TextBlock] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        page_height = page.rect.height

        # ── Table detection (judgment call #1 above) ──
        table_rects: list[fitz.Rect] = []
        for table in page.find_tables().tables:
            rect = fitz.Rect(table.bbox)
            table_rects.append(rect)
            rows = table.extract()
            table_text = "\n".join(" | ".join(cell or "" for cell in row) for row in rows)
            blocks.append(TextBlock(
                page_number=page_number,
                text=table_text,
                font_size=0.0,
                is_bold=False,
                bbox=tuple(rect),
                block_type="table",
            ))

        # ── Underline line segments on this page (judgment call #2 above) ──
        underline_segments: list[tuple[float, float, float]] = []  # (x0, x1, y)
        for drawing in page.get_drawings():
            for item in drawing["items"]:
                if item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1.0:  # near-horizontal
                    underline_segments.append((min(p1.x, p2.x), max(p1.x, p2.x), p1.y))

        def _has_underline(bbox: tuple) -> bool:
            x0, _, x1, y1 = bbox
            span_width = x1 - x0
            if span_width <= 0:
                return False
            for lx0, lx1, ly in underline_segments:
                # bbox y1 includes full font descent metrics even when the
                # span has no descender glyphs, so a real underline can sit
                # a few pt above the reported y1 as well as slightly below it.
                if not (y1 - 3.0 <= ly <= y1 + 4.0):
                    continue
                overlap = min(x1, lx1) - max(x0, lx0)
                if overlap / span_width >= 0.5:
                    return True
            return False

        # ── Text spans ──
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue  # image block
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        continue
                    bbox = span["bbox"]
                    if bbox[1] < 50 or bbox[1] > page_height - 50:
                        continue  # running header/footer

                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    if any(rect.contains(center) for rect in table_rects):
                        continue  # already captured as a table block above

                    is_bold = bool(span["flags"] & (1 << 4))
                    blocks.append(TextBlock(
                        page_number=page_number,
                        text=text,
                        font_size=span["size"],
                        is_bold=is_bold,
                        bbox=tuple(bbox),
                        block_type="text",
                        is_underline=_has_underline(bbox),
                    ))

    doc.close()
    blocks.sort(key=lambda b: (b.page_number, b.bbox[1], b.bbox[0]))
    return blocks


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

    JUDGMENT CALL: our real handbook has no Word bookmarks (confirmed via the
    synthetic test PDF, which also ships without embedded bookmarks), so
    Strategy B is the one that actually matters. The generic "two largest
    font sizes on the page" heuristic in the docstring misfires here because
    the title page (26pt), "Document Control" (16pt) and "CONTENTS" (18pt)
    front-matter headers are larger than the real level-1 chapter headings
    and would get mistaken for the ToC. Since the user's PDF formatting notes
    say the real document uses numbered headings ("1.", "1.1", "3.2.1"),
    Strategy B here first tries matching that numbering scheme directly
    (regex on block text) and only falls back to the pure font-size
    heuristic if no numbered headings are found at all.
    """
    doc = fitz.open(pdf_path)
    raw_toc = doc.get_toc()
    doc.close()  # done with the fitz handle either way — Windows locks the
                 # file otherwise, blocking any later re-write of the same path
    if raw_toc:
        return [TocEntry(level=level, title=title.strip(), start_page=page)
                for level, title, page in raw_toc]

    blocks = parse_pdf(pdf_path)
    entries: list[TocEntry] = []

    level1_re = re.compile(r"^\d+\.\s+\S")       # "1.  Introduction"
    level2_re = re.compile(r"^\d+\.\d+\s+\S")    # "3.1  Objective"
    # A numbered line ending in a dot-leader + page number is a ToC-page entry
    # ("1.  Introduction ..... 4"), not an actual heading — exclude it, or the
    # Contents page itself gets misread as a second copy of every heading.
    toc_leader_re = re.compile(r"\.{3,}\s*\d+$")

    for block in blocks:
        if block.block_type == "table":
            continue
        text = block.text.strip()
        if toc_leader_re.search(text):
            continue
        if level2_re.match(text):
            entries.append(TocEntry(level=2, title=text, start_page=block.page_number))
        elif level1_re.match(text):
            entries.append(TocEntry(level=1, title=text, start_page=block.page_number))

    if entries:
        return entries

    # Fallback: pure font-size heuristic (generic case, no numbering at all).
    font_sizes = sorted({b.font_size for b in blocks if b.block_type == "text"}, reverse=True)
    if len(font_sizes) < 2:
        return []
    level1_size, level2_size = font_sizes[0], font_sizes[1]

    for block in blocks:
        if block.block_type == "table":
            continue
        text = block.text.strip()
        if text.isdigit() or len(text) <= 3:
            continue
        if block.font_size >= level1_size:
            entries.append(TocEntry(level=1, title=text, start_page=block.page_number))
        elif block.font_size >= level2_size:
            entries.append(TocEntry(level=2, title=text, start_page=block.page_number))

    return entries


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
