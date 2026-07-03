"""
Day 1 — Person A
Hierarchical Chunker
====================
Converts the flat list[TextBlock] + list[TocEntry] from parser.py
into list[RawChunk] ready for DB insertion.

Chunking hierarchy (in priority order — never cross a chapter boundary):
    Chapter  →  Section  →  token-budget split

Token budget is enforced with tiktoken so it matches the LLM's context window.

Quick start (run directly to inspect chunks):
    python -m app.ingestion.chunker data/pdfs/hr_handbook.pdf
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken

from app.ingestion.parser import TextBlock, TocEntry, extract_toc, parse_pdf
from app.config import get_settings

settings = get_settings()
_tok = tiktoken.get_encoding("cl100k_base")


# ── Output dataclass (sync this with Person B's Chunk model on Day 1 EOD) ────

@dataclass
class RawChunk:
    chapter: str
    section_title: Optional[str]
    start_page: int
    end_page: int
    text: str
    chunk_type: str = "prose"   # "prose" | "table"


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(_tok.encode(text))


def assign_chapter(page: int, toc: list[TocEntry]) -> tuple[str, Optional[str]]:
    """
    Walk the ToC in reverse to find the most recent level-1 and level-2
    entries whose start_page <= page.

    Returns (chapter_title, section_title).
    section_title is None if no level-2 entry precedes this page.
    """
    chapter = "Unknown Chapter"
    section: Optional[str] = None

    for entry in reversed(toc):
        if entry.start_page > page:
            continue
        if entry.level == 2 and section is None:
            section = entry.title
        if entry.level == 1:
            chapter = entry.title
            break

    return chapter, section


def _is_heading(block: TextBlock, heading_font_size: float) -> bool:
    """
    Heuristic: a block is a section heading if its font_size >= heading_font_size
    and it's short enough to be a title (< 120 chars).
    Tune heading_font_size after inspecting real PDFs with parser.py.
    """
    return block.font_size >= heading_font_size and len(block.text.strip()) < 120


# ── Main chunking function ────────────────────────────────────────────────────

def chunk_document(blocks: list[TextBlock], toc: list[TocEntry]) -> list[RawChunk]:
    """
    Produce list[RawChunk] in document order.

    Implementation guide:

    Step 1 — Detect heading font size threshold.
        Collect all font sizes from blocks. The second-largest distinct size
        (largest is usually the document title) is a good threshold for
        section headings. Alternatively, use the level-2 ToC entry titles
        to find matching blocks and read their font_size directly.

    Step 2 — Group blocks into sections.
        Walk blocks in order. When you hit a heading block (per _is_heading),
        flush the current accumulator as a RawChunk (if non-empty) and start
        a new section with that heading's text as section_title.

    Step 3 — Token-budget split within a section.
        Keep a running accumulator of block texts. When adding a block would
        push count_tokens(accumulator) over CHUNK_MAX_TOKENS:
            a. Save the accumulator as a RawChunk.
            b. Seed the next accumulator with the last CHUNK_OVERLAP_TOKENS
               worth of text from the previous chunk (for boundary continuity).
        start_page = page of first block in accumulator.
        end_page   = page of last block in accumulator.

    Step 4 — Chapter boundary enforcement.
        Use assign_chapter(page, toc) on the first block of each accumulator.
        If the chapter changes mid-accumulator, flush immediately.
        Never let a chunk span two chapters.

    Step 5 — Table blocks.
        If block.block_type == "table", flush the current prose accumulator,
        emit the table as a single RawChunk(chunk_type="table") regardless
        of token count, then continue.

    Step 6 — Flush the final accumulator when blocks are exhausted.

    Returns list[RawChunk]. Filter out any chunks whose text.strip() is empty.
    """
    raise NotImplementedError


# ── Quick inspection script ───────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python -m app.ingestion.chunker data/pdfs/hr_handbook.pdf

    Prints chunk count, token distribution, and first 5 chunks so you can
    verify boundaries and page citations before moving to generation.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.chunker <path_to_pdf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    blocks = parse_pdf(path)
    toc = extract_toc(path)
    chunks = chunk_document(blocks, toc)

    token_counts = [count_tokens(c.text) for c in chunks]
    print(f"\nTotal chunks : {len(chunks)}")
    print(f"Token range  : {min(token_counts)} – {max(token_counts)}")
    print(f"Mean tokens  : {sum(token_counts) // len(token_counts)}")
    tables = sum(1 for c in chunks if c.chunk_type == "table")
    print(f"Table chunks : {tables}")

    print("\n=== First 5 chunks ===")
    for i, c in enumerate(chunks[:5]):
        print(f"\n[{i}] {c.chapter} / {c.section_title} | pp.{c.start_page}-{c.end_page} | {count_tokens(c.text)} tokens | {c.chunk_type}")
        print(c.text[:300])
        print("...")
