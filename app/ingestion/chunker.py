"""
Hierarchical Chunker
=====================
Converts the flat list[TextBlock] + list[TocEntry] from parser.py
into list[RawChunk] ready for DB insertion.

Chunking hierarchy (in priority order — never cross a chapter boundary):
    Chapter  →  Section  →  token-budget split

Token budget is enforced with tiktoken so it matches the LLM's context window.

Quick start (run directly to inspect chunks):
    python -m app.ingestion.chunker data/pdfs/im_policy_test.pdf
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


# ── Output dataclass (mirrors Chunk in models.py) ─────────────────────────────

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


def _tail_by_tokens(text: str, n_tokens: int) -> str:
    """Return the last n_tokens worth of text, decoded back to a string."""
    if n_tokens <= 0 or not text:
        return ""
    tokens = _tok.encode(text)
    return _tok.decode(tokens[-n_tokens:])


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
    """
    return block.font_size >= heading_font_size and len(block.text.strip()) < 120


# ── Main chunking function ────────────────────────────────────────────────────

def chunk_document(blocks: list[TextBlock], toc: list[TocEntry]) -> list[RawChunk]:
    """
    Produce list[RawChunk] in document order.

    Groups blocks into sections using a detected heading font-size threshold
    (derived by matching blocks against ToC entry titles, since a generic
    "second-largest font size" heuristic would pick up title-page and
    ToC-page headers, which are larger than real body headings). Within a
    section, text accumulates until it would exceed settings.chunk_max_tokens,
    at which point the chunk is flushed and the next one is seeded with the
    last settings.chunk_overlap_tokens of text for boundary continuity. A
    chunk never spans two chapters. Table blocks are flushed and emitted
    immediately as their own chunk (chunk_type="table"), regardless of token
    count. Returns list[RawChunk] with all empty-text chunks filtered out.

    JUDGMENT CALLS:

    1. Front-matter exclusion: any block on a page before the first ToC
       entry's start_page is dropped before chunking begins — the title,
       document control, and ToC pages aren't teachable content, and the
       ToC's dotted-leader lines would otherwise get chunked as prose.

    2. Bold+underline sub-headings: the handbook also uses bold+underlined
       text with no number prefix (e.g. "Access Registration") as a
       section-boundary marker, alongside numbered headings. These aren't in
       the ToC and won't clear the font-size threshold used for numbered
       headings, so they're checked separately via block.is_bold and
       block.is_underline (see parser.py) and treated the same as a
       font-size heading: flush + start a new section.
    """
    if toc:
        min_content_page = min(e.start_page for e in toc)
        blocks = [b for b in blocks if b.page_number >= min_content_page]

    if not blocks:
        return []

    toc_titles = {e.title.strip() for e in toc}
    matched_sizes = [
        b.font_size for b in blocks
        if b.block_type == "text" and b.text.strip() in toc_titles
    ]
    if matched_sizes:
        heading_font_size = min(matched_sizes)
    else:
        sizes = sorted({b.font_size for b in blocks if b.block_type == "text"}, reverse=True)
        heading_font_size = sizes[1] if len(sizes) >= 2 else (sizes[0] if sizes else 0.0)

    chunks: list[RawChunk] = []

    acc_text_parts: list[str] = []
    acc_first_page: Optional[int] = None
    acc_last_page: Optional[int] = None
    acc_chapter: Optional[str] = None
    section_title: Optional[str] = None

    def flush():
        nonlocal acc_text_parts, acc_first_page, acc_last_page
        text = "\n".join(acc_text_parts).strip()
        if text:
            chunks.append(RawChunk(
                chapter=acc_chapter or "Unknown Chapter",
                section_title=section_title,
                start_page=acc_first_page,
                end_page=acc_last_page,
                text=text,
                chunk_type="prose",
            ))
        acc_text_parts = []
        acc_first_page = None
        acc_last_page = None

    def start_new(block: TextBlock, chapter: str):
        nonlocal acc_text_parts, acc_first_page, acc_last_page, acc_chapter
        acc_chapter = chapter
        acc_text_parts = [block.text]
        acc_first_page = block.page_number
        acc_last_page = block.page_number

    for block in blocks:
        block_chapter, _ = assign_chapter(block.page_number, toc)

        # Table blocks flush the current prose accumulator and emit immediately, whole.
        if block.block_type == "table":
            flush()
            chunks.append(RawChunk(
                chapter=block_chapter,
                section_title=section_title,
                start_page=block.page_number,
                end_page=block.page_number,
                text=block.text,
                chunk_type="table",
            ))
            continue

        # New section boundary: numbered heading or bold+underline sub-heading.
        is_heading = _is_heading(block, heading_font_size)
        is_subheading = block.is_bold and block.is_underline and len(block.text.strip()) < 120
        if is_heading or is_subheading:
            flush()
            section_title = block.text.strip()
            start_new(block, block_chapter)
            continue

        # Chapter boundary enforcement (defensive: normally already caught by
        # the heading check above, since every real chapter starts with a
        # numbered heading).
        if acc_text_parts and block_chapter != acc_chapter:
            flush()
            start_new(block, block_chapter)
            continue

        # Token-budget split with overlap seeding.
        if acc_text_parts:
            candidate = "\n".join(acc_text_parts + [block.text])
            if count_tokens(candidate) > settings.chunk_max_tokens:
                prev_text = "\n".join(acc_text_parts)
                flush()
                overlap = _tail_by_tokens(prev_text, settings.chunk_overlap_tokens)
                acc_chapter = block_chapter
                acc_text_parts = ([overlap] if overlap else []) + [block.text]
                acc_first_page = block.page_number
                acc_last_page = block.page_number
                continue

        if not acc_text_parts:
            acc_chapter = block_chapter
            acc_first_page = block.page_number
        acc_text_parts.append(block.text)
        acc_last_page = block.page_number

    flush()
    return [c for c in chunks if c.text.strip()]


# ── Quick inspection script ───────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python -m app.ingestion.chunker data/pdfs/im_policy_test.pdf

    Prints chunk count, token distribution, and first 5 chunks so you can
    verify boundaries and page citations.
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
