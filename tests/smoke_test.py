"""
Day 3 — Both
End-to-End Smoke Test
=====================
Run AFTER all modules are implemented:
    python -m pytest tests/smoke_test.py -v

Tests are ordered to mirror the actual pipeline so failures pinpoint the
broken stage clearly.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "3.1 Annual Leave Entitlement\n"
    "All permanent employees are entitled to 21 days of paid annual leave per "
    "calendar year. Employees who join mid-year will receive leave on a "
    "pro-rata basis calculated from their date of joining. Leave must be "
    "applied for and approved by the line manager at least 5 working days in "
    "advance except in cases of emergency."
)

SAMPLE_ANSWER = (
    "[Chapter: Leave Policy | Section: 3.1 Annual Leave Entitlement | Pages: 42-42]\n"
    "All permanent employees are entitled to 21 days of paid annual leave per "
    "calendar year."
)

BAD_ANSWER = (
    "[Chapter: Leave Policy | Section: 3.1 Annual Leave Entitlement | Pages: 42-42]\n"
    "Employees receive 21 vacation days annually."  # paraphrase — should fail
)


# ── Stage 1: Verifier ─────────────────────────────────────────────────────────

def test_verifier_passes_exact_quote():
    from app.generation.verifier import verify_qa_pair
    ok, score = verify_qa_pair(SAMPLE_ANSWER, SAMPLE_TEXT)
    assert ok, f"Expected verification to pass, got score={score:.1f}"


def test_verifier_rejects_paraphrase():
    from app.generation.verifier import verify_qa_pair
    ok, score = verify_qa_pair(BAD_ANSWER, SAMPLE_TEXT)
    assert not ok, f"Expected verification to fail for paraphrase, got score={score:.1f}"


def test_strip_citation():
    from app.generation.verifier import strip_citation
    quote = strip_citation(SAMPLE_ANSWER)
    assert "All permanent employees" in quote
    assert "[Chapter:" not in quote


# ── Stage 2: Dedup ────────────────────────────────────────────────────────────

def test_dedup_empty_accepted():
    from app.diversity.dedup import encode_question, is_duplicate
    emb = encode_question("What is the annual leave entitlement?")
    dup, sim = is_duplicate(emb, [])
    assert not dup
    assert sim == 0.0


def test_dedup_detects_near_duplicate():
    from app.diversity.dedup import encode_question, is_duplicate
    q1 = "What is the annual leave entitlement for permanent employees?"
    q2 = "How many days of annual leave do permanent employees get?"
    e1 = encode_question(q1)
    e2 = encode_question(q2)
    dup, sim = is_duplicate(e2, [e1])
    assert dup, f"Expected near-duplicate to be flagged (sim={sim:.3f})"


def test_dedup_passes_different_question():
    from app.diversity.dedup import encode_question, is_duplicate
    q1 = "What is the annual leave entitlement?"
    q2 = "What is the notice period for resignation?"
    e1 = encode_question(q1)
    e2 = encode_question(q2)
    dup, sim = is_duplicate(e2, [e1])
    assert not dup, f"Expected different question to pass dedup (sim={sim:.3f})"


# ── Stage 3: Chunker (requires a real PDF — skip if not present) ──────────────

SAMPLE_PDF = Path("data/pdfs").glob("*.pdf")

@pytest.mark.skipif(
    not any(Path("data/pdfs").glob("*.pdf")),
    reason="No PDF in data/pdfs — place a handbook there to run this test"
)
def test_chunker_produces_chunks():
    pdf_path = next(Path("data/pdfs").glob("*.pdf"))
    from app.ingestion.parser import parse_pdf, extract_toc
    from app.ingestion.chunker import chunk_document, count_tokens

    blocks = parse_pdf(pdf_path)
    toc    = extract_toc(pdf_path)
    chunks = chunk_document(blocks, toc)

    assert len(chunks) > 0, "No chunks produced"

    for c in chunks:
        assert c.chapter,            f"Chunk missing chapter: {c}"
        assert c.text.strip(),       f"Chunk has empty text: {c}"
        assert c.start_page > 0,     f"Invalid start_page: {c}"
        assert c.end_page >= c.start_page, f"end_page < start_page: {c}"
        tokens = count_tokens(c.text)
        from app.config import get_settings
        assert tokens <= get_settings().chunk_max_tokens * 1.1, \
            f"Chunk too large ({tokens} tokens): {c.chapter} / {c.section_title}"

    print(f"\n✓ {len(chunks)} chunks from {pdf_path.name}")


# ── Stage 4: Export format ────────────────────────────────────────────────────

def test_alpaca_format():
    from app.export.formatter import _to_alpaca
    from app.models import QAPair
    pair = QAPair(
        job_id=1, chunk_id=1, question="Q?", base_question_index=0,
        answer=SAMPLE_ANSWER, quote_verified=True,
        chapter="Leave Policy", start_page=42, end_page=42,
    )
    record = _to_alpaca(pair)
    assert record["instruction"] == "Q?"
    assert record["input"] == ""
    assert "[Chapter:" in record["output"]


def test_sharegpt_format():
    from app.export.formatter import _to_sharegpt
    from app.models import QAPair
    pair = QAPair(
        job_id=1, chunk_id=1, question="Q?", base_question_index=0,
        answer=SAMPLE_ANSWER, quote_verified=True,
        chapter="Leave Policy", start_page=42, end_page=42,
    )
    record = _to_sharegpt(pair)
    assert record["conversations"][0]["from"] == "human"
    assert record["conversations"][1]["from"] == "gpt"
    assert "[Chapter:" in record["conversations"][1]["value"]
