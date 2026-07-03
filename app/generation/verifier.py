"""
Day 2 — Person A
Quote Verifier
==============
This is the anti-hallucination guarantee layer.

The LLM was asked to copy text verbatim. This module independently confirms
the returned answer actually appears in (or closely matches) the source chunk
text — without trusting the LLM's compliance.

A QA pair that fails verification is NEVER written to the DB.

Verification pipeline per answer:
  1. Strip the citation header line to isolate the quote.
  2. Exact substring check (fast path — handles fully compliant responses).
  3. If that fails, sliding-window fuzzy match via RapidFuzz token_sort_ratio.
  4. If fuzzy score < FUZZY_THRESHOLD → REJECTED.
  5. Optional autocorrect: find the best-matching span and substitute it back
     in, preserving the citation header. Useful for minor whitespace/OCR drift.

Quick test:
    python -m app.generation.verifier
"""
from __future__ import annotations

from rapidfuzz import fuzz

# Tune this after running on real PDFs on Day 2.
# 88 is a good starting point — it rejects clear paraphrases but allows
# minor whitespace normalisation differences from the PDF extractor.
FUZZY_THRESHOLD = 88

# Sliding window step size in characters.
# Smaller = more accurate but slower. 10 is a good balance.
WINDOW_STEP = 10


# ── Citation header handling ──────────────────────────────────────────────────

def strip_citation(answer: str) -> str:
    """
    Remove the [Chapter: ... | Pages: ...] header line the LLM prepends.
    Returns just the verbatim quote portion.

    Implementation guide:
    The citation header always ends with ']'. Find the first ']' in the string,
    then strip everything up to and including the first newline after it.

    Example input:
        "[Chapter: Leave Policy | Section: Annual Leave | Pages: 42-42]\n
         All permanent employees are entitled to 21 days..."

    Example output:
        "All permanent employees are entitled to 21 days..."

    Edge case: if no ']' is found, return the whole answer stripped.
    """
    raise NotImplementedError


def rebuild_answer(citation_header: str, corrected_quote: str) -> str:
    """Reconstruct a full answer string from its two parts."""
    return f"{citation_header}\n{corrected_quote}"


def extract_citation_header(answer: str) -> str:
    """Return just the [Chapter: ...] header line, or empty string if absent."""
    if ']' in answer:
        idx = answer.index(']')
        return answer[:idx + 1].strip()
    return ""


# ── Core verification ─────────────────────────────────────────────────────────

def verify_qa_pair(answer: str, source_chunk_text: str) -> tuple[bool, float]:
    """
    Returns (is_verified: bool, match_score: float 0-100).

    is_verified is True when the quote exists verbatim (score=100) or
    fuzzy score >= FUZZY_THRESHOLD.

    Implementation guide:
    1.  quote = strip_citation(answer).strip()
        If quote is empty, return (False, 0.0).

    2.  Exact check:
            if quote in source_chunk_text:
                return (True, 100.0)

    3.  Sliding window fuzzy match:
        window_size = len(quote)
        If window_size > len(source_chunk_text):
            Run fuzz.token_sort_ratio(quote, source_chunk_text) directly.
            Return (score >= FUZZY_THRESHOLD, score).

        Otherwise:
            best = 0.0
            for i in range(0, len(source_chunk_text) - window_size + 1, WINDOW_STEP):
                window = source_chunk_text[i : i + window_size]
                score  = fuzz.token_sort_ratio(quote, window)
                if score > best:
                    best = score
                if best == 100.0:
                    break   # can't do better
            return (best >= FUZZY_THRESHOLD, best)
    """
    raise NotImplementedError


# ── Optional autocorrect ──────────────────────────────────────────────────────

def autocorrect_quote(answer: str, source_chunk_text: str) -> str | None:
    """
    Find the best-matching span in source_chunk_text and return a corrected
    answer with that span substituted in, preserving the citation header.

    Returns None if the best span score is below FUZZY_THRESHOLD
    (caller should reject the pair entirely).

    Use case: the LLM copied the quote almost perfectly but with minor
    whitespace normalisation or hyphenation differences introduced by the
    PDF extractor. This recovers those cases instead of discarding them.

    Implementation guide:
    1.  quote = strip_citation(answer).strip()
    2.  header = extract_citation_header(answer)
    3.  Slide a window of len(quote) across source_chunk_text (step=WINDOW_STEP),
        track (best_score, best_span).
    4.  If best_score < FUZZY_THRESHOLD: return None
    5.  return rebuild_answer(header, best_span)
    """
    raise NotImplementedError


# ── Quick manual test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    SOURCE = (
        "All permanent employees are entitled to 21 days of paid annual leave "
        "per calendar year. Employees who join mid-year will receive leave on a "
        "pro-rata basis calculated from their date of joining."
    )

    GOOD_ANSWER = (
        "[Chapter: Leave Policy | Section: Annual Leave | Pages: 42-42]\n"
        "All permanent employees are entitled to 21 days of paid annual leave "
        "per calendar year."
    )

    BAD_ANSWER = (
        "[Chapter: Leave Policy | Section: Annual Leave | Pages: 42-42]\n"
        "Employees get 21 days off every year as paid leave."  # paraphrase
    )

    FUZZY_ANSWER = (
        "[Chapter: Leave Policy | Section: Annual Leave | Pages: 42-42]\n"
        "All permanent employees are entitled to 21 days of paid  annual leave "  # double space
        "per calender year."  # typo
    )

    for label, ans in [("GOOD", GOOD_ANSWER), ("BAD", BAD_ANSWER), ("FUZZY", FUZZY_ANSWER)]:
        ok, score = verify_qa_pair(ans, SOURCE)
        print(f"{label:6s} | verified={ok} | score={score:.1f}")

    print("\nAutocorrect on FUZZY:")
    corrected = autocorrect_quote(FUZZY_ANSWER, SOURCE)
    print(corrected)
