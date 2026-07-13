"""
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

# 88 rejects clear paraphrases while allowing minor whitespace normalisation
# differences introduced by the PDF extractor.
FUZZY_THRESHOLD = 88

# Sliding window step size in characters.
# Smaller = more accurate but slower. 10 is a good balance.
WINDOW_STEP = 10


# ── Citation header handling ──────────────────────────────────────────────────

def strip_citation(answer: str) -> str:
    """
    Remove the [Chapter: ... | Pages: ...] header line the LLM prepends.
    Returns just the verbatim quote portion. If no ']' is found (no header
    present), returns the whole answer stripped.

    Example input:
        "[Chapter: Leave Policy | Section: Annual Leave | Pages: 42-42]\n
         All permanent employees are entitled to 21 days..."

    Example output:
        "All permanent employees are entitled to 21 days..."
    """
    if ']' not in answer:
        return answer.strip()

    idx = answer.index(']')
    rest = answer[idx + 1:]
    newline_idx = rest.find('\n')
    if newline_idx == -1:
        return rest.strip()
    return rest[newline_idx + 1:].strip()


def rebuild_answer(citation_header: str, corrected_quote: str) -> str:
    """Reconstruct a full answer string from its two parts."""
    return f"{citation_header}\n{corrected_quote}"


def build_citation_header(chapter: str, section: str | None, start_page: int, end_page: int) -> str:
    """
    Build a citation header from trusted chunk metadata (DB columns), not the
    LLM's own header text. The LLM sometimes mangles the header it was asked
    to copy (e.g. literally writing "None" for table-derived chunks) even
    when the quote body itself is fine — this guarantees a correct header
    regardless of what the LLM produced.
    """
    return f"[Chapter: {chapter} | Section: {section or ''} | Pages: {start_page}-{end_page}]"


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

    is_verified is True when the quote (after stripping the citation header)
    exists verbatim in source_chunk_text (score=100), or when a sliding-window
    fuzzy match via RapidFuzz token_sort_ratio scores >= FUZZY_THRESHOLD. The
    window slides across source_chunk_text in WINDOW_STEP-character steps,
    keeping the best score seen; if the quote is empty, returns (False, 0.0).
    """
    quote = strip_citation(answer).strip()
    if not quote:
        return (False, 0.0)

    if quote in source_chunk_text:
        return (True, 100.0)

    window_size = len(quote)
    if window_size > len(source_chunk_text):
        score = fuzz.token_sort_ratio(quote, source_chunk_text)
        return (score >= FUZZY_THRESHOLD, score)

    best = 0.0
    for i in range(0, len(source_chunk_text) - window_size + 1, WINDOW_STEP):
        window = source_chunk_text[i:i + window_size]
        score = fuzz.token_sort_ratio(quote, window)
        if score > best:
            best = score
        if best == 100.0:
            break
    return (best >= FUZZY_THRESHOLD, best)


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
    """
    quote = strip_citation(answer).strip()
    header = extract_citation_header(answer)

    if not quote:
        return None

    window_size = len(quote)
    if window_size > len(source_chunk_text):
        best_score = fuzz.token_sort_ratio(quote, source_chunk_text)
        best_span = source_chunk_text
    else:
        best_score = 0.0
        best_span = ""
        for i in range(0, len(source_chunk_text) - window_size + 1, WINDOW_STEP):
            window = source_chunk_text[i:i + window_size]
            score = fuzz.token_sort_ratio(quote, window)
            if score > best_score:
                best_score = score
                best_span = window
            if best_score == 100.0:
                break

    if best_score < FUZZY_THRESHOLD:
        return None
    return rebuild_answer(header, best_span)


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
