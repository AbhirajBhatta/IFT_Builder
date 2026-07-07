"""
Day 2 — Person A
QA Pair Generator
=================
Calls the LLM with a chunk, parses the JSON response, and returns raw
(unverified) QA dicts.

IMPORTANT: This module does NOT verify quotes.
           Always pass results through verifier.verify_qa_pair() before
           writing anything to the DB. The runner.py enforces this.

Quick test (one chunk, no DB needed):
    python -m app.generation.qa_generator
"""
from __future__ import annotations

import json
import re

from app.generation.llm_client import chat_completion
from app.generation.prompts import QA_SYSTEM, QA_USER, VARIATION_SYSTEM, VARIATION_USER
from app.config import get_settings

settings = get_settings()


def _parse_json_array(raw: str) -> list:
    """
    Safely parse a JSON array from LLM output.
    Strips accidental markdown fences (```json ... ```) before parsing.
    Raises ValueError with the raw string if parsing fails so the caller
    can log it and decide whether to retry.
    """
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        result = json.loads(clean)
        if not isinstance(result, list):
            raise ValueError(f"Expected JSON array, got {type(result).__name__}")
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\nRaw output:\n{raw}") from e


async def generate_qa_pairs(
    chunk_text: str,
    chapter: str,
    section: str,
    start_page: int,
    end_page: int,
    n: int | None = None,
) -> list[dict]:
    """
    Returns list of {"question": str, "answer": str} dicts.
    These are NOT verified — always pass through verifier.verify_qa_pair().

    Implementation guide:
    1.  Build the system prompt from QA_SYSTEM, formatting the citation
        placeholders: {chapter}, {section}, {start_page}, {end_page}.
    2.  Build the user prompt from QA_USER, formatting {chunk_text} and
        {n_questions} (use n or settings.n_questions_per_chunk).
    3.  Call chat_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
    4.  Parse with _parse_json_array(response).
    5.  Validate each item has "question" and "answer" keys.
        Skip (log and continue) items that are malformed rather than crashing.
    6.  Return the validated list.
    """
    n_questions = n or settings.n_questions_per_chunk

    system_prompt = QA_SYSTEM.format(
        chapter=chapter, section=section, start_page=start_page, end_page=end_page
    )
    user_prompt = QA_USER.format(chunk_text=chunk_text, n_questions=n_questions)

    response = await chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    raw_pairs = _parse_json_array(response)

    pairs = []
    for item in raw_pairs:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
        pairs.append({"question": item["question"], "answer": item["answer"]})
    return pairs


async def generate_variations(question: str, m: int | None = None) -> list[str]:
    """
    Returns list of M rephrased versions of the input question.

    Implementation guide:
    1.  m = m or settings.m_variations_per_question
    2.  Build system + user prompts from VARIATION_SYSTEM / VARIATION_USER.
    3.  Call chat_completion with low temperature (0.7 is good for variation —
        slightly higher than QA generation to encourage diverse phrasing).
    4.  Parse with _parse_json_array(response).
    5.  Return list[str]. If fewer than m variations are returned, that's
        acceptable — don't retry just for count.
    """
    m = m or settings.m_variations_per_question

    system_prompt = VARIATION_SYSTEM.format(m=m)
    user_prompt = VARIATION_USER.format(question=question, m=m)

    response = await chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=0.7)

    variations = _parse_json_array(response)
    return [v for v in variations if isinstance(v, str)]


# ── Quick manual test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    SAMPLE_CHUNK = """\
    3.1 Annual Leave Entitlement
    All permanent employees are entitled to 21 days of paid annual leave per calendar year.
    Employees who join mid-year will receive leave on a pro-rata basis calculated from their
    date of joining. Leave must be applied for and approved by the line manager at least
    5 working days in advance except in cases of emergency.
    """

    async def main():
        print("=== Generating QA pairs ===")
        pairs = await generate_qa_pairs(
            chunk_text=SAMPLE_CHUNK,
            chapter="Leave Policy",
            section="Annual Leave",
            start_page=42,
            end_page=42,
            n=3,
        )
        for i, p in enumerate(pairs):
            print(f"\n[{i}] Q: {p['question']}")
            print(f"     A: {p['answer'][:200]}")

        if pairs:
            print("\n=== Generating variations for first question ===")
            variations = await generate_variations(pairs[0]["question"], m=2)
            for v in variations:
                print(f"  - {v}")

    asyncio.run(main())
