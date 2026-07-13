"""
QA Pair Generator
==================
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

    Builds the QA_SYSTEM/QA_USER prompts (filling in the citation metadata
    and chunk text), calls the LLM, and parses the response as a JSON array
    via _parse_json_array(). Items missing a "question" or "answer" key are
    skipped rather than raising, so one malformed item doesn't fail the
    whole chunk. n defaults to settings.n_questions_per_chunk when omitted.
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
    Returns list of up to m rephrased versions of the input question.

    Builds the VARIATION_SYSTEM/VARIATION_USER prompts and calls the LLM at
    temperature=0.7 (higher than QA generation, to encourage diverse
    phrasing). Parses the response as a JSON array of strings via
    _parse_json_array(). m defaults to settings.m_variations_per_question
    when omitted. Returning fewer than m variations is acceptable — the
    caller does not retry purely to hit the count.
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
