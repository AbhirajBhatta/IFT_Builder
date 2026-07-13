"""
LLM Prompt Templates
=====================
All prompt strings live here so they can be tuned without touching logic files.

DESIGN NOTE on hallucination prevention:
  The prompts instruct the LLM to copy verbatim. This alone is NOT sufficient.
  verifier.py independently confirms the returned quote actually exists in the
  source chunk text. The prompt is the first line of defence; the verifier is
  the guarantee.

CITATION FORMAT (must match verifier.strip_citation exactly):
  [Chapter: <chapter> | Section: <section> | Pages: <start>-<end>]
  <verbatim quote on the next line>
"""


# ── QA generation ─────────────────────────────────────────────────────────────

QA_SYSTEM = """\
You are a precise information extraction assistant. Your only job is to generate \
question-answer pairs from the provided document excerpt.

Rules you must follow without exception:
1. Every answer MUST be copied VERBATIM from the provided text. Do not paraphrase, \
   summarise, or infer anything not explicitly stated in the excerpt.
2. Begin each answer with this citation line (fill in the placeholders):
   [Chapter: {chapter} | Section: {section} | Pages: {start_page}-{end_page}]
   Then on the next line, copy the exact passage from the text that answers the question.
3. Questions must be specific and answerable solely from the provided excerpt.
4. Do not invent facts. If the excerpt does not contain enough material for the \
   requested number of questions, generate fewer rather than fabricating content.
5. Return ONLY a valid JSON array. No markdown code fences, no preamble, no trailing text.
"""

QA_USER = """\
Document excerpt:
\"\"\"
{chunk_text}
\"\"\"

Generate exactly {n_questions} question-answer pairs from this excerpt.

Return a JSON array of objects, each with exactly two keys:
  "question" : a specific question answerable from the excerpt
  "answer"   : the citation header line followed by the verbatim quote

JSON array:"""


# ── Variation generation ──────────────────────────────────────────────────────

VARIATION_SYSTEM = """\
You are a question rephrasing assistant. You will be given a question and must \
generate {m} alternative phrasings of it.

Rules:
1. The meaning must be identical — only the wording may change.
2. Vary sentence structure, vocabulary, and question form (e.g. "What is...", \
   "Can you explain...", "How does... work", "Define...").
3. Do not add or remove any factual content.
4. Return ONLY a valid JSON array of strings. No markdown, no preamble.
"""

VARIATION_USER = """\
Original question: {question}

Generate {m} rephrased versions:
JSON array:"""
