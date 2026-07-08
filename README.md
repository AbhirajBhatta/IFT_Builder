# IFT Dataset Builder

Converts a company policy handbook PDF into an Instruction Fine-Tuning (IFT)
dataset for QLoRA/LoRA fine-tuning, with verified verbatim quotes, fault-tolerant
background processing, and SSE live progress streaming.

**Status: fully implemented and tested end-to-end** (upload → parse → chunk →
generate → verify → dedup → export), against a synthetic test PDF and real
Groq API calls. See [Implementation Status](#implementation-status) below for
what changed from the original 3-day plan, and
[VDI_MIGRATION.md](VDI_MIGRATION.md) before moving this to the company VDI.

---

## Stack

| Layer       | Library                        |
|-------------|--------------------------------|
| API         | FastAPI + Uvicorn              |
| Database    | SQLite via SQLModel            |
| PDF parsing | PyMuPDF (fitz)                 |
| Tokeniser   | tiktoken                       |
| LLM calls   | httpx (async, any OAI-compat.) |
| Embeddings  | sentence-transformers          |
| Fuzzy match | RapidFuzz                      |
| Tests       | pytest (dev only)              |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in LLM_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000 — upload a PDF, click Start Generation.

Run the test suite: `python -m pytest tests/smoke_test.py -v`

---

## Implementation Status

Everything in the original ownership split (parsing, chunking, generation,
verification, worker/checkpointing, dedup, export, routes) is implemented and
has been run end-to-end multiple times against both a synthetic test PDF and
the real Groq API. Deviations from the original plan below.

### General "add PDF" instead of HR/Finance document types
The original design assumed two fixed handbooks (`document_type: "hr" |
"finance"`). The real target is a single company policy document (a 41-page
Information Management Policy), so the document-type selector was removed
from the frontend entirely — upload just takes a PDF. `Job.document_type`
still exists in the schema (defaulted to `"policy"` server-side) so no DB
migration was needed for this change.

### Chunking handles two heading styles, not just numbering
The real handbook uses numbered headings (`1.`, `1.1`, `3.2.1`) **and**
bold+underlined text with no number prefix (e.g. "Access Registration") as a
secondary heading style. `parser.py` detects underlines by scanning
`page.get_drawings()` for lines under a span's baseline (PyMuPDF's font flags
have no underline bit), and `chunker.py` treats both styles as section
boundaries. Numbered sub-subsections (e.g. `3.3.1`–`3.3.6`) are *not* treated
as new sections — they fold into their parent section as one chunk, per the
"stepwise guide" note in the original PDF formatting spec.

### Table detection via PyMuPDF, front-matter excluded
The parser docstring didn't specify how `block_type="table"` gets set — this
uses `page.find_tables()` (available in the pinned PyMuPDF 1.24.3) to detect
table regions and collapse each into one block. Separately, chunker.py now
skips any page before the first ToC entry's start page, so the title/document
control/contents pages don't get chunked as fake content.

### Citation headers are rebuilt from trusted DB metadata, not the LLM's own text
The LLM is asked to prepend a citation header to each answer. In testing,
this worked for prose chunks but the LLM sometimes wrote `[Chapter: None |
Section: None | Pages: None]` for table-derived chunks — a real trained-model
citation risk, since the verifier only validated the quote body, not the
header. `runner.py` now always rebuilds the header from the chunk's own DB
columns (`build_citation_header()` in `verifier.py`) after verification
passes, regardless of what the LLM wrote.

### Per-job configurable dataset size
`Job.n_questions_per_chunk` / `Job.m_variations_per_question` (nullable —
`None` falls back to the `.env` defaults) let each upload override
`N_QUESTIONS_PER_CHUNK`/`M_VARIATIONS_PER_QUESTION` without touching config.
Exposed in the frontend as two number inputs, pre-filled with the defaults
(5 questions/chunk, 3 variations/question — sized for a ~40-page document:
~35 chunks × 5 × 4 ≈ 700 raw variants before dedup, settling to roughly
300-500 verified pairs). Lower for a quick test run, raise for a larger
handbook.

### Resuming a job is now an actual API call
The original plan's checkpoint design (`_get_pending_chunks()` returns
anything not `done`, including failed chunks under the retry limit) was
fully implemented, but nothing exposed it — `POST /jobs/` always created a
brand-new job. Added `POST /jobs/{job_id}/resume`, which just re-launches
`run_job()` for an existing job id. The frontend shows a "Retry Failed
Chunks" button when a job finishes with `failed_chunks > 0` (e.g. after
hitting an LLM provider's rate limit mid-run).

### Known test calibration gap
`tests/smoke_test.py::test_dedup_detects_near_duplicate` fails against the
real `all-MiniLM-L6-v2` model — its example question pair scores 0.792
similarity, just under the configured `DEDUP_SIMILARITY_THRESHOLD=0.85`. The
`is_duplicate()` implementation itself is correct (verified separately with a
pair that scores 0.952 and is correctly flagged) — this is either the test's
example pair being too loosely phrased, or the threshold being stricter than
intended. Not changed unilaterally since it affects final dataset diversity;
worth a decision before relying on this test in CI.

---

## Resuming an Interrupted or Rate-Limited Job

The DB is the checkpoint — each `Chunk` row tracks `pending` / `in_progress`
/ `done` / `failed`, and `Job.completed_chunks` / `failed_chunks` are updated
per chunk. To resume:

```bash
curl -X POST http://localhost:8000/jobs/{job_id}/resume
```

`_get_pending_chunks()` returns everything not `done` — including chunks that
failed transiently (e.g. an LLM provider's rate limit) and are still under
`MAX_CHUNK_RETRIES` (3) — so the job continues from where it stopped without
redoing already-completed chunks. The frontend surfaces this as a "Retry
Failed Chunks" button once a job finishes with failures.

---

## Output Format (Alpaca)

```json
[
  {
    "instruction": "What is the annual leave entitlement for permanent employees?",
    "input": "",
    "output": "[Chapter: Leave Policy | Section: 3.1 Annual Leave | Pages: 42-42]\nAll permanent employees are entitled to 21 days of paid annual leave per calendar year."
  }
]
```

The citation header inside `output` is intentional — the fine-tuned model learns
to emit chapter and page citations as part of every answer at inference time.
`input` is always empty by design — the instruction is self-contained, no
extra context is needed. `sharegpt` format is also available via
`?format=sharegpt` on the export endpoint.

---

## Key Design Decisions

**Why is the citation header inside the answer text and not sidecar metadata?**
Because the model only sees the `output` field during training. Metadata is
invisible to the model unless you explicitly include it in the text. Putting
the citation inside `output` means the model learns to produce it.

**Why SQLite and not a real task queue?**
One document, one server, one team. SQLite + asyncio gives resumability
without Celery/Redis infrastructure overhead. Revisit if scope grows.

**Why substring/fuzzy match for verification and not just trust the LLM?**
LLMs comply with "copy verbatim" instructions ~85-90% of the time. The remaining
10-15% produces confident paraphrases that are impossible to distinguish by eye
at scale. The verifier catches these programmatically with zero false negatives.

**Why rebuild the citation header from DB metadata instead of trusting the
LLM's own header text?**
The verifier only checked the quote body, not the header — and the LLM
reliably mangled headers for table-derived chunks (writing literal `"None"`)
even when the quote itself was fine. Since the ground-truth chapter/section/
page data already lives on the `Chunk` row, reconstructing the header from
that is strictly more reliable than parsing what the LLM wrote.
