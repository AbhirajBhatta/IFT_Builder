# IFT Dataset Builder

Converts HR / Finance handbook PDFs into Instruction Fine-Tuning (IFT) datasets
for QLoRA/LoRA fine-tuning, with verified verbatim quotes, fault-tolerant
background processing, and SSE live progress streaming.

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

---

## 3-Day Implementation Plan

### Day 1 — Person A: Parsing
1. `app/ingestion/parser.py` — implement `parse_pdf()` and `extract_toc()`
2. `app/ingestion/chunker.py` — implement `chunk_document()`
3. **Validate**: `python -m app.ingestion.chunker data/pdfs/hr_handbook.pdf`
   Print chunks, check page numbers match real PDF, check no chapter boundary crossings.
4. **EOD sync with Person B**: share `RawChunk` dataclass fields.
   Confirm they match `Chunk` model in `app/models.py`.

### Day 1 — Person B: Infrastructure
1. `requirements.txt`, `.env`, `app/config.py`, `app/database.py` — already written, review and install.
2. `app/models.py` — already written, review field names with Person A at EOD.
3. `app/generation/prompts.py` — already written, read and understand the citation format.
4. **Run**: `uvicorn app.main:app --reload` — confirm startup creates `db/ift.db`.

### Day 2 — Person A: Generation + Verification
1. `app/generation/llm_client.py` — implement `chat_completion()`
2. `app/generation/qa_generator.py` — implement `generate_qa_pairs()` and `generate_variations()`
3. `app/generation/verifier.py` — implement `strip_citation()`, `verify_qa_pair()`, `autocorrect_quote()`
4. **Validate**: `python -m app.generation.qa_generator` (uses built-in sample chunk)
   Check that returned quotes exist verbatim in the source text.
5. **EOD sync**: run `python -m app.generation.verifier` — confirm GOOD passes, BAD fails.

### Day 2 — Person B: Worker + Dedup
1. `app/diversity/dedup.py` — implement `is_duplicate()`
2. `app/worker/runner.py` — implement `run_job()` following the step-by-step guide in the file
3. **Validate**: call `run_job()` in a small asyncio script on one real chunk.
   Check a QAPair row appears in the DB with `quote_verified=True`.
4. **EOD sync**: Person A tests qa_generator on a real chunk, Person B feeds output to runner.
   Fix any interface mismatch (e.g. missing fields, wrong key names in pair dict).

### Day 3 — Person A: API Routes + SSE
1. `app/api/routes.py` — implement `create_job()` and `export_dataset()`
2. `app/api/sse.py` — already written, just included in router via `main.py`
3. `app/main.py` — already written
4. **Validate**: POST to `/jobs/` with a real PDF via curl or the browser.
   Watch SSE events fire in browser DevTools → Network → EventStream.

### Day 3 — Person B: Export + Frontend + E2E
1. `app/export/formatter.py` — implement `export_job()`
2. `frontend/index.html` — already written, test in browser
3. **Validate**: `python -m pytest tests/smoke_test.py -v`
4. Full end-to-end test: upload real handbook, wait for completion, download JSON,
   load into training framework and confirm it parses.

---

## Resuming an Interrupted Job

The DB is the checkpoint. On restart, simply call `run_job(job_id)` again.
`_get_pending_chunks()` returns all chunks not in `done` state — including
`in_progress` chunks from the crashed run — so the job continues from where it stopped.

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

---

## Key Design Decisions

**Why is the citation header inside the answer text and not sidecar metadata?**
Because the model only sees the `output` field during training. Metadata is
invisible to the model unless you explicitly include it in the text. Putting
the citation inside `output` means the model learns to produce it.

**Why SQLite and not a real task queue?**
Two fixed documents, one server, one team. SQLite + asyncio gives resumability
without Celery/Redis infrastructure overhead. Revisit if scope grows.

**Why substring/fuzzy match for verification and not just trust the LLM?**
LLMs comply with "copy verbatim" instructions ~85-90% of the time. The remaining
10-15% produces confident paraphrases that are impossible to distinguish by eye
at scale. The verifier catches these programmatically with zero false negatives.
