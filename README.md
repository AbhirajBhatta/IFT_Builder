# IFT Dataset Builder

Converts a company policy handbook PDF into an Instruction Fine-Tuning (IFT)
dataset for QLoRA/LoRA fine-tuning, with verified verbatim quotes, fault-tolerant
background processing, and SSE live progress streaming.

The pipeline: upload a PDF → parse → chunk → generate QA pairs via LLM →
verify each answer against the source text → deduplicate → export as
Alpaca or ShareGPT JSON.

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

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Create your `.env` file:
   ```bash
   cp .env.example .env
   ```
   Fill in `LLM_API_KEY` (and `LLM_BASE_URL`/`LLM_MODEL` if not using OpenAI
   directly). **Quote values that contain special characters** (spaces, `#`,
   `$`, etc.) — e.g. `LLM_API_KEY="sk-..."` — since an unquoted value with a
   `#` will be truncated at the `#` by the `.env` parser. Quoting is always
   safe even when not strictly required.

3. First run will download the `all-MiniLM-L6-v2` sentence-transformers
   model (~92 MB) the first time deduplication runs. If the target machine
   has no internet access, pre-download it on a machine that does and copy
   the Hugging Face cache folder
   (`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`)
   to the same path on the target machine, or point `HF_HOME` at a folder
   containing it.

4. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The `db/` directory and SQLite database file, and the `data/pdfs/` and
   `data/output/` directories, are created automatically on first run — no
   manual folder setup required.

5. Open http://localhost:8000 — upload a PDF, click Start Generation.

Run the test suite: `python -m pytest tests/smoke_test.py -v`

---

## Usage

- **Start a job**: upload a PDF via the web UI, or `POST /jobs/` with the
  file and optional `n_questions`/`m_variations` overrides.
- **Track progress**: `GET /jobs/{job_id}/stream` (SSE) or poll
  `GET /jobs/{job_id}`.
- **Resume a job**: if a job finishes with failed chunks (e.g. after an LLM
  provider rate limit), `POST /jobs/{job_id}/resume` re-launches processing
  for only the chunks that aren't done yet — no rework of completed chunks.
  If the failure was caused by an exhausted/invalid LLM key or a blocked
  endpoint, update credentials first — see below.
- **Export the dataset**: `GET /jobs/{job_id}/export?format=alpaca` (or
  `sharegpt`).

### Running out of API quota mid-job

No work is lost when chunks fail — completed chunks stay `done`, and the
real error for the most recent failure is shown live in the frontend. If it
looks like a quota/rate-limit or network/proxy issue, the **LLM
Credentials** box on the page is automatically highlighted with a hint.

1. Fill in the API Key / Base URL / Model fields you want to change (leave
   the rest blank) and click **Save Credentials**.
2. This takes effect immediately for the running server — **no restart
   required** — and is also written back to `.env` so it survives a real
   restart later.
3. Click **Retry Failed Chunks** (or `POST /jobs/{job_id}/resume`). Only
   chunks still `failed`/`pending` are retried (up to `MAX_CHUNK_RETRIES`,
   default 3) — already-`done` chunks are untouched.

The same update can be made directly via the API:
`POST /settings/llm-credentials` with a JSON body of
`{"api_key": "...", "base_url": "...", "model": "..."}` (all fields
optional — only the ones provided are changed). The key and base URL are
write-only and are never echoed back in the response.

---

## API Reference

| Method | Path                          | Description                                       |
|--------|-------------------------------|----------------------------------------------------|
| POST   | `/jobs/`                      | Upload PDF, create chunks, launch generation       |
| GET    | `/jobs/{job_id}`               | Job status + progress counters                     |
| GET    | `/jobs/{job_id}/stream`        | SSE live progress stream                           |
| POST   | `/jobs/{job_id}/resume`        | Resume an interrupted/partially-failed job         |
| GET    | `/jobs/{job_id}/export`        | Download the dataset (`?format=alpaca\|sharegpt`)  |
| POST   | `/settings/llm-credentials`    | Hot-swap the LLM API key / base URL / model        |

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
The verifier only checks the quote body, not the header, and the LLM can
mangle headers for table-derived chunks (writing literal `"None"`) even when
the quote itself is fine. Since the ground-truth chapter/section/page data
already lives on the `Chunk` row, reconstructing the header from that is
strictly more reliable than parsing what the LLM wrote.
