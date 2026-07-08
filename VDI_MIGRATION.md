# VDI Migration Notes

This is a **flag-only** list — nothing here has been "fixed" for portability.
Each item is something that behaved a certain way on a local dev laptop
(free-tier Groq, locally-downloaded models, open internet, Windows dev
machine) that may behave differently once this moves to the company VDI
(likely restricted network, Windows, manual `.env` retyping, the real ~41-page
handbook instead of the synthetic test PDF). Address deliberately once
actually on the VDI, not before.

---

## 1. Hardcoded paths / Windows vs Unix

No absolute or backslash-hardcoded paths were found in `app/` — all file
access goes through `pathlib.Path` with relative, forward-slash defaults
(`config.py`: `db_path = "db/ift.db"`, `data_output_dir = "data/output"`;
`routes.py` derives `data/pdfs/` from `data_output_dir`'s parent). This
should behave the same on Windows. Worth re-confirming once real absolute
paths enter the picture (e.g. if the VDI mounts the handbook from a network
share with a UNC path like `\\server\share\...`, which `pathlib` handles but
is worth testing explicitly).

**Real Windows-specific issue found this session, already fixed in code:**
`parser.py`'s `parse_pdf()`/`extract_toc()` opened PDFs via `fitz.open()` but
never called `doc.close()`. On Windows (unlike Unix), an unclosed file handle
blocks any later attempt to overwrite the same path — this caused a genuine
`PermissionError` mid-session when re-uploading a file with the same name.
Fixed by adding explicit `doc.close()` calls. Still worth stress-testing on
the VDI: re-uploading a PDF with an identical filename while a long-lived
uvicorn process is running.

**Separately observed, not a code bug:** during testing, `data/pdfs/
im_policy_test.pdf` got locked by some non-Python Windows process (not
uvicorn/pytest — confirmed via `Get-Process`), most likely Explorer's PDF
thumbnail/preview handler or a PDF viewer, after the file had been
opened/viewed several times. On a VDI where users may have the real handbook
open in Acrobat while testing, re-uploading a same-named file could hit this.
No code fix applies here — it's an OS/third-party-app lock, not ours.

---

## 2. sentence-transformers model download (`all-MiniLM-L6-v2`)

Currently downloads automatically on first call to `dedup.encode_question()`
(lazy-loaded in `_get_model()`). Confirmed local cache location and size:

```
C:\Users\<user>\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2
(91.6 MB)
```

If the VDI has no internet access, this download will fail at first
`is_duplicate()` call (i.e. as soon as the first job actually tries to
generate QA pairs). To pre-stage it: copy that whole cache folder to the
equivalent path under the VDI user's profile (or set `HF_HOME` to point at a
folder you've copied it into, if the profile path differs). No code change
needed — `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`
will use the local cache without hitting the network if it's already there.

---

## 3. `requirements.txt` — one big untracked dependency

`sentence-transformers==3.0.0` pulls in **torch** as a transitive dependency
(confirmed installed: `torch==2.12.1`), which is **not pinned in
`requirements.txt`** at all. Torch is a large (hundreds of MB+), highly
platform/CUDA-specific package. Two risks:

- If the VDI has no internet for `pip install`, torch needs an offline wheel
  matched to the VDI's exact Python version and CPU/GPU — pre-download the
  correct wheel (CPU-only build is almost certainly right for a VDI; no need
  for a CUDA build unless the VDI has a GPU) and note the resolved version
  before leaving the dev laptop, since an unpinned transitive dependency can
  silently resolve to a different torch build.
- `pymupdf` also ships platform-specific compiled wheels (not pure Python) —
  same offline-wheel concern applies.

Added `pytest==9.1.1` to `requirements.txt` this session (marked dev/test
only) to run `smoke_test.py` — same offline consideration applies, though
it's pure Python and much smaller.

---

## 4. LLM provider — Groq free tier is a dev-only stand-in

`.env` currently points at Groq (`LLM_BASE_URL=https://api.groq.com/openai/v1`,
`LLM_MODEL=llama-3.3-70b-versatile`) with a personal free-tier key. Confirmed
this session that the free tier caps out at **100,000 tokens/day per model**
— we hit this mid-test (job 7 failed 13/19 chunks on exactly this limit; see
`error_message` column on those `Chunk` rows for the raw 429 response). This
is fine for local dev/iteration but:

- Whatever LLM endpoint the company approves for production must be swapped
  in via `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` — `llm_client.py` already
  supports any OpenAI-compatible endpoint, no code change needed, just config.
- Per company policy, `.env` must be **manually retyped** on the VDI, not
  copied from this machine — don't paste the current Groq key there, and
  don't assume the file transfers as-is.
- The retry/backoff logic in `llm_client.py` (`MAX_RETRIES=4`,
  `BASE_DELAY=2.0`, doubling) is tuned for *transient* 429s/5xx — it gives up
  after ~30s total. A hard daily-quota 429 (like we hit) won't resolve within
  that window; this is expected, not a bug, but worth knowing the retry logic
  won't paper over a real quota exhaustion, whatever provider is used in
  production.

---

## 5. Chunking/parsing assumptions tuned to the synthetic test PDF

No real handbook PDF has been available locally this whole sprint — all
parser/chunker validation ran against a synthetic 10-page PDF I generated to
mimic the real document's structure (title → doc control → ToC → numbered +
bold/underline headings → 4 tables), based on screenshots of the real
handbook, not the actual file. Specific things that should be re-validated
against the real ~41-page Information Management Policy PDF once available:

- **Heading font-size detection** (`chunker.py`'s `heading_font_size`,
  derived by matching blocks against ToC titles) — assumes level-1 and
  level-2 headings share a consistent, distinguishable font size throughout
  the real document. Untested against real inconsistencies (e.g. if some
  headings were manually resized in the source Word doc).
- **Bold+underline sub-heading detection** — the underline-detection heuristic
  (scanning `page.get_drawings()` for lines near a span's baseline) was
  tuned against my synthetic PDF's programmatically-drawn underlines. Word's
  actual underline rendering in an exported PDF may differ in exact
  positioning — re-verify against the real doc's "Access Registration"-style
  headings (see the PDF formatting notes screenshot).
- **Table detection** (`page.find_tables()`) — worked cleanly on my
  synthetic PDF's simple grid-line tables. The real doc's actual table
  borders/styling (possibly no visible gridlines, merged cells, etc.) may
  behave differently — PyMuPDF's table detection is heuristic-based.
- **No PDF bookmarks assumption** — my synthetic PDF was deliberately built
  without embedded ToC bookmarks to exercise `extract_toc()`'s Strategy B
  (font-size/regex fallback) rather than Strategy A (`doc.get_toc()`), since
  that seemed like the riskier untested path. If the real handbook (being a
  Word export) *does* have embedded bookmarks, Strategy A will actually be
  used instead and has had zero real-world testing this sprint.

---

## 6. Performance/scale assumptions

`dedup.py`'s docstring originally cited a "two 500-page handbooks" scale
estimate (~6,660 questions) from the original two-document design — now
updated to reflect the real single ~41-page document (~700-800 questions
estimated). The dedup approach (pairwise cosine similarity, no FAISS) is
more than fast enough at either scale, so this was never a real risk — just
a stale comment, now corrected to match reality rather than left to mislead
anyone re-reading the code later.

The `CHUNK_MAX_TOKENS=600` / `CHUNK_OVERLAP_TOKENS=75` defaults and the
frontend's default 5 questions/3 variations were sized for this ~41-page
document specifically (see README's Implementation Status section) — these
should hold for the real handbook given it's the same rough page count as
what they were calibrated against, but haven't been validated against the
*actual* file, only the synthetic stand-in.

---

## 7. Everything else observed this session

- **`db/ift.db` schema migrations are manual.** `SQLModel.metadata.create_all()`
  only creates missing *tables*, not missing *columns* on existing tables.
  When `Job.n_questions_per_chunk`/`m_variations_per_question` were added
  this session, the existing local `db/ift.db` needed a manual `ALTER TABLE`
  (done this session, non-destructively). If the schema changes again before
  the VDI's DB is created fresh, this won't be an issue there (fresh
  `create_all()` will include all current columns) — but if the VDI ever
  carries forward an existing `ift.db` across a code update, the same manual
  migration step will be needed again.
- **Uvicorn's `--reload` flag matters during iteration.** Running without it
  meant code edits to already-imported modules (e.g. `runner.py`) weren't
  picked up until a manual restart — caused confusing "why didn't my fix
  apply" moments this session. Not VDI-specific, but worth keeping in mind
  when setting up the VDI's run process.
