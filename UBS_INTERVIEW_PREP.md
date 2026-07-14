# UBS Interview Prep: IFT Dataset Builder

Interview date: July 10, 2026  
Target framing: Data/AI internship project

## 1. 60-Second Project Pitch

I built an Instruction Fine-Tuning dataset builder that converts a company policy PDF into a high-quality training dataset for LoRA/QLoRA fine-tuning.

The system takes a PDF handbook, extracts its text, tables, headings, and page metadata, chunks it into section-aware pieces, uses an LLM to generate question-answer pairs, verifies that every answer is a verbatim quote from the source, removes semantically duplicate questions, and exports the final dataset in Alpaca or ShareGPT format.

The main engineering idea was that LLMs are useful for generation but should not be blindly trusted. So I designed the pipeline with deterministic guardrails: quote verification, citation reconstruction from database metadata, semantic deduplication, and checkpointed background processing. The final result is a fine-tuning dataset where answers contain grounded citations like chapter, section, and page number.

## 2. One-Line Summary

This project is a reliable data-generation pipeline around an LLM: the LLM creates candidate Q&A pairs, but deterministic code decides what is safe enough to enter the training dataset.

## 3. Architecture Story

The pipeline is:

PDF upload -> PDF parsing -> document chunking -> LLM Q&A generation -> quote verification -> semantic deduplication -> database checkpointing -> JSON export

In simple terms:

1. The user uploads a policy PDF through the frontend.
2. FastAPI saves the file and starts ingestion.
3. PyMuPDF extracts text spans, font metadata, tables, headings, and page numbers.
4. The chunker groups content by chapter and section, while respecting a token budget.
5. The worker sends each chunk to an OpenAI-compatible LLM endpoint.
6. The LLM returns candidate questions and citation-backed answers.
7. The verifier checks whether each answer quote really appears in the source chunk.
8. The dedup module embeds questions and rejects near-duplicates.
9. Verified Q&A pairs are saved in SQLite.
10. The export module writes Alpaca or ShareGPT JSON for fine-tuning.

## 4. Mental Model Of Components

### Frontend

The frontend is a minimal HTML/JavaScript interface. It lets the user upload a PDF, configure the number of questions and variations, start generation, monitor progress, retry failed chunks, and download the final dataset.

### API Layer

FastAPI exposes endpoints for creating jobs, resuming jobs, checking job status, streaming progress, and exporting datasets.

Important endpoints:

- `POST /jobs/`: upload PDF, parse it, create chunks, start background job.
- `POST /jobs/{job_id}/resume`: retry unfinished or failed chunks.
- `GET /jobs/{job_id}`: read job status.
- `GET /jobs/{job_id}/stream`: stream live progress using Server-Sent Events.
- `GET /jobs/{job_id}/export`: download final Alpaca or ShareGPT JSON.

### Ingestion

The ingestion layer extracts document structure from the PDF. It uses PyMuPDF to read text spans, font sizes, bold formatting, page numbers, table regions, and underline drawings.

This matters because a policy handbook is not just plain text. The model needs context like chapter, section, page number, and table content so the generated dataset stays grounded.

### Chunking

The chunker converts extracted PDF blocks into training-ready chunks.

It tries not to split across chapter boundaries, starts new chunks at section headings, keeps chunks under a token budget, and adds overlap when splitting long sections. This gives the LLM enough context without exceeding prompt limits.

### Generation

The generation layer calls an OpenAI-compatible LLM API using `httpx`. It asks the model to generate specific questions and verbatim quote answers from each chunk.

The important point is that the LLM output is treated as a candidate, not as truth.

### Verification

The verifier is the anti-hallucination layer.

It strips the citation header from the answer, checks whether the quote appears exactly in the source chunk, and if needed uses fuzzy matching through RapidFuzz to handle minor whitespace or formatting differences.

If the answer is paraphrased or unsupported, it is rejected and never written to the final dataset.

### Deduplication

The dedup module uses `sentence-transformers` to embed questions. It compares each new question with already accepted questions using cosine similarity.

This prevents the final dataset from being filled with many versions of the same question.

### Worker And Checkpointing

The background worker processes chunks one by one and stores status in SQLite. Each chunk has a status such as `pending`, `in_progress`, `done`, or `failed`.

This makes the pipeline resumable. If the server stops or the LLM provider rate-limits midway, completed chunks do not need to be regenerated.

### Export

The export layer converts verified Q&A rows into common fine-tuning formats:

- Alpaca: `instruction`, `input`, `output`
- ShareGPT: conversation-style human/GPT messages

The citation header is placed inside the model output so the fine-tuned model learns to produce citations at inference time.

## 5. Tech Stack Explanation

### FastAPI

I used FastAPI because the project needed a lightweight backend with file upload, JSON APIs, background task launching, and streaming progress. It is simple to develop with and fits a Python ML/data workflow well.

Interview answer:

> I chose FastAPI because the project is Python-first and API-driven. It gave me clean request handling for uploads, easy JSON endpoints, and async support for long-running LLM calls and SSE progress streaming.

### SQLite + SQLModel

SQLite stores jobs, chunks, and verified Q&A pairs. SQLModel gives typed Python models and ORM-style access.

Interview answer:

> I used SQLite because this was a single-document, single-server pipeline. I did not need Redis, Celery, or a distributed queue. The database also acts as the checkpoint store, so job progress survives interruptions.

### PyMuPDF

PyMuPDF extracts text, tables, page numbers, font sizes, bold text, and drawings from PDFs.

Interview answer:

> I used PyMuPDF because the PDF was not just raw text. I needed structure: page numbers, headings, tables, and formatting signals. Those signals helped preserve section boundaries and produce accurate citations.

### tiktoken

`tiktoken` estimates token counts before sending chunks to the LLM.

Interview answer:

> Chunking by characters is risky because LLM limits are token-based. I used `tiktoken` so chunk sizes align more closely with the model context window.

### httpx

`httpx` is used for async LLM API calls.

Interview answer:

> I used `httpx` because the LLM endpoint is HTTP-based and the pipeline benefits from async I/O. The client also has retry and exponential backoff for rate limits and transient server errors.

### RapidFuzz

RapidFuzz verifies quote faithfulness.

Interview answer:

> The LLM is instructed to copy verbatim, but prompts are not guarantees. RapidFuzz lets me reject paraphrased answers while still tolerating minor PDF extraction differences like whitespace or hyphenation.

### sentence-transformers

`sentence-transformers` embeds questions for semantic deduplication.

Interview answer:

> Exact string matching would miss paraphrased duplicates. Embeddings let me catch questions that mean the same thing even if they use different words.

### Server-Sent Events

SSE streams progress from backend to frontend.

Interview answer:

> SSE was enough because progress updates are one-way: server to browser. I did not need the complexity of WebSockets.

## 6. Key Design Decisions

### Why not trust the LLM directly?

Because LLMs can follow the instruction most of the time but still occasionally paraphrase, omit details, or invent facts. In a fine-tuning dataset, even a small percentage of bad examples can teach the model bad behavior.

Strong answer:

> I treated the LLM as a generator, not a source of truth. It proposes candidate Q&A pairs, but deterministic verification decides whether the answer is allowed into the dataset.

### Why put citations inside the answer text?

During fine-tuning, the model learns from the output text. If citations are stored only as metadata, the model never learns to generate them.

Strong answer:

> The goal was not just to store citations for humans. The goal was to train the model to produce citation-backed answers. So the citation header had to be part of the `output` field itself.

### Why rebuild citation headers from DB metadata?

The LLM might produce a correct quote but an incorrect citation header. The database already knows the true chapter, section, and page range for each chunk.

Strong answer:

> I did not trust the LLM's citation header. After quote verification, I reconstructed the citation from trusted chunk metadata in the database.

### Why SQLite instead of Celery/Redis?

The project scope was one uploaded policy document, one local server, and a manageable number of chunks. SQLite was enough for persistence and resumability.

Strong answer:

> I avoided unnecessary infrastructure. SQLite gave me persistence, checkpoints, and simple querying. If the project needed many concurrent users or distributed workers, I would move to Postgres plus a real queue.

### Why semantic deduplication?

Fine-tuning benefits from diversity. Many near-identical questions would make the dataset repetitive.

Strong answer:

> I wanted multiple phrasings, but not noise. Semantic embeddings helped reject questions that were too similar while keeping genuinely different questions.

### Why chunk by sections and not fixed pages?

Policy documents are structured by chapters and sections. Fixed-size page chunks can split ideas awkwardly.

Strong answer:

> Section-aware chunking preserves the meaning and citation context. Token limits still matter, but structure comes first.

## 7. Data/AI Value

This project is valuable because dataset quality is one of the biggest factors in fine-tuning success.

The system improves data quality through:

- Grounded answers: every answer must come from the source PDF.
- Verbatim verification: paraphrases and hallucinations are rejected.
- Citation learning: the model sees citations in the training output.
- Semantic diversity: duplicate questions are filtered.
- Format compatibility: output supports Alpaca and ShareGPT.
- Resumability: long-running LLM generation can recover from failures.

Interview phrase:

> The project is less about simply calling an LLM and more about building the quality-control system around it.

## 8. Example Output

Alpaca format:

```json
{
  "instruction": "How often must a formal review of user access rights be carried out?",
  "input": "",
  "output": "[Chapter: 6. User Access Management | Section: Application Access Control | Pages: 9-9]\n6.3.7 A formal review of user access rights shall be carried out every six months."
}
```

How to explain it:

> The `instruction` is the generated user question. The `input` is empty because the question is self-contained. The `output` contains both the citation and the verified answer, because that is exactly what we want the fine-tuned model to learn to generate.

## 9. Likely Interview Questions And Answers

### Q1. Tell me about your internship project.

I built an IFT dataset builder for company policy documents. It takes a PDF handbook and converts it into a fine-tuning dataset made of verified question-answer pairs. The pipeline extracts structured content, chunks it by section, uses an LLM to generate candidate Q&A pairs, verifies that every answer is a verbatim quote from the source, removes semantically duplicate questions, and exports the dataset in Alpaca or ShareGPT format.

The main challenge was making LLM-generated data reliable. I solved that by treating the LLM output as a candidate and adding deterministic verification before anything enters the final dataset.

### Q2. What problem were you solving?

The problem was that company policy documents contain useful domain knowledge, but they are not directly usable for instruction fine-tuning. Manually creating hundreds of grounded Q&A examples would be slow and error-prone.

My system automates the creation of those examples while preserving traceability through citations and quote verification.

### Q3. What is instruction fine-tuning?

Instruction fine-tuning is the process of training a language model on examples of instructions and ideal responses, so it learns to follow similar instructions at inference time.

In this project, each training example has a user-style question and a grounded answer copied from the policy document.

### Q4. Why LoRA or QLoRA?

LoRA and QLoRA are parameter-efficient fine-tuning methods. Instead of updating all model weights, they train small adapter matrices. This makes fine-tuning cheaper and more practical on limited hardware.

This project creates the dataset that could be used by LoRA/QLoRA training frameworks.

### Q5. How did you prevent hallucinations?

I used a two-layer approach. First, the prompt tells the LLM to copy answers verbatim from the source chunk. Second, and more importantly, the verifier checks whether the returned answer actually appears in the original chunk.

If the quote is not found through exact or fuzzy matching, the pair is rejected.

### Q6. Why use fuzzy matching if answers must be verbatim?

PDF extraction sometimes introduces small formatting differences: extra spaces, line breaks, hyphenation, or minor OCR-like artifacts. Fuzzy matching lets the system tolerate those harmless differences while still rejecting real paraphrases.

### Q7. How does deduplication work?

Each question is converted into an embedding using `sentence-transformers`. Since the embeddings are normalized, cosine similarity can be computed with a dot product. If a new question is too similar to an already accepted question, it is skipped.

This keeps the dataset diverse and avoids overrepresenting the same concept.

### Q8. How does resumability work?

The SQLite database stores the status of every chunk. A chunk can be `pending`, `in_progress`, `done`, or `failed`. When a job resumes, the runner processes every chunk that is not already `done`.

That means completed chunks are not repeated, and failed chunks can be retried.

### Q9. Why did you use SSE?

The frontend only needed one-way progress updates from the server. SSE is simpler than WebSockets and works well for streaming job status like completed chunks, failed chunks, and percentage progress.

### Q10. What was the hardest part?

The hardest part was reliability. It is easy to call an LLM and generate text, but harder to ensure the generated data is safe for fine-tuning. I had to add verification, deduplication, checkpointing, retry handling, and citation reconstruction so the final dataset would be trustworthy.

### Q11. What would you improve next?

I would add a human review interface for borderline examples, better evaluation metrics for final dataset quality, and a production-grade queue if the system needed to support multiple users or many documents concurrently. I would also tune the deduplication threshold on real policy data.

### Q12. What did you learn?

I learned that AI systems need strong non-AI guardrails. The LLM is powerful for generation, but reliability comes from the surrounding system: data extraction, verification, persistence, observability, and evaluation.

## 10. Strengths To Emphasize To UBS

For a Data/AI internship, emphasize these:

- You understand LLMs as probabilistic systems.
- You know prompts alone are not enough.
- You built deterministic validation around model output.
- You designed for dataset quality, not just generation volume.
- You used embeddings for semantic similarity.
- You understand fine-tuning data formats.
- You considered failure modes like rate limits and interrupted jobs.
- You made practical engineering tradeoffs instead of overengineering.

## 11. Weaknesses Or Limitations To Explain Honestly

Do not hide limitations. Explain them maturely.

Potential limitations:

- The PDF parser is tuned to the document structure and should be revalidated on new PDF formats.
- SQLite works for one-user/local usage, but not for high-concurrency production.
- Deduplication threshold needs calibration on real data.
- LLM provider rate limits can interrupt generation.
- A human review step would improve quality before production use.

Good framing:

> I deliberately kept the first version simple and reliable for the expected scope. I also documented where the design would need to evolve for production scale.

## 12. Deep-Dive Follow-Up Answers

### If asked about database schema

The main tables are:

- `Job`: one uploaded PDF processing run.
- `Chunk`: a section/page-aware piece of the source document.
- `QAPair`: a verified question-answer pair linked to a job and chunk.

The important design is that chunks store source metadata like chapter, section, start page, end page, text, status, and retry count. That makes both citation generation and resumability straightforward.

### If asked about failure handling

The LLM client retries rate limits and 5xx errors with exponential backoff. At the chunk level, failures are recorded in the database, and the job can be resumed later.

### If asked about scaling

For the current scope, pairwise embedding comparison is fine because the expected dataset is hundreds of examples, not millions. If this scaled to many documents or large corpora, I would consider vector indexing with FAISS or a vector database.

### If asked about security or compliance

For a company environment, I would avoid sending sensitive documents to an unapproved external LLM provider. The app is already configurable for any OpenAI-compatible endpoint, so the provider could be swapped to an approved internal endpoint or private deployment.

### If asked about evaluation

Current validation includes unit/smoke tests for quote verification, deduplication, chunking, and export format. For production, I would add human sampling, citation accuracy checks, duplicate-rate metrics, and downstream fine-tuning evaluation.

## 13. STAR Story

Situation:

Company policy handbooks contain valuable domain knowledge, but they are not directly usable for fine-tuning a language model.

Task:

Build a tool that converts a PDF policy document into a reliable instruction fine-tuning dataset with grounded, citation-backed answers.

Action:

I built a FastAPI pipeline that parses the PDF, chunks content by document structure, generates candidate Q&A pairs with an LLM, verifies quote faithfulness with exact and fuzzy matching, removes semantically duplicate questions with embeddings, stores progress in SQLite, streams progress to the frontend, and exports Alpaca/ShareGPT JSON.

Result:

The final system can process a policy PDF end-to-end and produce a verified dataset suitable for LoRA/QLoRA fine-tuning, while handling interruptions, rate limits, and data-quality risks.

## 14. Final Memory Hooks

Remember this simple phrase:

Generate, verify, deduplicate, export.

Remember this architecture:

PDF -> chunks -> LLM candidates -> verified Q&A -> fine-tuning JSON

Remember this core insight:

The LLM creates possibilities; the pipeline enforces trust.

