"""
API Routes
==========
POST /jobs/                     — ingest PDF, create chunks, launch background job
GET  /jobs/{id}                 — job status + progress counters
GET  /jobs/{id}/export          — download the final IFT JSON dataset
POST /settings/llm-credentials  — hot-swap LLM API key/base URL/model
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session

from app.config import get_settings, update_llm_settings
from app.database import get_session
from app.export.formatter import export_job
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import extract_toc, parse_pdf
from app.models import Chunk, Job, JobStatus
from app.worker.runner import run_job

settings = get_settings()
router = APIRouter()


class LLMCredentialsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


@router.post("/settings/llm-credentials")
async def update_llm_credentials(body: LLMCredentialsUpdate):
    """
    Hot-swaps the LLM API key / base URL / model in place (see
    update_llm_settings in config.py) and persists them to .env. Only
    fields provided are changed. The key and base URL are write-only —
    never echoed back in the response, only the model name is.
    """
    update_llm_settings(api_key=body.api_key, base_url=body.base_url, model=body.model)
    return {"status": "updated", "model": get_settings().llm_model}


@router.post("/jobs/")
async def create_job(
    file: UploadFile = File(...),
    n_questions: int | None = Form(None),
    m_variations: int | None = Form(None),
    session: Session = Depends(get_session),
):
    """
    Accepts an uploaded PDF, parses and chunks it, creates the Job row and
    its Chunk rows, and launches background generation (asyncio.create_task,
    not awaited — the response returns immediately). Returns
    {"job_id": ..., "total_chunks": ...}.

    n_questions / m_variations let the caller override the dataset-size
    settings (settings.n_questions_per_chunk / m_variations_per_question)
    for this job only. None means "use the .env default" — threaded through
    to generate_qa_pairs()/generate_variations() by runner.py.

    document_type is not a user-facing input — the frontend takes a single
    general PDF upload with no document-type selector, so document_type is
    defaulted server-side (below) to keep models.py's Job schema stable.
    """
    pdf_dir = Path(settings.data_output_dir).parent / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / file.filename
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    blocks = parse_pdf(pdf_path)
    toc = extract_toc(pdf_path)
    chunks = chunk_document(blocks, toc)

    job = Job(
        document_name=file.filename,
        document_type="policy",
        status=JobStatus.PARSING,
        total_chunks=len(chunks),
        n_questions_per_chunk=n_questions,
        m_variations_per_question=m_variations,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    for idx, rc in enumerate(chunks):
        session.add(Chunk(
            job_id=job.id,
            chapter=rc.chapter,
            section_title=rc.section_title,
            start_page=rc.start_page,
            end_page=rc.end_page,
            chunk_index=idx,
            chunk_type=rc.chunk_type,
            text=rc.text,
        ))
    session.commit()

    asyncio.create_task(run_job(job.id))

    return {"job_id": job.id, "total_chunks": len(chunks)}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: int, session: Session = Depends(get_session)):
    """
    Re-launches run_job() for an existing job id. runner.py's checkpoint
    design (_get_pending_chunks) picks up anything not yet 'done', including
    chunks that failed transiently (e.g. a rate-limited LLM call) and are
    still under MAX_CHUNK_RETRIES — this endpoint is how that resume
    capability is triggered from the API.
    """
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    asyncio.create_task(run_job(job_id))
    return {"job_id": job_id, "status": "resumed"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, session: Session = Depends(get_session)):
    """Return the Job row as JSON (status, progress counters, timestamps)."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/export")
async def export_dataset(
    job_id: int,
    format: str = Query("alpaca", regex="^(alpaca|sharegpt)$"),
):
    """
    Exports the job's verified QA pairs in the requested format and returns
    the file for download. Raises 404 if the job doesn't exist or has no
    verified pairs yet.
    """
    try:
        path = export_job(job_id, format)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FileResponse(path, media_type="application/json", filename=path.name)
