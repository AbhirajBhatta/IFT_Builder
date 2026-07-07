"""
Day 3 — Person A
API Routes
==========
POST /jobs/             — ingest PDF, create chunks, launch background job
GET  /jobs/{id}         — job status + progress counters
GET  /jobs/{id}/export  — download the final IFT JSON dataset
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.config import get_settings
from app.database import get_session
from app.export.formatter import export_job
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import extract_toc, parse_pdf
from app.models import Chunk, Job, JobStatus
from app.worker.runner import run_job

settings = get_settings()
router = APIRouter()


@router.post("/jobs/")
async def create_job(
    file: UploadFile = File(...),
    n_questions: int | None = Form(None),
    m_variations: int | None = Form(None),
    session: Session = Depends(get_session),
):
    """
    n_questions / m_variations let the frontend override the dataset-size
    settings (settings.n_questions_per_chunk / m_variations_per_question)
    per job. None means "use the .env default" — threaded through to
    generate_qa_pairs()/generate_variations() by runner.py.

    NOTE: document_type is no longer a user-facing choice — the original
    two-document (hr/finance) design assumed two fixed handbooks, but the
    real target is a single general "add PDF" upload (per user decision).
    document_type is defaulted server-side below so models.py's Job schema
    (Person B's, unchanged) doesn't need a migration.

    Implementation guide:

    1.  Save the uploaded file to data/pdfs/<original_filename>.
        Use shutil.copyfileobj or file.read() + Path.write_bytes().
        Create data/pdfs/ if it doesn't exist.

    2.  Parse + chunk:
            pdf_path = Path(settings.data_output_dir).parent / "pdfs" / file.filename
            blocks   = parse_pdf(pdf_path)
            toc      = extract_toc(pdf_path)
            chunks   = chunk_document(blocks, toc)

    3.  Create the Job row and flush to get its id:
            job = Job(
                document_name=file.filename,
                document_type=document_type,
                status=JobStatus.PARSING,
                total_chunks=len(chunks),
            )
            session.add(job); session.commit(); session.refresh(job)

    4.  Bulk-insert Chunk rows (status=pending):
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

    5.  Fire the background job (do NOT await):
            asyncio.create_task(run_job(job.id))

    6.  Return:
            {"job_id": job.id, "total_chunks": len(chunks)}
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
    Re-launch run_job() for an existing job. runner.py's checkpoint design
    (_get_pending_chunks) already picks up anything not 'done' — including
    chunks that failed transiently (e.g. LLM rate-limit exhaustion) and are
    still under MAX_CHUNK_RETRIES. This just exposes that resume capability,
    which previously had no entry point — POST /jobs/ always started a new
    job with fresh chunks instead of retrying an existing one.
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
    Implementation guide:
    1.  Call export_job(job_id, format) → returns Path to the output JSON.
    2.  Return FileResponse(path, media_type="application/json",
            filename=path.name).
    3.  Raise 404 if the job doesn't exist or has no verified pairs yet.
    """
    try:
        path = export_job(job_id, format)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FileResponse(path, media_type="application/json", filename=path.name)
