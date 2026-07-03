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
    document_type: str = Form(...),    # "hr" | "finance"
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """
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
    raise NotImplementedError


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
    raise NotImplementedError
