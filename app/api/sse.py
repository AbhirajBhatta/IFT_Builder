"""
Day 3 — Person A
SSE Progress Stream
===================
GET /jobs/{id}/stream

Polls the DB every 1.5 seconds and pushes a JSON progress event to the browser.
Fully decoupled from the background runner — the runner writes to the DB,
this endpoint reads from it. They share no in-process state.

SSE event format:
    data: {"status":"generating","completed":12,"total":80,"failed":0,"pct":15}

Connection lifecycle:
    - Browser opens EventSource("/jobs/{id}/stream")
    - Server streams events until status is "done" or "failed", then closes
    - If the browser disconnects mid-job (tab closed, network blip), the job
      keeps running server-side. When the browser reconnects it gets current
      state immediately on the first event.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.database import engine
from app.models import Job, JobStatus

router = APIRouter()

POLL_INTERVAL = 1.5   # seconds between DB reads


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: int):

    async def event_generator():
        while True:
            with Session(engine) as s:
                job = s.get(Job, job_id)

            if not job:
                yield 'data: {"error": "job not found"}\n\n'
                return

            pct = (
                round(job.completed_chunks / job.total_chunks * 100)
                if job.total_chunks > 0 else 0
            )

            payload = json.dumps({
                "job_id":    job_id,
                "status":    job.status,
                "completed": job.completed_chunks,
                "failed":    job.failed_chunks,
                "total":     job.total_chunks,
                "pct":       pct,
            })
            yield f"data: {payload}\n\n"

            # Stop streaming once the job reaches a terminal state
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                return

            await asyncio.sleep(POLL_INTERVAL)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",    # prevent Nginx from buffering SSE
        },
    )
