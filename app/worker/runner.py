"""
Day 2 — Person B
Background Job Runner + Checkpoint Helpers
==========================================
Single asyncio task that drives the full generation pipeline for one job.

INTERRUPTION TOLERANCE:
    The SQLite DB is the checkpoint store. Each chunk row has a status field:
        pending      → not yet started
        in_progress  → started but not finished (e.g. server died mid-chunk)
        done         → successfully completed, QA pairs written to DB
        failed       → errored out, retry_count incremented

    On resume (restart + re-call run_job), _get_pending_chunks returns all
    chunks NOT in 'done' state — including in_progress ones from the crashed
    run. Those are automatically retried. No special resume flag needed.

SSE DECOUPLING:
    This runner only writes to the DB (job.completed_chunks, chunk.status).
    The SSE endpoint (api/sse.py) polls the DB independently on its own timer.
    This means SSE connections can drop/reconnect without affecting the job,
    and the job keeps running even when no browser is connected.

CONCURRENCY:
    asyncio.sleep(0) after each chunk yields to the event loop so SSE
    generator coroutines can send their next event. Without this the
    long-running CPU/IO work would block the event loop and SSE would stall.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from sqlmodel import Session, select

from app.config import get_settings
from app.database import engine
from app.diversity.dedup import encode_question, embedding_to_json, is_duplicate, load_accepted_embeddings
from app.generation.qa_generator import generate_qa_pairs, generate_variations
from app.generation.verifier import verify_qa_pair, autocorrect_quote
from app.models import Chunk, ChunkStatus, Job, JobStatus, QAPair

settings = get_settings()
logger = logging.getLogger(__name__)

MAX_CHUNK_RETRIES = 3


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _get_pending_chunks(job_id: int) -> list[Chunk]:
    """
    Return all chunks for this job that are not yet 'done', ordered by
    chunk_index. This drives both initial processing and resume-after-failure.
    """
    with Session(engine) as s:
        stmt = (
            select(Chunk)
            .where(Chunk.job_id == job_id, Chunk.status != ChunkStatus.DONE)
            .order_by(Chunk.chunk_index)
        )
        return s.exec(stmt).all()


def _mark_chunk(chunk_id: int, status: ChunkStatus, error: str | None = None) -> None:
    """Update chunk status and propagate counters to the parent Job row."""
    with Session(engine) as s:
        chunk = s.get(Chunk, chunk_id)
        chunk.status = status
        if error:
            chunk.error_message = error
        if status == ChunkStatus.FAILED:
            chunk.retry_count += 1
        s.add(chunk)

        # Update parent job counters so SSE can show live progress
        job = s.get(Job, chunk.job_id)
        job.updated_at = datetime.utcnow()
        if status == ChunkStatus.DONE:
            job.completed_chunks += 1
        elif status == ChunkStatus.FAILED:
            job.failed_chunks += 1
        s.add(job)
        s.commit()


def _set_job_status(job_id: int, status: JobStatus) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.status = status
        job.updated_at = datetime.utcnow()
        s.add(job)
        s.commit()


def _save_qa_pair(pair: QAPair) -> None:
    with Session(engine) as s:
        s.add(pair)
        s.commit()


# ── Main job runner ───────────────────────────────────────────────────────────

async def run_job(job_id: int) -> None:
    """
    Entry point — called as an asyncio background task from api/routes.py:
        asyncio.create_task(run_job(job_id))

    Implementation guide:

    Step 0 — Set job status to GENERATING.
        _set_job_status(job_id, JobStatus.GENERATING)

    Step 1 — Load accepted embeddings for dedup (handles resume case where
        some QA pairs already exist from a previous interrupted run).
        accepted_embeddings = load_accepted_embeddings(job_id)

    Step 2 — Get all pending/in_progress chunks.
        chunks = _get_pending_chunks(job_id)

    Step 3 — For each chunk:
        a.  Skip if retry_count >= MAX_CHUNK_RETRIES (log and mark failed).
        b.  _mark_chunk(chunk.id, ChunkStatus.IN_PROGRESS)
        c.  try:
                pairs = await generate_qa_pairs(
                    chunk_text=chunk.text,
                    chapter=chunk.chapter,
                    section=chunk.section_title or "",
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                )

                for base_idx, pair in enumerate(pairs):
                    question = pair["question"]
                    answer   = pair["answer"]

                    # --- Verification ---
                    verified, score = verify_qa_pair(answer, chunk.text)
                    if not verified:
                        # Try autocorrect before discarding
                        corrected = autocorrect_quote(answer, chunk.text)
                        if corrected:
                            answer   = corrected
                            verified = True
                            score    = 100.0
                        else:
                            logger.warning(
                                f"Chunk {chunk.id} Q{base_idx}: quote rejected "
                                f"(score={score:.1f})"
                            )
                            continue

                    # --- Dedup check ---
                    q_emb = encode_question(question)
                    dup, sim = is_duplicate(q_emb, accepted_embeddings)
                    if dup:
                        logger.info(
                            f"Chunk {chunk.id} Q{base_idx}: duplicate "
                            f"(sim={sim:.3f}), skipping"
                        )
                        continue

                    # --- Save base QA pair ---
                    _save_qa_pair(QAPair(
                        job_id=job_id,
                        chunk_id=chunk.id,
                        question=question,
                        base_question_index=base_idx,
                        variation_index=0,
                        answer=answer,
                        quote_verified=verified,
                        chapter=chunk.chapter,
                        section_title=chunk.section_title,
                        start_page=chunk.start_page,
                        end_page=chunk.end_page,
                        question_embedding=embedding_to_json(q_emb),
                    ))
                    accepted_embeddings.append(q_emb)

                    # --- Generate and save variations ---
                    variations = await generate_variations(question)
                    for v_idx, v_question in enumerate(variations, start=1):
                        v_emb = encode_question(v_question)
                        v_dup, v_sim = is_duplicate(v_emb, accepted_embeddings)
                        if v_dup:
                            continue
                        _save_qa_pair(QAPair(
                            job_id=job_id,
                            chunk_id=chunk.id,
                            question=v_question,
                            base_question_index=base_idx,
                            variation_index=v_idx,
                            answer=answer,     # same verified answer
                            quote_verified=True,
                            chapter=chunk.chapter,
                            section_title=chunk.section_title,
                            start_page=chunk.start_page,
                            end_page=chunk.end_page,
                            question_embedding=embedding_to_json(v_emb),
                        ))
                        accepted_embeddings.append(v_emb)

                _mark_chunk(chunk.id, ChunkStatus.DONE)

            except Exception as exc:
                logger.error(f"Chunk {chunk.id} failed: {exc}", exc_info=True)
                _mark_chunk(chunk.id, ChunkStatus.FAILED, error=str(exc))

        d.  await asyncio.sleep(0)  # yield to event loop → SSE can fire

    Step 4 — Determine final job status.
        Reload job from DB. If completed_chunks + failed_chunks == total_chunks:
            If failed_chunks == total_chunks: FAILED
            Else: DONE
        _set_job_status(job_id, final_status)
    """
    raise NotImplementedError
