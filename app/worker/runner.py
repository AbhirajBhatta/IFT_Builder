"""
Background Job Runner + Checkpoint Helpers
============================================
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
from app.generation.verifier import (
    verify_qa_pair, autocorrect_quote, strip_citation, rebuild_answer, build_citation_header,
)
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
    """Update chunk status and recompute the parent Job's progress counters
    from actual chunk rows (not incrementally). A retried chunk transitions
    failed -> in_progress -> done, so naive +=/-= bookkeeping keyed off "the
    previous status" drifts (the in_progress hop erases the failed signal
    before a success can undo it) — counting real rows can't drift."""
    with Session(engine) as s:
        chunk = s.get(Chunk, chunk_id)
        chunk.status = status
        if error:
            chunk.error_message = error
        if status == ChunkStatus.FAILED:
            chunk.retry_count += 1
        s.add(chunk)

        job = s.get(Job, chunk.job_id)
        job.updated_at = datetime.utcnow()
        job.completed_chunks = len(s.exec(
            select(Chunk).where(Chunk.job_id == job.id, Chunk.status == ChunkStatus.DONE)
        ).all())
        job.failed_chunks = len(s.exec(
            select(Chunk).where(Chunk.job_id == job.id, Chunk.status == ChunkStatus.FAILED)
        ).all())
        if status == ChunkStatus.FAILED and error:
            job.error_message = error  # most recent failure reason
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

    Sets the job to GENERATING, loads any already-accepted question
    embeddings for this job (so dedup is correct across a resume), and
    processes every chunk not yet marked 'done'. For each chunk: generates
    QA pairs, verifies each answer's quote against the chunk text
    (auto-correcting minor drift where possible, otherwise skipping the
    pair), rebuilds the citation header from trusted chunk metadata, checks
    the question against previously-accepted embeddings for duplicates,
    saves the pair, then generates and saves non-duplicate phrasing
    variations of the question against the same verified answer. Chunks
    that error out are marked failed (up to MAX_CHUNK_RETRIES) and
    processing continues with the next chunk. Yields to the event loop
    after each chunk (asyncio.sleep(0)) so the SSE endpoint can report live
    progress. When all chunks are processed, sets the job to DONE (or
    FAILED if every chunk failed).
    """
    _set_job_status(job_id, JobStatus.GENERATING)

    with Session(engine) as s:
        job = s.get(Job, job_id)
        n_questions = job.n_questions_per_chunk
        m_variations = job.m_variations_per_question

    accepted_embeddings = load_accepted_embeddings(job_id)
    chunks = _get_pending_chunks(job_id)

    for chunk in chunks:
        if chunk.retry_count >= MAX_CHUNK_RETRIES:
            logger.warning(f"Chunk {chunk.id} exceeded max retries, marking failed")
            _mark_chunk(chunk.id, ChunkStatus.FAILED, error="Max retries exceeded")
            await asyncio.sleep(0)
            continue

        _mark_chunk(chunk.id, ChunkStatus.IN_PROGRESS)
        try:
            pairs = await generate_qa_pairs(
                chunk_text=chunk.text,
                chapter=chunk.chapter,
                section=chunk.section_title or "",
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                n=n_questions,
            )

            for base_idx, pair in enumerate(pairs):
                question = pair["question"]
                answer = pair["answer"]

                # --- Verification ---
                verified, score = verify_qa_pair(answer, chunk.text)
                if not verified:
                    corrected = autocorrect_quote(answer, chunk.text)
                    if corrected:
                        answer = corrected
                        verified = True
                        score = 100.0
                    else:
                        logger.warning(
                            f"Chunk {chunk.id} Q{base_idx}: quote rejected "
                            f"(score={score:.1f})"
                        )
                        continue

                # --- Rebuild header from trusted chunk metadata ---
                # The LLM sometimes mangles its own citation header (e.g.
                # literally writing "None" for table-derived chunks) even
                # when the quote body is fine — never trust that text.
                quote = strip_citation(answer).strip()
                trusted_header = build_citation_header(
                    chunk.chapter, chunk.section_title, chunk.start_page, chunk.end_page
                )
                answer = rebuild_answer(trusted_header, quote)

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
                variations = await generate_variations(question, m=m_variations)
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
            # Include the exception type (e.g. "ProxyError" vs "RuntimeError")
            # alongside the message — the frontend uses this to tell a quota
            # failure apart from a network/proxy failure.
            _mark_chunk(chunk.id, ChunkStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

        await asyncio.sleep(0)  # yield to event loop → SSE can fire

    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job.completed_chunks + job.failed_chunks == job.total_chunks:
            final_status = (
                JobStatus.FAILED if job.failed_chunks == job.total_chunks
                else JobStatus.DONE
            )
        else:
            final_status = JobStatus.FAILED

    _set_job_status(job_id, final_status)
