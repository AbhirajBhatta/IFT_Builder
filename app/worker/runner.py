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
import logging
from datetime import datetime

from sqlmodel import Session, select

from app.config import get_settings
from app.database import engine
from app.diversity.dedup import (
    encode_question,
    embedding_to_json,
    is_duplicate,
    load_accepted_embeddings,
)
from app.generation.qa_generator import (
    generate_qa_pairs,
    generate_variations,
)
from app.generation.verifier import (
    verify_qa_pair,
    autocorrect_quote,
)
from app.models import Chunk, ChunkStatus, Job, JobStatus, QAPair

settings = get_settings()
logger = logging.getLogger(__name__)

MAX_CHUNK_RETRIES = 3


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def _get_pending_chunks(job_id: int) -> list[Chunk]:
    with Session(engine) as s:
        stmt = (
            select(Chunk)
            .where(
                Chunk.job_id == job_id,
                Chunk.status != ChunkStatus.DONE,
            )
            .order_by(Chunk.chunk_index)
        )
        return s.exec(stmt).all()


def _mark_chunk(
    chunk_id: int,
    status: ChunkStatus,
    error: str | None = None,
) -> None:
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

        if status == ChunkStatus.DONE:
            job.completed_chunks += 1

        elif status == ChunkStatus.FAILED:
            job.failed_chunks += 1

        s.add(job)
        s.commit()


def _set_job_status(job_id: int, status: JobStatus):
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.status = status
        job.updated_at = datetime.utcnow()
        s.add(job)
        s.commit()


def _save_qa_pair(pair: QAPair):
    with Session(engine) as s:
        s.add(pair)
        s.commit()


# ──────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────

async def run_job(job_id: int) -> None:
    """
    Executes the complete QA generation pipeline for a job.
    """

    logger.info("Starting job %s", job_id)

    _set_job_status(job_id, JobStatus.GENERATING)

    accepted_embeddings = load_accepted_embeddings(job_id)

    chunks = _get_pending_chunks(job_id)

    for chunk in chunks:

        if chunk.retry_count >= MAX_CHUNK_RETRIES:
            logger.warning(
                "Chunk %s exceeded max retries.",
                chunk.id,
            )
            _mark_chunk(
                chunk.id,
                ChunkStatus.FAILED,
                "Maximum retries exceeded.",
            )
            continue

        _mark_chunk(chunk.id, ChunkStatus.IN_PROGRESS)

        try:

            pairs = await generate_qa_pairs(
                chunk_text=chunk.text,
                chapter=chunk.chapter,
                section=chunk.section_title or "",
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )

            for base_idx, pair in enumerate(pairs):

                question = pair["question"]
                answer = pair["answer"]

                ##################################################
                # Verification
                ##################################################

                verified, score = verify_qa_pair(
                    answer,
                    chunk.text,
                )

                if not verified:

                    corrected = autocorrect_quote(
                        answer,
                        chunk.text,
                    )

                    if corrected:
                        answer = corrected
                        verified = True
                        score = 100.0

                    else:
                        logger.warning(
                            "Chunk %s Q%s rejected (score %.2f)",
                            chunk.id,
                            base_idx,
                            score,
                        )
                        continue

                ##################################################
                # Deduplication
                ##################################################

                q_embedding = encode_question(question)

                duplicate, similarity = is_duplicate(
                    q_embedding,
                    accepted_embeddings,
                )

                if duplicate:
                    logger.info(
                        "Duplicate skipped (chunk=%s sim=%.3f)",
                        chunk.id,
                        similarity,
                    )
                    continue

                ##################################################
                # Save base question
                ##################################################

                _save_qa_pair(
                    QAPair(
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
                        question_embedding=embedding_to_json(
                            q_embedding
                        ),
                    )
                )

                accepted_embeddings.append(q_embedding)

                ##################################################
                # Variations
                ##################################################

                variations = await generate_variations(question)

                for variation_index, variation in enumerate(
                    variations,
                    start=1,
                ):

                    variation_embedding = encode_question(
                        variation
                    )

                    duplicate, similarity = is_duplicate(
                        variation_embedding,
                        accepted_embeddings,
                    )

                    if duplicate:
                        continue

                    _save_qa_pair(
                        QAPair(
                            job_id=job_id,
                            chunk_id=chunk.id,
                            question=variation,
                            base_question_index=base_idx,
                            variation_index=variation_index,
                            answer=answer,
                            quote_verified=True,
                            chapter=chunk.chapter,
                            section_title=chunk.section_title,
                            start_page=chunk.start_page,
                            end_page=chunk.end_page,
                            question_embedding=embedding_to_json(
                                variation_embedding
                            ),
                        )
                    )

                    accepted_embeddings.append(
                        variation_embedding
                    )

            _mark_chunk(
                chunk.id,
                ChunkStatus.DONE,
            )

        except Exception as exc:

            logger.exception(
                "Chunk %s failed.",
                chunk.id,
            )

            _mark_chunk(
                chunk.id,
                ChunkStatus.FAILED,
                error=str(exc),
            )

        # Allow SSE coroutine to run
        await asyncio.sleep(0)

    ##############################################################
    # Determine final job status
    ##############################################################

    with Session(engine) as s:

        job = s.get(Job, job_id)

        processed = (
            job.completed_chunks +
            job.failed_chunks
        )

        if processed == job.total_chunks:

            if job.failed_chunks == job.total_chunks:
                final_status = JobStatus.FAILED
            else:
                final_status = JobStatus.DONE

        else:
            final_status = JobStatus.GENERATING

    _set_job_status(job_id, final_status)

    logger.info(
        "Job %s finished with status %s",
        job_id,
        final_status,
    )