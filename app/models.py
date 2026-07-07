from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING    = "pending"
    PARSING    = "parsing"
    GENERATING = "generating"
    DONE       = "done"
    FAILED     = "failed"


class ChunkStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    FAILED      = "failed"


# ── Tables ───────────────────────────────────────────────────────────────────

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_name: str                        # original filename, e.g. "im_policy.pdf"
    document_type: str                        # defaulted server-side (routes.py); no longer user-facing
    status: JobStatus = JobStatus.PENDING
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    # Per-job dataset size overrides — None means "use settings default"
    # (see config.py: n_questions_per_chunk=5, m_variations_per_question=3)
    n_questions_per_chunk: Optional[int] = None
    m_variations_per_question: Optional[int] = None


class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)

    # Source location — drives citation accuracy
    chapter: str
    section_title: Optional[str] = None
    start_page: int
    end_page: int
    chunk_index: int                          # position in document order
    chunk_type: str = "prose"                 # "prose" | "table"

    # Raw text of this chunk (what the LLM sees + what quotes are verified against)
    text: str

    # Checkpoint state — the DB row IS the checkpoint
    status: ChunkStatus = ChunkStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None


class QAPair(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    chunk_id: int = Field(foreign_key="chunk.id", index=True)

    question: str
    base_question_index: int                  # which of the N base questions (0-indexed)
    variation_index: int = 0                  # 0 = original, 1..M = variations

    # Answer — MUST be a verified verbatim quote from the source chunk
    answer: str
    quote_verified: bool = False              # True only after verifier passes

    # Citation metadata (also embedded inside answer text for training signal)
    chapter: str
    section_title: Optional[str] = None
    start_page: int
    end_page: int

    # Stored as JSON-serialised list[float] — used for dedup across runs
    question_embedding: Optional[str] = None
