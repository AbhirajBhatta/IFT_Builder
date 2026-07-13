"""
Deduplication
=============
Embeds questions with sentence-transformers and rejects new questions
that are too similar to already-accepted ones (cosine similarity >= threshold).

Since embeddings are L2-normalised, cosine similarity = dot product, which
is a single matrix multiply — fast enough at this document's scale (tens of
chunks, low hundreds to low thousands of questions) without FAISS or any
clustering.

The model is loaded lazily on first call so startup time is unaffected.
"""
from __future__ import annotations

import json

import numpy as np

from app.config import get_settings

settings = get_settings()

_model = None


# ── Embedding ─────────────────────────────────────────────────────────────────

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: 384-dim, fast, good semantic similarity
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode_question(text: str) -> np.ndarray:
    """Return a 384-dim L2-normalised embedding vector."""
    return _get_model().encode(text, normalize_embeddings=True)


def embedding_to_json(vec: np.ndarray) -> str:
    """Serialise for storage in QAPair.question_embedding (TEXT column)."""
    return json.dumps(vec.tolist())


def json_to_embedding(s: str) -> np.ndarray:
    """Deserialise from QAPair.question_embedding."""
    return np.array(json.loads(s), dtype=np.float32)


# ── Load existing embeddings from DB (used on job resume) ─────────────────────

def load_accepted_embeddings(job_id: int) -> list[np.ndarray]:
    """
    Load embeddings of all already-accepted QA pairs for this job.
    Called at the start of run_job so dedup works correctly on resume.
    """
    from sqlmodel import Session, select
    from app.database import engine
    from app.models import QAPair

    with Session(engine) as s:
        pairs = s.exec(
            select(QAPair).where(QAPair.job_id == job_id)
        ).all()

    return [
        json_to_embedding(p.question_embedding)
        for p in pairs
        if p.question_embedding
    ]


# ── Deduplication check ───────────────────────────────────────────────────────

def is_duplicate(
    candidate: np.ndarray,
    accepted: list[np.ndarray],
) -> tuple[bool, float]:
    """
    Returns (is_duplicate: bool, max_cosine_similarity: float).

    is_duplicate is True when the highest similarity to any accepted
    question is >= settings.dedup_similarity_threshold.

    Returns (False, 0.0) immediately if accepted is empty. Since embeddings
    are L2-normalised, cosine_sim(a, b) = dot(a, b), so this reduces to one
    matrix multiply against all accepted embeddings.
    """
    if not accepted:
        return (False, 0.0)

    matrix = np.stack(accepted)
    scores = matrix @ candidate
    best = float(scores.max())
    return (best >= settings.dedup_similarity_threshold, best)
