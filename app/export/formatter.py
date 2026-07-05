"""
Day 3 — Person B
IFT Dataset Formatter
=====================
Converts verified QAPair rows into the final JSON training format.

Supported output formats:

  alpaca (default):
    {"instruction": "<question>", "input": "", "output": "<citation + quote>"}
    Used by: Axolotl, LLaMA-Factory, most QLoRA scripts.

  sharegpt:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    Used by: FastChat, some Axolotl configs.

CRITICAL: The citation header [Chapter: ... | Pages: ...] is INSIDE the
"output" / "gpt" field — not sidecar metadata. This is intentional: the
fine-tuned model must learn to emit citations as part of its answer at
inference time. Metadata-only storage would make citations invisible to the
model during training.

Quick test:
    python -m app.export.formatter <job_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlmodel import Session, select

from app.config import get_settings
from app.database import engine
from app.models import QAPair

settings = get_settings()


# ── Format converters ─────────────────────────────────────────────────────────

def _to_alpaca(pair: QAPair) -> dict:
    return {
        "instruction": pair.question,
        "input":       "",
        "output":      pair.answer,   # citation header + verbatim quote
    }


def _to_sharegpt(pair: QAPair) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": pair.question},
            {"from": "gpt",   "value": pair.answer},
        ]
    }


# ── Main export function ──────────────────────────────────────────────────────

def export_job(job_id: int, fmt: str = "alpaca") -> Path:
    """
    Query all verified QAPairs for job_id, format them, write to
    data/output/job_{job_id}_{fmt}.json, and return the Path.

    Implementation guide:
    1.  Open a Session and query:
            select(QAPair).where(
                QAPair.job_id == job_id,
                QAPair.quote_verified == True,
            )
        If result is empty, raise ValueError("No verified pairs for this job").

    2.  Choose converter:
            converters = {"alpaca": _to_alpaca, "sharegpt": _to_sharegpt}
            convert = converters[fmt]

    3.  Build the list:
            data = [convert(p) for p in pairs]

    4.  Write to output file:
            out_dir = Path(settings.data_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"job_{job_id}_{fmt}.json"
            out_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    5.  Print a summary and return out_path.
    """
    raise NotImplementedError


# ── Quick inspection script ───────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage: python -m app.export.formatter <job_id> [alpaca|sharegpt]

    Exports and prints the first 3 records so you can verify the format
    before loading it into your training framework.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m app.export.formatter <job_id> [alpaca|sharegpt]")
        sys.exit(1)

    job_id = int(sys.argv[1])
    fmt    = sys.argv[2] if len(sys.argv) > 2 else "alpaca"

    path = export_job(job_id, fmt)
    print(f"\nExported to: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Total records: {len(data)}")
    print("\n=== First 3 records ===")
    for record in data[:3]:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print()
