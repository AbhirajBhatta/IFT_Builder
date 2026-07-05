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

CRITICAL:
The citation header is embedded INSIDE the model output so the fine-tuned
model learns to produce citations during inference.

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


# ──────────────────────────────────────────────────────────────
# Citation Helper
# ──────────────────────────────────────────────────────────────

def _citation(pair: QAPair) -> str:
    """
    Build a citation header from the QAPair metadata.
    """

    citation = (
        f"[Chapter: {pair.chapter}"
        f" | Pages: {pair.start_page}-{pair.end_page}]"
    )

    if pair.section_title:
        citation += f" [Section: {pair.section_title}]"

    return citation


# ──────────────────────────────────────────────────────────────
# Format converters
# ──────────────────────────────────────────────────────────────

def _to_alpaca(pair: QAPair) -> dict:
    return {
        "instruction": pair.question,
        "input": "",
        "output": f"{_citation(pair)}\n\n{pair.answer}",
    }


def _to_sharegpt(pair: QAPair) -> dict:
    return {
        "conversations": [
            {
                "from": "human",
                "value": pair.question,
            },
            {
                "from": "gpt",
                "value": f"{_citation(pair)}\n\n{pair.answer}",
            },
        ]
    }


# ──────────────────────────────────────────────────────────────
# Main export function
# ──────────────────────────────────────────────────────────────

def export_job(job_id: int, fmt: str = "alpaca") -> Path:
    """
    Export all verified QA pairs for a job.

    Parameters
    ----------
    job_id : int
        ID of the completed job.

    fmt : str
        "alpaca" or "sharegpt"

    Returns
    -------
    Path
        Path to exported JSON file.
    """

    converters = {
        "alpaca": _to_alpaca,
        "sharegpt": _to_sharegpt,
    }

    if fmt not in converters:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            f"Supported formats: {list(converters.keys())}"
        )

    with Session(engine) as session:
        pairs = session.exec(
            select(QAPair).where(
                QAPair.job_id == job_id,
                QAPair.quote_verified == True,
            )
        ).all()

    if not pairs:
        raise ValueError(
            f"No verified QA pairs found for job {job_id}."
        )

    convert = converters[fmt]

    data = [
        convert(pair)
        for pair in pairs
    ]

    out_dir = Path(settings.data_output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / f"job_{job_id}_{fmt}.json"

    out_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("IFT DATASET EXPORT")
    print("=" * 60)
    print(f"Job ID      : {job_id}")
    print(f"Format      : {fmt}")
    print(f"Records     : {len(data)}")
    print(f"Output File : {out_path}")
    print("=" * 60)

    return out_path


# ──────────────────────────────────────────────────────────────
# Quick inspection script
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python -m app.export.formatter <job_id> "
            "[alpaca|sharegpt]"
        )
        sys.exit(1)

    job_id = int(sys.argv[1])

    fmt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "alpaca"
    )

    path = export_job(job_id, fmt)

    print(f"\nExport completed successfully.")
    print(f"Saved to: {path}")

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    print(f"\nTotal records: {len(data)}")

    print("\n========== SAMPLE RECORDS ==========\n")

    for record in data[:3]:
        print(
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
            )
        )
        print()