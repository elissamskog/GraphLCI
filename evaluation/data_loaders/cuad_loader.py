"""
CUAD Dataset Loader
Location: evaluation/data_loaders/cuad_loader.py
Owner: ROOT Instance

Loads contracts from the CUAD (Contract Understanding Atticus Dataset)
via HuggingFace datasets. Uses the SQuAD-format JSON which has one row
per contract (510 total), each with title and paragraphs.
"""

from __future__ import annotations

import sys
import os
from typing import List

# Ensure project root is on sys.path for schemas.contracts
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import load_dataset
from schemas.contracts import RawDocument


_CUAD_JSON = "hf://datasets/theatticusproject/cuad/CUAD_v1/CUAD_v1.json"


class CUADLoader:
    """Load CUAD contracts from the SQuAD-format JSON.

    The JSON has 510 entries, one per contract, each with ``title`` and
    ``paragraphs`` (list of dicts with ``context`` key).  We concatenate
    all paragraph contexts into a single document string.
    """

    def __init__(self) -> None:
        self._dataset = load_dataset(
            "json", data_files=_CUAD_JSON, field="data", split="train",
        )

    def get_contracts(self, limit: int = 10) -> List[RawDocument]:
        """Return up to *limit* contracts as RawDocument instances."""
        documents: List[RawDocument] = []

        for row in self._dataset:
            title = row["title"]
            # Concatenate all paragraph contexts for the full contract text
            paragraphs = row.get("paragraphs", [])
            context_parts = [p["context"] for p in paragraphs if "context" in p]
            full_text = "\n\n".join(context_parts)

            documents.append(
                RawDocument(
                    doc_id=title,
                    content=full_text,
                    metadata={"source": "cuad", "title": title},
                )
            )

            if len(documents) >= limit:
                break

        return documents
