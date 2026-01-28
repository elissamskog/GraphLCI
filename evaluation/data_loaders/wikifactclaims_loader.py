"""
WikiFactClaims Dataset Loader
Location: evaluation/data_loaders/wikifactclaims_loader.py
Owner: ROOT Instance

Loads WikiFactClaims rows from a JSONL file. Each row contains a HotpotQA
question, 10 context passages, supporting-fact annotations, and 3 derived
claims with SUPPORTS / REFUTES / NOT_ENOUGH_INFO labels.

The key difference from SciFact: context passages are inline per-row (10 docs),
so indexing happens per-row rather than against a global corpus.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schemas.contracts import RawDocument
from schemas.evaluation_contracts import (
    SciFactVerdict,
    WikiClaimLabel,
    WikiFactClaimsRow,
)

_logger = logging.getLogger(__name__)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "wikifactclaims")

_LABEL_MAP = {
    "SUPPORTS": SciFactVerdict.SUPPORTS,
    "REFUTES": SciFactVerdict.REFUTES,
    "NOT_ENOUGH_INFO": SciFactVerdict.NOT_ENOUGH_INFO,
}


class WikiFactClaimsLoader:
    """Load WikiFactClaims rows from a local JSONL file.

    Expected JSONL schema per line::

        {
          "_id": "...",
          "question": "...",
          "answer": "...",
          "type": "comparison" | "bridge",
          "level": "medium" | "hard",
          "supporting_facts": {"title": [...], "sent_id": [...]},
          "context": {"title": [...], "sentences": [[...], ...]},
          "claims": [
            {"text": "...", "label": "SUPPORTS"},
            ...
          ]
        }
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(_DATA_DIR, "wikifactclaims.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"WikiFactClaims data not found at {path}. "
                "Place wikifactclaims.jsonl in data/raw/wikifactclaims/."
            )
        self._rows: List[WikiFactClaimsRow] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    _logger.warning("Skipping invalid JSON at line %d: %s", line_no, e)
                    continue

                claims_raw = raw.get("claims", [])
                claims = []
                for c in claims_raw:
                    label_str = c.get("label", "NOT_ENOUGH_INFO")
                    label = _LABEL_MAP.get(label_str, SciFactVerdict.NOT_ENOUGH_INFO)
                    claims.append(WikiClaimLabel(text=c["text"], label=label))

                sf = raw.get("supporting_facts", {})
                ctx = raw.get("context", {})

                self._rows.append(WikiFactClaimsRow(
                    row_id=str(raw.get("_id", f"row-{line_no}")),
                    question=raw["question"],
                    answer=raw.get("answer", ""),
                    question_type=raw.get("type", "unknown"),
                    level=raw.get("level", "unknown"),
                    claims=claims,
                    supporting_facts_titles=sf.get("title", []),
                    supporting_facts_sent_ids=sf.get("sent_id", []),
                    context_titles=ctx.get("title", []),
                    context_sentences=ctx.get("sentences", []),
                ))
        _logger.info("Loaded %d WikiFactClaims rows from %s", len(self._rows), path)

    def get_rows(self, limit: Optional[int] = None) -> List[WikiFactClaimsRow]:
        """Return loaded rows, optionally capped at *limit*."""
        if limit is not None:
            return self._rows[:limit]
        return list(self._rows)

    @staticmethod
    def build_corpus_for_row(row: WikiFactClaimsRow) -> List[RawDocument]:
        """Convert the 10 inline context passages into RawDocuments.

        Uses the article title as ``doc_id`` so that planner scope document IDs
        can be compared directly against ``supporting_facts_titles``.
        """
        documents: List[RawDocument] = []
        for title, sentences in zip(row.context_titles, row.context_sentences):
            content = title + "\n\n" + " ".join(sentences)
            documents.append(RawDocument(
                doc_id=title,
                content=content,
                metadata={
                    "source": "wikifactclaims",
                    "type": "wikipedia_passage",
                    "title": title,
                    "sentences": sentences,
                    "row_id": row.row_id,
                },
            ))
        return documents
