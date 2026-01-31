"""
MultiHop Test Dataset Loader
Location: evaluation/data_loaders/multihop_loader.py
Owner: ROOT Instance

Loads the custom multi-hop reasoning test dataset designed to test:
1. Multi-hop linkage (entity resolution across paragraphs)
2. Temporal logic (date extraction and comparison)
3. Boundary/NEI (detecting missing information)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schemas.contracts import RawDocument
from schemas.evaluation_contracts import SciFactVerdict

import yaml

_logger = logging.getLogger(__name__)

def _get_configured_path():
    config_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    default_path = os.path.join(_PROJECT_ROOT, "data", "raw", "multihop")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config and 'data' in config and 'multihop_path' in config['data']:
                configured = config['data']['multihop_path']
                if os.path.exists(configured):
                    return configured
                _logger.warning(f"Configured MultiHop path not found: {configured}")
        except Exception as e:
            _logger.warning(f"Failed to read config.yaml: {e}")
            
    return default_path

_DATA_DIR = _get_configured_path()

_LABEL_MAP = {
    "SUPPORTS": SciFactVerdict.SUPPORTS,
    "REFUTES": SciFactVerdict.REFUTES,
    "NOT_ENOUGH_INFO": SciFactVerdict.NOT_ENOUGH_INFO,
}


@dataclass
class MultiHopClaim:
    """A claim with its label and reasoning test metadata."""
    text: str
    label: SciFactVerdict
    reasoning_test: str  # multi_hop_linkage, temporal_logic, boundary_nei
    required_hops: List[str]


@dataclass 
class MultiHopRow:
    """A single test case with context paragraphs and claims."""
    row_id: str
    question: str
    answer: str
    question_type: str  # "multihop"
    level: str
    claims: List[MultiHopClaim]
    supporting_facts_titles: List[str]
    context_titles: List[str]
    context_sentences: List[List[str]]


class MultiHopLoader:
    """Load multi-hop reasoning test cases from a JSON file.
    
    Expected JSON schema::
    
        [
          {
            "_id": "...",
            "question": "...",
            "answer": "...",
            "type": "multihop",
            "level": "hard",
            "supporting_facts": {"title": [...], "sent_id": [...]},
            "context": {"title": [...], "sentences": [[...], ...]},
            "claims": [
              {
                "text": "...",
                "label": "SUPPORTS|REFUTES|NOT_ENOUGH_INFO",
                "reasoning_test": "multi_hop_linkage|temporal_logic|boundary_nei",
                "required_hops": ["hop1", "hop2", ...]
              }
            ]
          }
        ]
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(_DATA_DIR, "multihop_test.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MultiHop test data not found at {path}. "
                "Place multihop_test.json in data/raw/multihop/."
            )
        self._rows: List[MultiHopRow] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            claims_raw = item.get("claims", [])
            claims = []
            for c in claims_raw:
                label_str = c.get("label", "NOT_ENOUGH_INFO")
                label = _LABEL_MAP.get(label_str, SciFactVerdict.NOT_ENOUGH_INFO)
                claims.append(MultiHopClaim(
                    text=c["text"],
                    label=label,
                    reasoning_test=c.get("reasoning_test", "unknown"),
                    required_hops=c.get("required_hops", []),
                ))

            sf = item.get("supporting_facts", {})
            ctx = item.get("context", {})

            self._rows.append(MultiHopRow(
                row_id=str(item.get("_id", f"multihop-{len(self._rows)+1}")),
                question=item["question"],
                answer=item.get("answer", ""),
                question_type=item.get("type", "multihop"),
                level=item.get("level", "hard"),
                claims=claims,
                supporting_facts_titles=sf.get("title", []),
                context_titles=ctx.get("title", []),
                context_sentences=ctx.get("sentences", []),
            ))
        _logger.info("Loaded %d MultiHop test rows from %s", len(self._rows), path)

    def get_rows(self, limit: Optional[int] = None) -> List[MultiHopRow]:
        """Return loaded rows, optionally capped at *limit*."""
        if limit is not None:
            return self._rows[:limit]
        return list(self._rows)

    @staticmethod
    def build_corpus_for_row(row: MultiHopRow) -> List[RawDocument]:
        """Convert context paragraphs into RawDocuments."""
        documents: List[RawDocument] = []
        for title, sentences in zip(row.context_titles, row.context_sentences):
            content = title + "\n\n" + " ".join(sentences)
            documents.append(RawDocument(
                doc_id=title,
                content=content,
                metadata={
                    "source": "multihop_test",
                    "type": "test_paragraph",
                    "title": title,
                    "sentences": sentences,
                    "row_id": row.row_id,
                },
            ))
        return documents
