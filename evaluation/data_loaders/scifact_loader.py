"""
SciFact Dataset Loader
Location: evaluation/data_loaders/scifact_loader.py
Owner: ROOT Instance

Loads scientific claims and abstracts from the AllenAI SciFact dataset
via HuggingFace datasets. Builds per-claim mini-corpora for evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path for schemas.contracts
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schemas.contracts import (
    RawCorpus,
    RawDocument,
    SciFactTestCase,
    SciFactVerdict,
)

_logger = logging.getLogger(__name__)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "scifact")


def _load_corpus_jsonl(path: str) -> Dict[str, dict]:
    """Load corpus from a local JSONL file into a doc_id -> row dict."""
    corpus: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            corpus[doc_id] = row
    return corpus


def _load_claims_jsonl(path: str) -> Dict[str, list]:
    """Load claims from a local JSONL file into a split -> rows dict."""
    by_split: Dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split = row.pop("split", "validation")
            by_split.setdefault(split, []).append(row)
    return by_split


def _load_from_huggingface() -> Tuple[Dict[str, dict], Dict[str, list]]:
    """Download SciFact from HuggingFace and return (corpus_by_id, claims_rows)."""
    from datasets import load_dataset

    claims_ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)

    corpus_by_id: Dict[str, dict] = {}
    for split in corpus_ds:
        for row in corpus_ds[split]:
            doc_id = str(row["doc_id"])
            corpus_by_id[doc_id] = row

    claims_rows: Dict[str, list] = {}
    for split in claims_ds:
        claims_rows[split] = list(claims_ds[split])

    return corpus_by_id, claims_rows


class CorpusStrategy(Enum):
    """Strategy for building per-claim evaluation corpora."""
    CITED_ONLY = "CITED_ONLY"
    CITED_PLUS_DISTRACTORS = "CITED_PLUS_DISTRACTORS"
    FULL_CORPUS = "FULL_CORPUS"


# SciFact label string -> SciFactVerdict
_LABEL_MAP: Dict[str, SciFactVerdict] = {
    "SUPPORT": SciFactVerdict.SUPPORTS,
    "CONTRADICT": SciFactVerdict.REFUTES,
}


class SciFactLoader:
    """Load SciFact claims and corpus.

    Loads from local JSONL files in ``data/raw/scifact/`` when available,
    falling back to HuggingFace download otherwise.

    The ``allenai/scifact`` dataset has two configs:
    - ``claims``: claim_id, claim, evidence (dict), cited_doc_ids
    - ``corpus``: doc_id, title, abstract (list of sentences), structured
    """

    def __init__(self) -> None:
        corpus_path = os.path.join(_DATA_DIR, "corpus.jsonl")
        claims_path = os.path.join(_DATA_DIR, "claims.jsonl")

        if os.path.exists(corpus_path) and os.path.exists(claims_path):
            self._corpus_by_id = _load_corpus_jsonl(corpus_path)
            self._claims_rows = _load_claims_jsonl(claims_path)
        else:
            _logger.warning(
                "Local SciFact data not found in %s. "
                "Falling back to HuggingFace download. "
                "Run: python -m evaluation.data_loaders.download_scifact",
                _DATA_DIR,
            )
            self._corpus_by_id, self._claims_rows = _load_from_huggingface()

    def get_all_abstracts(self) -> List[RawDocument]:
        """Return all corpus abstracts as RawDocument instances."""
        documents: List[RawDocument] = []
        for doc_id, row in self._corpus_by_id.items():
            documents.append(self._row_to_document(doc_id, row))
        return documents

    def get_test_cases(self, limit: Optional[int] = None) -> List[SciFactTestCase]:
        """Return labeled test cases from the validation split.

        Only claims with at least one evidence annotation are included.
        Claims with no evidence labels map to NOT_ENOUGH_INFO.
        """
        # SciFact validation split has labels; test split does not
        split = "validation" if "validation" in self._claims_rows else "train"
        cases: List[SciFactTestCase] = []

        for row in self._claims_rows[split]:
            claim_id = row["id"]
            claim_text = row["claim"]
            evidence_map = row.get("evidence", {}) or {}
            cited_doc_ids = row.get("cited_doc_ids", []) or []

            # Build expected_evidence and determine verdict
            expected_evidence: Dict[str, List[int]] = {}
            verdict = SciFactVerdict.NOT_ENOUGH_INFO

            if evidence_map:
                for doc_id_key, annotations in evidence_map.items():
                    doc_id_str = str(doc_id_key)
                    if not annotations:
                        continue
                    # Each annotation has "sentences" and "label"
                    for ann in annotations:
                        sentences = ann.get("sentences", [])
                        label = ann.get("label", "")
                        if sentences:
                            if doc_id_str not in expected_evidence:
                                expected_evidence[doc_id_str] = []
                            expected_evidence[doc_id_str].extend(sentences)
                        if label in _LABEL_MAP:
                            verdict = _LABEL_MAP[label]

                    # Deduplicate sentence indices
                    if doc_id_str in expected_evidence:
                        expected_evidence[doc_id_str] = sorted(
                            set(expected_evidence[doc_id_str])
                        )

            cases.append(SciFactTestCase(
                claim_id=claim_id,
                claim_text=claim_text,
                expected_verdict=verdict,
                expected_evidence=expected_evidence,
            ))

            if limit and len(cases) >= limit:
                break

        return cases

    def build_corpus_for_claim(
        self,
        case: SciFactTestCase,
        strategy: CorpusStrategy = CorpusStrategy.CITED_PLUS_DISTRACTORS,
        num_distractors: int = 5,
    ) -> RawCorpus:
        """Build a mini-corpus for a single claim.

        Parameters
        ----------
        case:
            The test case to build a corpus for.
        strategy:
            How to select documents for the corpus.
        num_distractors:
            Number of random distractor documents (for CITED_PLUS_DISTRACTORS).
        """
        evidence_doc_ids = set(case.expected_evidence.keys())
        documents: List[RawDocument] = []

        if strategy == CorpusStrategy.FULL_CORPUS:
            documents = self.get_all_abstracts()
        else:
            # Add cited/evidence documents
            for doc_id in evidence_doc_ids:
                row = self._corpus_by_id.get(doc_id)
                if row:
                    documents.append(self._row_to_document(doc_id, row))

            if strategy == CorpusStrategy.CITED_PLUS_DISTRACTORS:
                # Add random distractors
                non_cited = [
                    did for did in self._corpus_by_id
                    if did not in evidence_doc_ids
                ]
                distractor_ids = random.sample(
                    non_cited, min(num_distractors, len(non_cited)),
                )
                for doc_id in distractor_ids:
                    row = self._corpus_by_id[doc_id]
                    documents.append(self._row_to_document(doc_id, row))

        return RawCorpus(
            corpus_id=f"scifact-claim-{case.claim_id}",
            documents=documents,
            ingestion_timestamp=datetime.now(),
        )

    def _row_to_document(self, doc_id: str, row: dict) -> RawDocument:
        """Convert a corpus row to a RawDocument."""
        title = row.get("title", "")
        abstract_sentences = row.get("abstract", []) or []
        content = title + "\n\n" + " ".join(abstract_sentences)

        return RawDocument(
            doc_id=doc_id,
            content=content,
            metadata={
                "source": "scifact",
                "type": "abstract",
                "domain": "SCIENTIFIC",
                "title": title,
                "sentences": abstract_sentences,
            },
        )


def load_scifact(
    limit: Optional[int] = 50,
    strategy: CorpusStrategy = CorpusStrategy.CITED_PLUS_DISTRACTORS,
    num_distractors: int = 5,
) -> Tuple[List[RawCorpus], List[SciFactTestCase]]:
    """Convenience function: load SciFact and build per-claim corpora.

    Returns
    -------
    corpora:
        One RawCorpus per claim (mini-corpus with evidence + distractors).
    test_cases:
        The corresponding SciFactTestCase list.
    """
    loader = SciFactLoader()
    cases = loader.get_test_cases(limit=limit)
    corpora = [
        loader.build_corpus_for_claim(case, strategy, num_distractors)
        for case in cases
    ]
    return corpora, cases
