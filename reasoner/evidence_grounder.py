"""
Evidence Grounder (v5)
Location: reasoner/evidence_grounder.py
Owner: REASONER Instance

Grounds claims and evidence back to source documents and chunks.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from schemas.contracts import (
    GroundedEvidence,
    RawDocument,
    TextChunk,
)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def find_quote_in_documents(
    quote: str,
    documents: Dict[str, RawDocument],
) -> Optional[GroundedEvidence]:
    """Find an exact or near-exact quote in the source documents.

    Parameters
    ----------
    quote:
        The quote text to locate.
    documents:
        Available source documents keyed by doc_id.

    Returns
    -------
    GroundedEvidence or None
        Grounded evidence if found, otherwise None.
    """
    if not quote or not documents:
        return None

    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return None

    # Try exact match first
    for doc_id, doc in documents.items():
        normalized_content = _normalize(doc.content)
        pos = normalized_content.find(normalized_quote)
        if pos != -1:
            return GroundedEvidence(
                claim=quote,
                source_document=doc_id,
                exact_quote=quote,
                char_offset=pos,
            )

    # Try fuzzy match with sliding window (threshold 0.8)
    quote_len = len(normalized_quote)
    best_match: Optional[GroundedEvidence] = None
    best_ratio = 0.0

    for doc_id, doc in documents.items():
        normalized_content = _normalize(doc.content)
        if len(normalized_content) < quote_len:
            continue

        step = max(1, quote_len // 4)
        for i in range(0, len(normalized_content) - quote_len + 1, step):
            window = normalized_content[i:i + quote_len]
            ratio = SequenceMatcher(None, normalized_quote, window).ratio()
            if ratio > best_ratio and ratio >= 0.8:
                best_ratio = ratio
                best_match = GroundedEvidence(
                    claim=quote,
                    source_document=doc_id,
                    exact_quote=quote,
                    char_offset=i,
                )

    return best_match


def ground_evidence(
    claims: List[str],
    documents: Dict[str, RawDocument],
    chunks: Optional[List[TextChunk]] = None,
) -> List[GroundedEvidence]:
    """Ground a list of claims to source documents.

    Parameters
    ----------
    claims:
        Claim strings to ground.
    documents:
        Available source documents keyed by doc_id.
    chunks:
        Optional text chunks for finer-grained grounding.

    Returns
    -------
    List[GroundedEvidence]
        Evidence records with source locations and quotes.
    """
    results: List[GroundedEvidence] = []

    for claim in claims:
        if not claim:
            continue

        # Try document-level grounding first
        grounded = find_quote_in_documents(claim, documents)
        if grounded is not None:
            results.append(grounded)
            continue

        # Try chunk-level grounding
        if chunks:
            normalized_claim = _normalize(claim)
            found_in_chunk = False
            for chunk in chunks:
                normalized_text = _normalize(chunk.text)
                if normalized_claim in normalized_text:
                    results.append(GroundedEvidence(
                        claim=claim,
                        source_document=chunk.source_doc_id,
                        source_chunk=chunk.chunk_id,
                        exact_quote=claim,
                        page_number=chunk.page_number,
                    ))
                    found_in_chunk = True
                    break
            if found_in_chunk:
                continue

        # Ungrounded: record claim with empty exact_quote
        results.append(GroundedEvidence(
            claim=claim,
            source_document="",
            exact_quote="",
        ))

    return results
