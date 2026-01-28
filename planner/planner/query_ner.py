"""
Query NER Module
Location: planner/planner/query_ner.py
Owner: PLANNER Instance

Dual-mode named entity recognition for legal queries.
Extracts entities before embedding to improve seed node retrieval
(HippoRAG-style entity-level embedding).

LLM path: few-shot prompt for legal entity types.
Offline path: regex-based noun-phrase chunking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# ============================================================================
# LOCAL DATACLASSES (internal to planner, not in schemas)
# ============================================================================

@dataclass
class NEREntity:
    """A named entity extracted from the query."""
    surface_form: str
    entity_type: str  # LEGAL_CONCEPT, PARTY, DOCUMENT, CLAUSE, QUOTED_TERM
    span: Tuple[int, int]  # (start, end) character offsets in original query


@dataclass
class NERResult:
    """Result of NER extraction on a query."""
    entities: List[NEREntity]
    raw_query: str
    method: str  # "llm" or "offline"


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

QUERY_NER_PROMPT = """\
You are a named entity recognizer for a legal document audit system.

TASK: Extract named entities from the user's query.

ENTITY TYPES:
1. LEGAL_CONCEPT - Legal terms, concepts, or principles (e.g., "liability cap", \
"indemnification", "force majeure", "data protection")
2. PARTY - Organizations, companies, or roles (e.g., "Vendor", "Customer", \
"Data Processor")
3. DOCUMENT - Named documents, regulations, or standards (e.g., "MSA", "DORA", \
"GDPR", "SOW-2024")
4. CLAUSE - Specific clause or section references (e.g., "Section 4.2", \
"Exhibit A", "Article 15")

QUERY: {query_text}

Respond in JSON format:
{{
  "entities": [
    {{"surface_form": "the exact text", "entity_type": "LEGAL_CONCEPT|PARTY|DOCUMENT|CLAUSE"}},
    ...
  ]
}}

If no entities are found, return {{"entities": []}}."""


# ============================================================================
# STOP WORDS (excluded from single-capitalized-word matching)
# ============================================================================

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "not", "and", "but",
    "or", "nor", "for", "yet", "so", "if", "then", "else", "when", "where",
    "how", "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "we", "you", "he", "she", "they", "me", "us", "him",
    "her", "them", "my", "our", "your", "his", "their", "with", "from",
    "into", "about", "between", "through", "during", "before", "after",
    "above", "below", "to", "of", "in", "on", "at", "by", "up", "out",
    "off", "over", "under", "again", "further", "any", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "only", "same", "than", "too", "very", "just", "also", "check",
    "compare", "find", "give", "summarize", "does", "across",
})


# ============================================================================
# QUERY NER CLASS
# ============================================================================

class QueryNER:
    """
    Dual-mode NER for legal queries.

    LLM mode: sends a few-shot prompt and parses JSON response.
    Offline mode: regex-based noun-phrase chunking for testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract(self, query_text: str) -> NERResult:
        """Extract named entities from a query string.

        Parameters
        ----------
        query_text:
            The user's query text.

        Returns
        -------
        NERResult
            Extracted entities with metadata.
        """
        llm_client = self.config.get('llm_client')
        if llm_client is not None:
            return self._extract_with_llm(llm_client, query_text)

        return self._extract_offline(query_text)

    # ------------------------------------------------------------------ LLM
    def _extract_with_llm(self, llm_client: Any, query_text: str) -> NERResult:
        """LLM-based NER extraction."""
        prompt = QUERY_NER_PROMPT.format(query_text=query_text)
        raw_response = llm_client.generate(prompt)
        return self._parse_llm_response(raw_response, query_text)

    def _parse_llm_response(
        self, raw_response: str, query_text: str,
    ) -> NERResult:
        """Parse JSON response from LLM into NERResult."""
        json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if not json_match:
            # Fallback to offline on parse failure
            result = self._extract_offline(query_text)
            result.method = "llm_fallback_offline"
            return result

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            result = self._extract_offline(query_text)
            result.method = "llm_fallback_offline"
            return result

        entities: List[NEREntity] = []
        for entry in data.get("entities", []):
            surface = entry.get("surface_form", "")
            etype = entry.get("entity_type", "LEGAL_CONCEPT")
            # Find span in original query
            start = query_text.find(surface)
            if start == -1:
                # Case-insensitive fallback
                start = query_text.lower().find(surface.lower())
            end = start + len(surface) if start >= 0 else -1
            entities.append(NEREntity(
                surface_form=surface,
                entity_type=etype,
                span=(max(start, 0), max(end, 0)),
            ))

        return NERResult(
            entities=entities,
            raw_query=query_text,
            method="llm",
        )

    # -------------------------------------------------------------- Offline
    def _extract_offline(self, query_text: str) -> NERResult:
        """Regex-based offline NER extraction.

        Rules:
        1. Quoted strings → QUOTED_TERM
        2. Title Case multi-word phrases → LEGAL_CONCEPT
        3. ALL-CAPS tokens (≥2 chars) → DOCUMENT
        4. Single capitalized words (not stop words, not at sentence start) → LEGAL_CONCEPT
        """
        entities: List[NEREntity] = []
        used_spans: List[Tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            for s, e in used_spans:
                if start < e and end > s:
                    return True
            return False

        def _add(surface: str, etype: str, start: int, end: int) -> None:
            if not _overlaps(start, end):
                entities.append(NEREntity(
                    surface_form=surface,
                    entity_type=etype,
                    span=(start, end),
                ))
                used_spans.append((start, end))

        # 1. Quoted strings → QUOTED_TERM
        for m in re.finditer(r'"([^"]+)"', query_text):
            _add(m.group(1), "QUOTED_TERM", m.start(1), m.end(1))

        # 2. ALL-CAPS tokens (≥2 chars) → DOCUMENT
        #    Also handles hyphenated ALL-CAPS like "SOW-2024"
        for m in re.finditer(r'\b([A-Z][A-Z0-9](?:[A-Z0-9]|[-][A-Z0-9])*)\b', query_text):
            token = m.group(1)
            if len(token) >= 2:
                _add(token, "DOCUMENT", m.start(1), m.end(1))

        # 3. Title Case multi-word phrases → LEGAL_CONCEPT
        #    Matches 2+ consecutive capitalized words
        for m in re.finditer(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query_text
        ):
            phrase = m.group(1)
            words = phrase.split()
            # Skip if all words are stop words
            if all(w.lower() in _STOP_WORDS for w in words):
                continue
            _add(phrase, "LEGAL_CONCEPT", m.start(1), m.end(1))

        # 4. Single capitalized words (not stop, not sentence-start) → LEGAL_CONCEPT
        for m in re.finditer(r'\b([A-Z][a-z]{2,})\b', query_text):
            word = m.group(1)
            start = m.start(1)
            if word.lower() in _STOP_WORDS:
                continue
            # Skip sentence-start words (position 0, or after ". " or "? " or "! ")
            if start == 0:
                continue
            preceding = query_text[:start].rstrip()
            if preceding and preceding[-1] in '.?!':
                continue
            _add(word, "LEGAL_CONCEPT", start, m.end(1))

        return NERResult(
            entities=entities,
            raw_query=query_text,
            method="offline",
        )
