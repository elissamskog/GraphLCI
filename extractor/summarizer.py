"""
Document Summarizer
Location: extractor/summarizer.py
Owner: EXTRACTOR Instance

Summarizes individual documents using an LLM, producing structured
DocumentSummary objects for downstream taxonomy discovery.
"""

from __future__ import annotations

import json
import re
from typing import Any

from schemas.contracts import DocumentSummary, RawDocument


def _parse_json(raw: str) -> dict:
    """Robustly extract a JSON object from an LLM response.

    Tries three strategies in order:
    1. Direct ``json.loads``
    2. Extract content from markdown ```json ... ``` fences
    3. Extract the first top-level ``{...}`` block via brace matching
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: markdown fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: brace matching for first top-level object
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except (json.JSONDecodeError, TypeError):
                        break

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")


class Summarizer:
    """Summarize a single document into a structured ``DocumentSummary``."""

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def summarize(self, doc: RawDocument) -> DocumentSummary:
        """Produce a ``DocumentSummary`` for *doc* via an LLM call."""
        prompt = self._build_prompt(doc)
        raw_response = self.llm.generate(prompt)
        return self._parse_response(doc.doc_id, raw_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(doc: RawDocument) -> str:
        return (
            "You are an analytical auditing expert. Analyze the following document "
            "and return a JSON object with exactly these keys:\n"
            '- "summary": a 2-3 sentence summary of the document\n'
            '- "key_themes": a list of 3-5 key themes\n'
            '- "apparent_domain": a specific domain label\n'
            '- "key_terms": a list of important terms or definitions found\n'
            '- "document_type": article, book, journal, report, whitepaper, or other\n\n'
            "Return ONLY the JSON object, no additional text.\n\n"
            f"--- DOCUMENT ID: {doc.doc_id} ---\n"
            f"{doc.content}\n"
            "--- END DOCUMENT ---"
        )

    @staticmethod
    def _parse_response(doc_id: str, raw: str) -> DocumentSummary:
        try:
            data = _parse_json(raw)
        except ValueError:
            # Fallback: return minimal summary with raw text
            return DocumentSummary(
                doc_id=doc_id,
                summary=raw[:500] if raw else "Unable to generate summary",
                key_themes=[],
                apparent_domain="unknown",
                key_terms=[],
                document_type="unknown",
            )

        return DocumentSummary(
            doc_id=doc_id,
            summary=data.get("summary", ""),
            key_themes=data.get("key_themes", []),
            apparent_domain=data.get("apparent_domain", "unknown"),
            key_terms=data.get("key_terms", []),
            document_type=data.get("document_type", "unknown"),
        )
