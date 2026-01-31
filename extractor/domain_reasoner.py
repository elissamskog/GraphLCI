"""
Domain Reasoner
Location: extractor/domain_reasoner.py
Owner: EXTRACTOR Instance

Discovers a domain taxonomy from a collection of document summaries
using an LLM to identify domains and cross-domain relationships.
"""

from __future__ import annotations

import json
import re
from typing import Any, List

from schemas.contracts import (
    DomainRelationship,
    DomainTaxonomy,
    DiscoveredDomain,
    DocumentSummary,
)


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


class DomainReasoner:
    """Discover a domain taxonomy from document summaries."""

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def discover_taxonomy(
        self, summaries: List[DocumentSummary]
    ) -> DomainTaxonomy:
        """Analyze *summaries* and return a ``DomainTaxonomy``."""
        prompt = self._build_prompt(summaries)
        raw_response = self.llm.generate(prompt)
        return self._parse_response(raw_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(summaries: List[DocumentSummary]) -> str:
        serialized = "\n\n".join(
            f"Document {s.doc_id}:\n"
            f"  Summary: {s.summary}\n"
            f"  Themes: {', '.join(s.key_themes)}\n"
            f"  Domain: {s.apparent_domain}\n"
            f"  Type: {s.document_type}\n"
            f"  Key terms: {', '.join(s.key_terms)}"
            for s in summaries
        )

        return (
            "You are an analytical auditing expert. Given the following "
            "document summaries, identify the distinct subject domains present and "
            "how they relate to one another.\n\n"
            "Return a JSON object with exactly these keys:\n"
            '- "domains": a list of objects, each with:\n'
            '    - "domain_id": a short unique identifier (e.g. "dom_media_history")\n'
            '    - "name": human-readable domain name\n'
            '    - "description": 1-2 sentence description\n'
            '    - "indicators": list of terms/phrases that indicate this domain\n'
            '- "cross_domain_relationships": a list of objects, each with:\n'
            '    - "source_domain": domain_id of source\n'
            '    - "target_domain": domain_id of target\n'
            '    - "relationship_type": one of: governs, updates, references, '
            "cites, supports, overlaps\n"
            '    - "reasoning": brief explanation of the relationship\n\n'
            "Return ONLY the JSON object, no additional text.\n\n"
            "--- DOCUMENT SUMMARIES ---\n"
            f"{serialized}\n"
            "--- END SUMMARIES ---"
        )

    @staticmethod
    def _parse_response(raw: str) -> DomainTaxonomy:
        try:
            data = _parse_json(raw)
        except ValueError:
            # Fallback: return empty taxonomy
            return DomainTaxonomy(domains=[], cross_domain_relationships=[])

        domains = [
            DiscoveredDomain(
                domain_id=d.get("domain_id", f"dom_{i}"),
                name=d.get("name", "unknown"),
                description=d.get("description", ""),
                indicators=d.get("indicators", []),
            )
            for i, d in enumerate(data.get("domains", []))
        ]

        relationships = [
            DomainRelationship(
                source_domain=r.get("source_domain", ""),
                target_domain=r.get("target_domain", ""),
                relationship_type=r.get("relationship_type", "references"),
                reasoning=r.get("reasoning", ""),
            )
            for r in data.get("cross_domain_relationships", [])
        ]

        return DomainTaxonomy(
            domains=domains,
            cross_domain_relationships=relationships,
        )
