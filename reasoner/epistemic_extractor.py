"""
Epistemic Extractor (v5)
Location: reasoner/epistemic_extractor.py
Owner: REASONER Instance

Extracts epistemic relationships from LLM responses.
"""

from __future__ import annotations

import json
import re
from typing import List

from schemas.contracts import EpistemicRelationship, InferredRelationship


def extract_relationships(llm_response: str) -> List[InferredRelationship]:
    """Parse LLM response to extract epistemic relationships.

    Parameters
    ----------
    llm_response:
        Raw text response from the LLM.

    Returns
    -------
    List[InferredRelationship]
        Extracted relationships with types, confidence, and reasoning.
    """
    try:
        match = re.search(r'\{[\s\S]*\}', llm_response)
        if not match:
            return []

        parsed = json.loads(match.group())
        raw_relationships = parsed.get("relationships", [])

        results: List[InferredRelationship] = []
        for rel in raw_relationships:
            # Map relationship string to enum, default UNRELATED
            rel_str = rel.get("relationship", "UNRELATED")
            try:
                relationship = EpistemicRelationship(rel_str)
            except ValueError:
                relationship = EpistemicRelationship.UNRELATED

            # Clamp confidence to [0.0, 1.0]
            confidence = float(rel.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            results.append(InferredRelationship(
                source_id=rel.get("source_id", ""),
                target_id=rel.get("target_id", ""),
                relationship=relationship,
                confidence=confidence,
                reasoning=rel.get("reasoning", ""),
                evidence_quote=rel.get("evidence_quote"),
            ))

        return results

    except (json.JSONDecodeError, KeyError, TypeError):
        return []
