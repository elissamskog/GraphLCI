"""
Claim Extractor (v5)
Location: extractor/claim_extractor.py
Owner: EXTRACTOR Instance

Extracts structured claims (subject-predicate-object) from text chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.contracts import ExtractedClaim, ExtractedEntity, TextChunk
from extractor.json_utils import parse_json_list, safe_str


def _safe_float(value, default: float = 0.5) -> float:
    """Safely convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_claims(
    llm_client,
    chunk: TextChunk,
    entities: List[ExtractedEntity],
    schema: Optional[Dict[str, Any]] = None,
) -> List[ExtractedClaim]:
    """Extract claims from a text chunk given known entities.

    Parameters
    ----------
    llm_client:
        LLM client implementing the generate() protocol.
    chunk:
        The text chunk to extract claims from.
    entities:
        Entities already extracted from this chunk.
    schema:
        Optional claim-type schema to guide extraction.

    Returns
    -------
    List[ExtractedClaim]
        Structured claims with subject, predicate, object, and confidence.
    """
    prompt = _build_prompt(chunk, entities, schema)
    raw_response = llm_client.generate(prompt)
    return _parse_response(raw_response, chunk)


def _build_prompt(
    chunk: TextChunk,
    entities: List[ExtractedEntity],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the OpenIE prompt for claim extraction."""
    entity_list = ", ".join(e.surface_form for e in entities) if entities else "(none)"

    schema_guidance = ""
    if schema:
        entity_types = schema.get("entity_types", [])
        if entity_types and isinstance(entity_types, list):
            type_names = [
                et.get("name", "")
                for et in entity_types
                if isinstance(et, dict) and et.get("name")
            ]
            if type_names:
                schema_guidance = (
                    f"\nFocus on claims involving these entity types: "
                    f"{', '.join(type_names)}\n"
                )
        else:
            # Fallback: flat {k: v} format for backward compatibility
            claim_types = ", ".join(str(v) for v in schema.values())
            schema_guidance = f"\nFocus on claims of type: {claim_types}\n"

    # ENHANCEMENT (H4): The prompt currently requests generic factual claims.
    # To produce more specific claims for downstream reasoning, it could
    # explicitly request temporal, quantitative, and comparative claim types
    # (e.g., "identify date comparisons, monetary thresholds, and cross-party
    # obligations").  See docs/investigations/EXTRACTOR_INVESTIGATION.md §H4.
    return (
        "Extract factual claims from the following text. Each claim should "
        "be a structured assertion with a subject, predicate, object, and "
        "description.\n\n"
        f"Known entities in this text: {entity_list}\n"
        f"{schema_guidance}\n"
        "Return a JSON array of objects, each with:\n"
        '- "claim": the claim text as a complete sentence\n'
        '- "subject": the subject entity of the claim\n'
        '- "object": the object entity of the claim\n'
        '- "predicate": the relationship/action between subject and object\n'
        '- "description": a brief description of the claim\n'
        '- "confidence": a float 0.0-1.0 indicating extraction confidence\n\n'
        "EXAMPLE:\n"
        'Text: "The Vendor\'s liability shall not exceed $1,000,000."\n'
        "Entities: Vendor, $1,000,000\n"
        "Output:\n"
        '[{"claim": "Vendor liability is capped at $1,000,000", '
        '"subject": "Vendor", "object": "$1,000,000", '
        '"predicate": "liability_capped_at", '
        '"description": "Contractual liability limitation", '
        '"confidence": 0.95}]\n\n'
        "Return ONLY the JSON array.\n\n"
        f"--- TEXT (chunk {chunk.chunk_id}) ---\n"
        f"{chunk.text}\n"
        "--- END TEXT ---"
    )


def _parse_response(raw: str, chunk: TextChunk) -> List[ExtractedClaim]:
    """Parse the LLM response into ExtractedClaim objects."""
    try:
        items = parse_json_list(raw)
    except ValueError:
        return []

    claims: List[ExtractedClaim] = []
    for i, data in enumerate(items):
        if not isinstance(data, dict):
            continue

        claim_text = safe_str(data.get("claim")).strip()
        subject_entity = safe_str(data.get("subject")).strip()

        # Skip claims with empty subject or claim text
        if not subject_entity or not claim_text:
            continue

        claim_id = f"{chunk.chunk_id}_claim_{i:03d}"
        claims.append(
            ExtractedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                subject_entity=subject_entity,
                object_entity=safe_str(data.get("object")).strip(),
                predicate=safe_str(data.get("predicate")).strip(),
                description=safe_str(data.get("description")).strip(),
                chunk_id=chunk.chunk_id,
                confidence=_safe_float(data.get("confidence"), 0.5),
            )
        )

    return claims
