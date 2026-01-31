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
    num_ctx: Optional[int] = None,
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
    num_ctx:
        Optional context window size.

    Returns
    -------
    List[ExtractedClaim]
        Structured claims with subject, predicate, object, and confidence.
    """
    prompt = _build_prompt(chunk, entities, schema)
    raw_response = llm_client.generate(prompt, num_ctx=num_ctx)
    return _parse_response(raw_response, chunk)


def extract_claims_batch(
    llm_client,
    chunks: List[TextChunk],
    entities_list: List[List[ExtractedEntity]],
    schema: Optional[Dict[str, Any]] = None,
    num_ctx: Optional[int] = None,
) -> List[List[ExtractedClaim]]:
    """Extract claims from multiple chunks in parallel."""
    if len(chunks) != len(entities_list):
        raise ValueError("chunks and entities_list must have same length")
        
    prompts = []
    for chunk, entities in zip(chunks, entities_list):
        prompts.append(_build_prompt(chunk, entities, schema))
        
    raw_responses = llm_client.batch_generate(prompts, num_ctx=num_ctx)
    
    results = []
    for raw, chunk in zip(raw_responses, chunks):
        results.append(_parse_response(raw, chunk))
    return results


def _build_prompt(
    chunk: TextChunk,
    entities: List[ExtractedEntity],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the auditing-focused OpenIE prompt for claim extraction."""
    entity_list = ", ".join(e.surface_form for e in entities) if entities else "(none)"

    # Domain-agnostic guidance covering all claim types
    audit_focus = (
        "Focus on extracting:\n"
        "1. TEMPORAL: Dates, durations, sequences (when something happened)\n"
        "2. RELATIONAL: Connections between entities (X married Y, X works at Y)\n"
        "3. ATTRIBUTIVE: Properties and characteristics (X is a Y, X was born in Y)\n"
        "4. QUANTITATIVE: Numbers, amounts, thresholds\n"
        "5. CAUSAL: Causes, effects, consequences\n"
        "6. SHARED CONTEXT: If multiple entities share a location or context (e.g. 'A is in the city where B is'), extract an explicit link between A and B (e.g. 'A is in the same city as B').\n\n"
        "Extract atomic, verifiable claims focusing on high-fidelity temporal and causal relationships."
    )

    return (
        "You are an evidence extraction engine. Extract ALL factual claims from the text.\n"
        "Each claim must be an atomic, standalone assertion that can be verified independently.\n\n"
        f"KNOWN ENTITIES: {entity_list}\n"
        f"{audit_focus}\n\n"
        "Return a JSON array of objects with:\n"
        '- "claim": The full factual assertion as a standalone sentence.\n'
        '- "subject": The primary entity.\n'
        '- "object": The related entity, value, or attribute.\n'
        '- "predicate": The relationship type (e.g., "founded_in", "employed_at", "is_a", "located_in").\n'
        '- "confidence": Float 0.0-1.0.\n\n'
        "EXAMPLE:\n"
        'Text: "Acme Corp was founded in 1985 by Jane Doe, who later became its CEO."\n'
        "Output:\n"
        "[\n"
        '  {"claim": "Acme Corp was founded in 1985", "subject": "Acme Corp", "object": "1985", "predicate": "founded_in", "confidence": 0.99},\n'
        '  {"claim": "Jane Doe founded Acme Corp", "subject": "Jane Doe", "object": "Acme Corp", "predicate": "founded", "confidence": 0.99},\n'
        '  {"claim": "Jane Doe became CEO of Acme Corp", "subject": "Jane Doe", "object": "Acme Corp", "predicate": "became_ceo_of", "confidence": 0.98}\n'
        "]\n\n"
        "Return ONLY the JSON array with ALL claims found in the text.\n\n"
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
