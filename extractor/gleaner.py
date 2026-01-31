"""
Gleaner (v5)
Location: extractor/gleaner.py
Owner: EXTRACTOR Instance

Iterative gleaning to recover missed entities and claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from schemas.contracts import ExtractedClaim, ExtractedEntity, TextChunk
from extractor.json_utils import parse_json_list, safe_str


@dataclass
class GleaningResult:
    """Result of a gleaning pass."""
    additional_entities: List[ExtractedEntity]
    additional_claims: List[ExtractedClaim]
    iterations_performed: int
    confidence_complete: float


def glean_missing(
    llm_client,
    chunk: TextChunk,
    entities: List[ExtractedEntity],
    claims: List[ExtractedClaim],
    max_iter: int = 1,
    skip_below_tokens: int = 800,
    num_ctx: Optional[int] = None,
) -> GleaningResult:
    """Iteratively glean missed entities and claims from a chunk.

    Parameters
    ----------
    llm_client:
        LLM client implementing the generate() protocol.
    chunk:
        The text chunk to re-examine.
    entities:
        Entities already extracted.
    claims:
        Claims already extracted.
    max_iter:
        Maximum gleaning iterations.
    skip_below_tokens:
        Skip gleaning for chunks with fewer tokens than this threshold.
    num_ctx:
        Optional context window size.

    Returns
    -------
    GleaningResult
        Additional entities/claims found and confidence of completeness.
    """
    # Short-circuit for small chunks
    if chunk.token_count < skip_below_tokens:
        return GleaningResult(
            additional_entities=[],
            additional_claims=[],
            iterations_performed=0,
            confidence_complete=1.0,
        )

    all_new_entities: List[ExtractedEntity] = []
    all_new_claims: List[ExtractedClaim] = []
    confidence = 0.0
    iterations = 0

    # Build working lists that grow with each iteration
    current_entities = list(entities)
    current_claims = list(claims)

    for iteration in range(max_iter):
        iterations = iteration + 1
        prompt = _build_prompt(chunk, current_entities, current_claims)
        raw_response = llm_client.generate(prompt, num_ctx=num_ctx)

        # Check for completion signal
        if "EXTRACTION COMPLETE" in raw_response:
            confidence = 1.0
            break

        # Parse additional items
        new_entities, new_claims = _parse_gleaning_response(raw_response, chunk, current_claims)

        if not new_entities and not new_claims:
            confidence = 0.9
            break

        all_new_entities.extend(new_entities)
        all_new_claims.extend(new_claims)
        current_entities.extend(new_entities)
        current_claims.extend(new_claims)

    return GleaningResult(
        additional_entities=all_new_entities,
        additional_claims=all_new_claims,
        iterations_performed=iterations,
        confidence_complete=confidence,
    )


def _build_prompt(
    chunk: TextChunk,
    entities: List[ExtractedEntity],
    claims: List[ExtractedClaim],
) -> str:
    """Build the gleaning prompt."""
    entity_list = ", ".join(e.surface_form for e in entities) if entities else "(none)"
    claim_list = "\n".join(
        f"  - {c.claim_text}" for c in claims
    ) if claims else "  (none)"

    return (
        "Re-examine the text for any ADDITIONAL entities or claims involving subjects "
        "NOT listed here: " + entity_list + "\n\n"
        "Return ONLY new findings in JSON. If complete, respond: EXTRACTION COMPLETE\n"
        "..."
    )


def _parse_gleaning_response(
    raw: str,
    chunk: TextChunk,
    existing_claims: List[ExtractedClaim],
) -> tuple:
    """Parse gleaning response into (entities, claims) tuple."""
    entities: List[ExtractedEntity] = []
    claims: List[ExtractedClaim] = []

    # Try to parse as a JSON object with entities and claims keys
    try:
        from extractor.json_utils import parse_json
        data = parse_json(raw)
    except ValueError:
        return entities, claims

    # Parse entities
    for item in data.get("entities", []):
        if not isinstance(item, dict):
            continue
        surface_form = safe_str(item.get("surface_form")).strip()
        if not surface_form:
            continue
        char_offset = chunk.text.find(surface_form)
        raw_desc = safe_str(item.get("description")).strip() or None
        entities.append(
            ExtractedEntity(
                surface_form=surface_form,
                entity_type=safe_str(item.get("entity_type"), "UNKNOWN").strip(),
                chunk_id=chunk.chunk_id,
                char_offset=char_offset if char_offset >= 0 else None,
                description=raw_desc,
            )
        )

    # Parse claims
    claim_base = len(existing_claims)
    for i, item in enumerate(data.get("claims", [])):
        if not isinstance(item, dict):
            continue
        claim_text = safe_str(item.get("claim")).strip()
        subject = safe_str(item.get("subject")).strip()
        if not subject or not claim_text:
            continue

        claim_id = f"{chunk.chunk_id}_claim_{claim_base + i:03d}"
        claims.append(
            ExtractedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                subject_entity=subject,
                object_entity=safe_str(item.get("object")).strip(),
                predicate=safe_str(item.get("predicate")).strip(),
                description=safe_str(item.get("description")).strip(),
                chunk_id=chunk.chunk_id,
                confidence=_safe_float(item.get("confidence"), 0.5),
            )
        )

    return entities, claims


def _safe_float(value, default: float = 0.5) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
