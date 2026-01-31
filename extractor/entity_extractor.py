"""
Entity Extractor (v5)
Location: extractor/entity_extractor.py
Owner: EXTRACTOR Instance

Extracts named entities from text chunks using LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.contracts import ExtractedEntity, TextChunk
from extractor.json_utils import parse_json_list, safe_str


def extract_entities(
    llm_client,
    chunk: TextChunk,
    schema: Optional[Dict[str, Any]] = None,
    num_ctx: Optional[int] = None,
) -> List[ExtractedEntity]:
    """Extract named entities from a text chunk.

    Parameters
    ----------
    llm_client:
        LLM client implementing the generate() protocol.
    chunk:
        The text chunk to extract entities from.
    schema:
        Optional entity type schema to guide extraction.
    num_ctx:
        Optional context window size.

    Returns
    -------
    List[ExtractedEntity]
        Extracted entities with types and positions.
    """
    prompt = _build_prompt(chunk, schema)
    raw_response = llm_client.generate(prompt, num_ctx=num_ctx)
    return _parse_response(raw_response, chunk)


def _build_prompt(chunk: TextChunk, schema: Optional[Dict[str, Any]] = None) -> str:
    """Build the NER prompt for the LLM."""
    type_guidance = (
        "Entity types to extract: named concepts, actors (persons, "
        "organizations), artifacts (documents, products, systems), "
        "and quantities (monetary amounts, dates, percentages)."
    )
    if schema:
        entity_types = schema.get("entity_types", [])
        if entity_types and isinstance(entity_types, list):
            lines = []
            for et in entity_types:
                if not isinstance(et, dict):
                    continue
                line = f"- {et.get('name', 'UNKNOWN')}: {et.get('description', '')}"
                indicators = et.get("indicators")
                if indicators:
                    line += f" (indicators: {', '.join(indicators)})"
                lines.append(line)
            if lines:
                type_guidance = f"Entity type definitions:\n" + "\n".join(lines)
        else:
            # Fallback: flat {k: v} format for backward compatibility
            type_defs = "\n".join(
                f"- {k}: {v}" for k, v in schema.items()
            )
            type_guidance = f"Entity type definitions:\n{type_defs}"

    return (
        "Extract ALL named entities for a factual audit. Return ONLY a JSON array.\n"
        "If no entities are found, return [].\n\n"
        f"{type_guidance}\n\n"
        "Return a JSON array with one object per entity:\n"
        '  {"sf": "entity mention", "t": "entity type", "d": "1-sentence context"}\n\n'
        "EXAMPLE:\n"
        'Text: "John Smith (born 1982) founded Acme Corp in New York in 2005."\n'
        'Output: [\n'
        '  {"sf": "John Smith", "t": "PERSON", "d": "Founder born 1982"},\n'
        '  {"sf": "Acme Corp", "t": "ORGANIZATION", "d": "Company founded in 2005"},\n'
        '  {"sf": "New York", "t": "LOCATION", "d": "Location of founding"},\n'
        '  {"sf": "1982", "t": "DATE", "d": "Birth year of John Smith"},\n'
        '  {"sf": "2005", "t": "DATE", "d": "Year Acme Corp was founded"}\n'
        ']\n\n'
        f"--- TEXT ({chunk.chunk_id}) ---\n"
        f"{chunk.text}\n"
    )




def _parse_response(raw: str, chunk: TextChunk) -> List[ExtractedEntity]:
    """Parse the LLM response into ExtractedEntity objects."""
    try:
        items = parse_json_list(raw)
    except ValueError:
        # Fallback: try to parse line-based format (- entity [type])
        return _parse_line_format(raw, chunk)

    seen: set = set()
    entities: List[ExtractedEntity] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Handle both abbreviated (sf, t, d) and full key names
        surface_form = safe_str(item.get("surface_form") or item.get("sf")).strip()
        if not surface_form or surface_form in seen:
            continue
        seen.add(surface_form)

        entity_type = safe_str(item.get("entity_type") or item.get("t"), "UNKNOWN").strip()
        char_offset = chunk.text.find(surface_form)
        raw_desc = safe_str(item.get("description") or item.get("d")).strip() or None
        entities.append(
            ExtractedEntity(
                surface_form=surface_form,
                entity_type=entity_type,
                chunk_id=chunk.chunk_id,
                char_offset=char_offset if char_offset >= 0 else None,
                description=raw_desc,
            )
        )

    return entities



def _parse_line_format(raw: str, chunk: TextChunk) -> List[ExtractedEntity]:
    """Fallback parser for '- entity [type]' line format."""
    import re

    entities: List[ExtractedEntity] = []
    seen: set = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        line = line[1:].strip()
        # Try to match "entity [type]" or "entity (type)"
        match = re.match(r"(.+?)\s*[\[\(](.+?)[\]\)]", line)
        if match:
            surface_form = match.group(1).strip()
            entity_type = match.group(2).strip()
        else:
            surface_form = line
            entity_type = "UNKNOWN"

        if not surface_form or surface_form in seen:
            continue
        seen.add(surface_form)

        char_offset = chunk.text.find(surface_form)
        entities.append(
            ExtractedEntity(
                surface_form=surface_form,
                entity_type=entity_type,
                chunk_id=chunk.chunk_id,
                char_offset=char_offset if char_offset >= 0 else None,
            )
        )

    return entities


def extract_entities_batch(
    llm_client,
    chunks: List[TextChunk],
    schema: Optional[Dict[str, Any]] = None,
    num_ctx: Optional[int] = None,
) -> List[List[ExtractedEntity]]:
    """Extract entities from multiple chunks in one or more LLM calls."""
    # Simple implementation: loop over extract_entities
    # Future optimization: batch prompt for MLX/Ollama
    results = []
    for chunk in chunks:
        results.append(extract_entities(llm_client, chunk, schema, num_ctx))
    return results


def refine_entities_batch(
    llm_client,
    chunks: List[TextChunk],
    batch_entities: List[List[ExtractedEntity]],
    num_ctx: Optional[int] = None,
) -> List[List[ExtractedEntity]]:
    """Schema-Guided Refinement: targeted extraction for specific entity types.
    
    Identifies high-value entities (e.g., CreativeWork, TV Series) and triggers
    a targeted pass to extract missing critical attributes.
    """
    refined_results: List[List[ExtractedEntity]] = [[] for _ in chunks]
    
    # Types that benefit from extra schema-guided extraction
    REFINE_TYPES = {"CREATIVEWORK", "TV SERIES", "MOVIE", "ORGANIZATION", "PERSON"}
    
    for i, (chunk, entities) in enumerate(zip(chunks, batch_entities)):
        to_refine = [e for e in entities if e.entity_type.upper() in REFINE_TYPES]
        if not to_refine:
            continue
            
        # Build a refinement prompt
        refine_prompts = []
        for entity in to_refine:
            refine_prompts.append(
                f"- For the {entity.entity_type} '{entity.surface_form}', extract critical attributes "
                f"(e.g., Network, Creator, Founder, Occupation, Date of Birth)."
            )
            
        prompt = (
            "Refine the following entity extractions from the provided text.\n"
            "For each targeted entity, extract ANY missing factual attributes as NEW entities.\n\n"
            "TARGETED ENTITIES:\n" + "\n".join(refine_prompts) + "\n\n"
            "Return ONLY a JSON array of objects:\n"
            '  {"sf": "attribute value", "t": "attribute type", "d": "context description"}\n\n'
            f"--- TEXT ({chunk.chunk_id}) ---\n"
            f"{chunk.text}\n"
        )
        
        try:
            raw_response = llm_client.generate(prompt, num_ctx=num_ctx)
            extra_entities = _parse_response(raw_response, chunk)
            # Tag them as refined
            for ee in extra_entities:
                if not ee.description:
                    ee.description = "Refined attribute"
                else:
                    ee.description = f"Refined: {ee.description}"
            refined_results[i] = extra_entities
        except Exception:
            # Fallback for LLM failure during refinement
            continue
            
    return refined_results
