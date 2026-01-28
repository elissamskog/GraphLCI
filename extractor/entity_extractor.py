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

    Returns
    -------
    List[ExtractedEntity]
        Extracted entities with types and positions.
    """
    prompt = _build_prompt(chunk, schema)
    raw_response = llm_client.generate(prompt)
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
        "Extract all named entities from the following text.\n\n"
        f"{type_guidance}\n\n"
        "Return a JSON array of objects, each with:\n"
        '- "surface_form": the exact text of the entity mention\n'
        '- "entity_type": the type category\n'
        '- "description": a brief (1-2 sentence) context-grounded description '
        "of what the entity is, drawn from the text\n\n"
        "EXAMPLE:\n"
        'Text: "The Vendor, defined as Acme Corp, shall provide IT services."\n'
        "Output:\n"
        '[{"surface_form": "Vendor", "entity_type": "ACTOR", '
        '"description": "The party providing services, defined as Acme Corp"}]\n\n'
        "Return ONLY the JSON array.\n\n"
        f"--- TEXT (chunk {chunk.chunk_id}) ---\n"
        f"{chunk.text}\n"
        "--- END TEXT ---"
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
        surface_form = safe_str(item.get("surface_form")).strip()
        if not surface_form or surface_form in seen:
            continue
        seen.add(surface_form)

        entity_type = safe_str(item.get("entity_type"), "UNKNOWN").strip()
        char_offset = chunk.text.find(surface_form)
        raw_desc = safe_str(item.get("description")).strip() or None
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
