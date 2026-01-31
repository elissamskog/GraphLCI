"""
Coreference Resolver (v5)
Location: extractor/coreference_resolver.py
Owner: EXTRACTOR Instance

Resolves coreferences across extracted entities to produce canonical forms.
"""

from __future__ import annotations

import json
from typing import List

from schemas.contracts import CoreferenceGroup, ExtractedEntity
from extractor.json_utils import parse_json_list, safe_str


def resolve_coreferences(
    llm_client,
    entities: List[ExtractedEntity],
) -> List[CoreferenceGroup]:
    """Group co-referent entity mentions into canonical forms.

    Parameters
    ----------
    llm_client:
        LLM client implementing the generate() protocol.
    entities:
        All extracted entities across chunks.

    Returns
    -------
    List[CoreferenceGroup]
        Groups of entity variants mapped to canonical forms.
    """
    # Collect unique surface forms
    unique_forms = list(dict.fromkeys(e.surface_form for e in entities))

    # If <= 3 unique forms, return each as a singleton group (skip LLM)
    if len(unique_forms) <= 3:
        return [
            CoreferenceGroup(canonical_form=form, variants=[form])
            for form in unique_forms
        ]

    prompt = _build_prompt(unique_forms)
    raw_response = llm_client.generate(prompt)
    groups = _parse_response(raw_response)

    # Ensure every surface form appears in some group
    grouped_forms: set = set()
    for g in groups:
        grouped_forms.add(g.canonical_form)
        for v in g.variants:
            grouped_forms.add(v)

    for form in unique_forms:
        if form not in grouped_forms:
            groups.append(CoreferenceGroup(canonical_form=form, variants=[form]))

    # Post-LLM: add bridge groups for substring-overlapping canonicals
    bridge_groups = find_partial_match_groups(groups)
    groups.extend(bridge_groups)

    return groups


def find_partial_match_groups(
    groups: List[CoreferenceGroup],
    min_token_length: int = 3,
) -> List[CoreferenceGroup]:
    """Identify substring-overlap relationships between coreference groups.

    When one group's canonical form is a significant substring of another
    (e.g., "Oxford" inside "University of Oxford"), create a bridge group
    containing both canonical forms. These groups produce SYNONYMY edges
    in the edge builder, connecting otherwise disjoint subgraphs.

    Parameters
    ----------
    groups:
        Existing coreference groups from LLM resolution.
    min_token_length:
        Minimum character length for a canonical form to be considered
        as a substring candidate (avoids spurious matches on short tokens).

    Returns
    -------
    List[CoreferenceGroup]
        Additional bridge groups for substring-overlapping canonicals.
    """
    bridge_groups: List[CoreferenceGroup] = []
    canonicals = [g.canonical_form for g in groups]
    seen_pairs: set = set()

    for i, shorter in enumerate(canonicals):
        if len(shorter) < min_token_length:
            continue
        shorter_lower = shorter.lower()
        for j, longer in enumerate(canonicals):
            if i == j:
                continue
            if len(longer) <= len(shorter):
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            if shorter_lower in longer.lower():
                seen_pairs.add(pair)
                # Create a bridge group linking the two canonicals
                bridge_groups.append(CoreferenceGroup(
                    canonical_form=longer,
                    variants=[longer, shorter],
                ))

    return bridge_groups


def _build_prompt(surface_forms: List[str]) -> str:
    """Build the coreference resolution prompt."""
    forms_json = json.dumps(surface_forms)

    return (
        "Identify which entities in the following list refer to the same "
        "real-world concept or entity. Group co-referent mentions and "
        "select a canonical (preferred) form for each group.\n\n"
        f"Entities: {forms_json}\n\n"
        "Return a JSON array of objects, each with:\n"
        '- "canonical": the preferred/canonical form of the entity\n'
        '- "variants": array of all surface forms that refer to this entity '
        "(including the canonical form)\n\n"
        "Return ONLY the JSON array.\n"
    )


def _parse_response(raw: str) -> List[CoreferenceGroup]:
    """Parse the LLM response into CoreferenceGroup objects."""
    try:
        items = parse_json_list(raw)
    except ValueError:
        return []

    groups: List[CoreferenceGroup] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        canonical = safe_str(item.get("canonical")).strip()
        if not canonical:
            continue
        variants = item.get("variants", [canonical])
        if not isinstance(variants, list):
            variants = [canonical]
        # Ensure canonical is in variants
        if canonical not in variants:
            variants.insert(0, canonical)
        groups.append(
            CoreferenceGroup(
                canonical_form=canonical,
                variants=[str(v) for v in variants],
            )
        )

    return groups
