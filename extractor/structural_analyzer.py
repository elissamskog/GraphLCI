"""
Structural Analyzer (v5)
Location: extractor/structural_analyzer.py
Owner: EXTRACTOR Instance

Deterministic pre-processing for structural edge injection.
Uses dependency parsing (spaCy/Stanza) to identify hierarchical relationships
that should be encoded as explicit graph edges, independent of LLM extraction.

This module implements Stage 1 of the three-stage extraction pipeline:
1. Deterministic Pre-processing (THIS MODULE)
2. LLM Claim Extraction (claim_extractor.py)
3. Parsing/Decomposition (extractor_controller.py)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from schemas.contracts import ExtractedEntity, ExtractedClaim, TextChunk

logger = logging.getLogger("knowledgeaudit.extractor.structural")

# Try to import spaCy; gracefully degrade if not available
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = None  # type: ignore


class StructuralAnalyzer:
    """Deterministic structural relationship detector.

    Uses NLP dependency parsing to identify hierarchical relationships
    that should be encoded as explicit edges in the knowledge graph.

    Examples of structural relationships:
    - Compound entities: "University of Oxford" → compound("University", "Oxford")
    - Appositives: "Jane Doe, CEO of Acme" → appos("Jane Doe", "CEO")
    - Genitives: "Oxford's library" → poss("library", "Oxford")
    - Prepositional modifiers: "Exeter College, University of Oxford"
    """

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        """Initialize the structural analyzer.

        Parameters
        ----------
        use_spacy:
            Whether to attempt using spaCy for dependency parsing.
        spacy_model:
            Name of the spaCy model to load (default: en_core_web_sm).
        """
        self._nlp: Optional[Language] = None

        if use_spacy and SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(spacy_model)
                logger.info("Structural analyzer initialized with spaCy model: %s", spacy_model)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. Run: python -m spacy download %s",
                    spacy_model, spacy_model
                )
                logger.info("Falling back to heuristic-based structural analysis")
        elif not SPACY_AVAILABLE:
            logger.info("spaCy not installed. Using heuristic-based structural analysis")

    def extract_structural_bridges(
        self,
        chunk: TextChunk,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedClaim]:
        """Extract structural bridge claims from a text chunk.

        This method identifies hierarchical relationships that should be
        explicitly encoded as edges, such as:
        - "Exeter College, University of Oxford" → part_of edge
        - "John Smith, CEO" → role_at edge (from appositive)

        Parameters
        ----------
        chunk:
            The text chunk being analyzed.
        entities:
            Entities already extracted from this chunk.

        Returns
        -------
        List[ExtractedClaim]
            Structural bridge claims with high confidence.
        """
        if self._nlp is not None:
            return self._extract_with_spacy(chunk, entities)
        else:
            return self._extract_with_heuristics(chunk, entities)

    def _extract_with_spacy(
        self,
        chunk: TextChunk,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedClaim]:
        """Use spaCy dependency parsing to extract structural relationships."""
        if self._nlp is None:
            return []

        doc = self._nlp(chunk.text)
        bridge_claims: List[ExtractedClaim] = []
        bridge_counter = 0

        # Build entity span lookup for fast matching
        entity_spans = {e.surface_form.lower(): e for e in entities}

        # Scan for dependency patterns
        for token in doc:
            # Pattern 1: Appositives (e.g., "Jane Doe, CEO of Acme")
            if token.dep_ == "appos":
                head_text = token.head.text
                appos_text = token.text

                # Check if both are entities
                head_entity = entity_spans.get(head_text.lower())
                appos_entity = entity_spans.get(appos_text.lower())

                if head_entity and appos_entity:
                    claim_text = f"{head_text} is {appos_text}"
                    claim_id = f"{chunk.chunk_id}_struct_{bridge_counter:03d}"

                    bridge_claims.append(
                        ExtractedClaim(
                            claim_id=claim_id,
                            claim_text=claim_text,
                            subject_entity=head_text,
                            object_entity=appos_text,
                            predicate="is_a",
                            description=f"Structural bridge from appositive: {token.dep_}",
                            chunk_id=chunk.chunk_id,
                            confidence=0.95,
                        )
                    )
                    bridge_counter += 1

            # Pattern 2: Prepositional phrases indicating location/hierarchy
            # E.g., "of" in "Exeter College of Oxford"
            if token.dep_ == "prep" and token.text.lower() == "of":
                head_noun = token.head
                obj_noun = [child for child in token.children if child.dep_ == "pobj"]

                if obj_noun:
                    child_text = head_noun.text
                    parent_text = obj_noun[0].text

                    child_entity = entity_spans.get(child_text.lower())
                    parent_entity = entity_spans.get(parent_text.lower())

                    if child_entity and parent_entity:
                        predicate = self._determine_predicate(child_entity.entity_type)
                        claim_text = f"{child_text} is {predicate.replace('_', ' ')} {parent_text}"
                        claim_id = f"{chunk.chunk_id}_struct_{bridge_counter:03d}"

                        bridge_claims.append(
                            ExtractedClaim(
                                claim_id=claim_id,
                                claim_text=claim_text,
                                subject_entity=child_text,
                                object_entity=parent_text,
                                predicate=predicate,
                                description=f"Structural bridge from prepositional phrase: {token.dep_}",
                                chunk_id=chunk.chunk_id,
                                confidence=0.95,
                            )
                        )
                        bridge_counter += 1

        return bridge_claims

    def _extract_with_heuristics(
        self,
        chunk: TextChunk,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedClaim]:
        """Fallback heuristic-based structural extraction (comma rule)."""
        bridge_claims: List[ExtractedClaim] = []
        bridge_counter = 0

        for entity in entities:
            surface = entity.surface_form

            # Heuristic 1: Comma-separated qualified entities
            if "," not in surface:
                continue

            parts = [p.strip() for p in surface.split(",", 1)]
            if len(parts) != 2:
                continue

            child_entity, parent_entity = parts

            # Determine predicate based on entity type
            predicate = self._determine_predicate(entity.entity_type)

            # Generate bridge claim
            claim_text = f"{child_entity} is {predicate.replace('_', ' ')} {parent_entity}"
            claim_id = f"{chunk.chunk_id}_struct_{bridge_counter:03d}"

            bridge_claim = ExtractedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                subject_entity=child_entity,
                object_entity=parent_entity,
                predicate=predicate,
                description=f"Structural bridge from comma heuristic: {surface}",
                chunk_id=chunk.chunk_id,
                confidence=0.95,
            )

            bridge_claims.append(bridge_claim)
            bridge_counter += 1

            # Heuristic 2: 'X of Y' (University of Oxford)
            if " of " in surface:
                parts = surface.split(" of ", 1)
                if len(parts) == 2:
                    child, parent = parts[0].strip(), parts[1].strip()
                    if child and parent:
                        claim_text = f"{child} is part of {parent}"
                        bridge_claims.append(ExtractedClaim(
                            claim_id=f"{chunk.chunk_id}_struct_{bridge_counter:03d}",
                            claim_text=claim_text,
                            subject_entity=child,
                            object_entity=parent,
                            predicate="part_of",
                            description=f"Structural bridge from 'of' heuristic: {surface}",
                            chunk_id=chunk.chunk_id,
                            confidence=0.85,
                        ))
                        bridge_counter += 1

            # Heuristic 3: 'X in Y' (Glass Spire in Manchester)
            if " in " in surface:
                parts = surface.split(" in ", 1)
                if len(parts) == 2:
                    child, parent = parts[0].strip(), parts[1].strip()
                    if child and parent:
                        claim_text = f"{child} is located in {parent}"
                        bridge_claims.append(ExtractedClaim(
                            claim_id=f"{chunk.chunk_id}_struct_{bridge_counter:03d}",
                            claim_text=claim_text,
                            subject_entity=child,
                            object_entity=parent,
                            predicate="located_in",
                            description=f"Structural bridge from 'in' heuristic: {surface}",
                            chunk_id=chunk.chunk_id,
                            confidence=0.85,
                        ))
                        bridge_counter += 1

        return bridge_claims

    @staticmethod
    def _determine_predicate(entity_type: str) -> str:
        """Determine the appropriate predicate based on entity type.

        Parameters
        ----------
        entity_type:
            The entity type (LOCATION, ORGANIZATION, etc.)

        Returns
        -------
        str
            The predicate to use (part_of, located_in, etc.)
        """
        entity_type_upper = entity_type.upper()

        if "LOCATION" in entity_type_upper or "PLACE" in entity_type_upper or "GPE" in entity_type_upper:
            return "located_in"
        elif "ORGANIZATION" in entity_type_upper or "ORG" in entity_type_upper:
            return "part_of"
        elif "PERSON" in entity_type_upper:
            return "affiliated_with"
        else:
            # Default fallback
            return "part_of"


def extract_structural_bridges_batch(
    analyzer: StructuralAnalyzer,
    chunks: List[TextChunk],
    entities_list: List[List[ExtractedEntity]],
) -> List[List[ExtractedClaim]]:
    """Extract structural bridges from multiple chunks.

    Parameters
    ----------
    analyzer:
        Configured StructuralAnalyzer instance.
    chunks:
        Text chunks to analyze.
    entities_list:
        Corresponding entity lists for each chunk.

    Returns
    -------
    List[List[ExtractedClaim]]
        Structural bridge claims for each chunk.
    """
    if len(chunks) != len(entities_list):
        raise ValueError("chunks and entities_list must have same length")

    results = []
    for chunk, entities in zip(chunks, entities_list):
        bridges = analyzer.extract_structural_bridges(chunk, entities)
        results.append(bridges)

    return results
