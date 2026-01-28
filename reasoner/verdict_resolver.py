"""
Verdict Resolver (v5)
Location: reasoner/verdict_resolver.py
Owner: REASONER Instance

Determines the final verdict type and calculates confidence factors.
"""

from __future__ import annotations

from typing import List

from schemas.contracts import (
    ConfidenceFactors,
    EpistemicRelationship,
    GroundedEvidence,
    InferredRelationship,
    VerdictType,
)

_SUPPORTING = {
    EpistemicRelationship.SUPPORTS,
    EpistemicRelationship.STRONGLY_SUPPORTS,
    EpistemicRelationship.PARTIALLY_SUPPORTS,
    EpistemicRelationship.EXTENDS,
}

_CONTRADICTING = {
    EpistemicRelationship.CONTRADICTS,
    EpistemicRelationship.STRONGLY_CONTRADICTS,
    EpistemicRelationship.PARTIALLY_CONTRADICTS,
    EpistemicRelationship.SUPERSEDES,
}

_STRONG_SUPPORT = {
    EpistemicRelationship.SUPPORTS,
    EpistemicRelationship.STRONGLY_SUPPORTS,
}

_STRONG_CONTRADICT = {
    EpistemicRelationship.CONTRADICTS,
    EpistemicRelationship.STRONGLY_CONTRADICTS,
}

_NEUTRAL = {
    EpistemicRelationship.UNRELATED,
    EpistemicRelationship.INSUFFICIENT_EVIDENCE,
}


def determine_verdict_type(
    relationships: List[InferredRelationship],
    evidence: List[GroundedEvidence],
) -> VerdictType:
    """Determine the verdict type from inferred relationships and evidence.

    Parameters
    ----------
    relationships:
        Epistemic relationships extracted by the REASONER.
    evidence:
        Grounded evidence records.

    Returns
    -------
    VerdictType
        The determined verdict type.
    """
    if not relationships and not evidence:
        return VerdictType.NOT_FOUND

    supporting = sum(1 for r in relationships if r.relationship in _SUPPORTING)
    contradicting = sum(1 for r in relationships if r.relationship in _CONTRADICTING)
    strong_support = sum(1 for r in relationships if r.relationship in _STRONG_SUPPORT)
    strong_contradict = sum(1 for r in relationships if r.relationship in _STRONG_CONTRADICT)

    # All neutral relationships
    if relationships and all(r.relationship in _NEUTRAL for r in relationships):
        return VerdictType.NOT_FOUND

    if contradicting > supporting and strong_contradict >= 2:
        return VerdictType.REFUTED
    if contradicting > supporting and strong_contradict < 2:
        return VerdictType.PARTIALLY_REFUTED
    if supporting > contradicting and strong_support >= 2:
        # Strict Rule: CONFIRMED requires at least one exact quote (C)
        # linked to a structural relationship (A).
        has_linked_structural = any(
            r.relationship in _STRONG_SUPPORT and
            r.evidence_quote and
            any(e.exact_quote == r.evidence_quote and e.source_chunk for e in evidence)
            for r in relationships
        )
        if has_linked_structural:
            return VerdictType.CONFIRMED
        else:
            return VerdictType.PARTIALLY_CONFIRMED

    if supporting > contradicting and strong_support < 2:
        return VerdictType.PARTIALLY_CONFIRMED

    return VerdictType.INCONCLUSIVE


def calculate_confidence(
    relationships: List[InferredRelationship],
    evidence: List[GroundedEvidence],
) -> ConfidenceFactors:
    """Calculate confidence factor breakdown.

    Parameters
    ----------
    relationships:
        Epistemic relationships extracted by the REASONER.
    evidence:
        Grounded evidence records.

    Returns
    -------
    ConfidenceFactors
        Breakdown of overall, evidence_quality, source_agreement, coverage.
    """
    # Evidence quality: average confidence of relationships
    if relationships:
        evidence_quality = sum(r.confidence for r in relationships) / len(relationships)
    else:
        evidence_quality = 0.0

    # Boost if grounded evidence has exact quotes
    if any(e.exact_quote for e in evidence):
        evidence_quality = min(1.0, evidence_quality + 0.1)

    # Source agreement: consistency of direction
    directional = [r for r in relationships if r.relationship not in _NEUTRAL]
    if directional:
        sup = sum(1 for r in directional if r.relationship in _SUPPORTING)
        con = sum(1 for r in directional if r.relationship in _CONTRADICTING)
        total = len(directional)
        source_agreement = max(sup, con) / total
    else:
        source_agreement = 0.0

    # Coverage: fraction of evidence grounded with quotes
    if evidence:
        grounded_with_quotes = sum(1 for e in evidence if e.exact_quote)
        coverage = grounded_with_quotes / len(evidence)
    else:
        coverage = 0.0

    # Overall: weighted average
    overall = 0.4 * evidence_quality + 0.3 * source_agreement + 0.3 * coverage
    overall = max(0.0, min(1.0, overall))

    return ConfidenceFactors(
        overall=overall,
        evidence_quality=evidence_quality,
        source_agreement=source_agreement,
        coverage=coverage,
    )
