"""
v5 Ground Truth Fixtures
Location: evaluation/v5_fixtures.py
Owner: ROOT Instance

Labeled test cases for v5 pipeline evaluation. Reuses the same MSA/DORA
document pairs from v4 fixtures but uses v5-specific expectations:
- expected_verdict (VerdictType)
- expected_relationship (EpistemicRelationship)
- expected_lane (LaneType)
- expected_intent (IntentType)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.contracts import (
    EpistemicRelationship,
    IntentType,
    LaneType,
    QueryRequest,
    RawDocument,
    ReasonerOutput,
    VerdictType,
)
from schemas.evaluation_contracts import (
    V5GroundTruthCase,
    V5GroundTruthResult,
)


# ============================================================================
# v5 FIXTURES
# ============================================================================

V5_SYNTH_01 = V5GroundTruthCase(
    case_id="v5-synth-01",
    description="MSA liability cap vs DORA unlimited liability — CONTRADICTS",
    documents=[
        RawDocument(
            doc_id="MSA",
            content=(
                "MASTER SERVICE AGREEMENT\n\n"
                "Section 1. DEFINITIONS\n"
                '"Vendor" means Acme Corp.\n\n'
                "Section 2. LIMITATION OF LIABILITY\n"
                "The Vendor shall not be liable for any indirect, incidental, "
                "or consequential damages. The Vendor's total aggregate "
                "liability shall not exceed $1,000,000."
            ),
            metadata={"type": "contract", "domain": "LEGAL"},
        ),
        RawDocument(
            doc_id="DORA",
            content=(
                "DIGITAL OPERATIONAL RESILIENCE ACT (DORA)\n\n"
                "Article 1. SCOPE\n"
                "This regulation establishes requirements for ICT risk "
                "management applicable to financial entities and their "
                "critical third-party service providers.\n\n"
                "Article 2. LIABILITY REQUIREMENTS\n"
                "Critical service providers shall maintain unlimited liability "
                "for operational failures. No contractual limitation of "
                "liability shall override this requirement."
            ),
            metadata={"type": "regulation", "domain": "REGULATORY"},
        ),
    ],
    query=QueryRequest(
        query_id="v5-synth-01-q",
        query_text="Does the MSA liability cap contradict DORA?",
    ),
    expected_verdict=VerdictType.CONFIRMED,
    expected_relationship=EpistemicRelationship.CONTRADICTS,
    expected_lane=LaneType.LANE_A_GRAPH_LCI,
    expected_intent=IntentType.COMPARISON,
    is_synthetic=True,
    gold_doc_ids=["MSA", "DORA"],
)

V5_SYNTH_02 = V5GroundTruthCase(
    case_id="v5-synth-02",
    description="Aligned regulation and guideline — NO contradiction",
    documents=[
        RawDocument(
            doc_id="DATA_REG",
            content=(
                "DATA PROTECTION REGULATION\n\n"
                "Article 1. DATA RETENTION\n"
                "Personal data shall be retained for no longer than 7 years.\n\n"
                "Article 2. DATA MINIMIZATION\n"
                "Only data necessary for the stated purpose shall be collected."
            ),
            metadata={"type": "regulation", "domain": "REGULATORY"},
        ),
        RawDocument(
            doc_id="GUIDELINE_01",
            content=(
                "DATA HANDLING GUIDELINE\n\n"
                "Section 1. RETENTION POLICY\n"
                "Organizations should retain personal data for a maximum "
                "of 7 years, in accordance with the Data Protection "
                "Regulation.\n\n"
                "Section 2. COLLECTION\n"
                "Collect only what is necessary for the stated purpose."
            ),
            metadata={"type": "contract", "domain": "LEGAL"},
        ),
    ],
    query=QueryRequest(
        query_id="v5-synth-02-q",
        query_text="Does the guideline conflict with the data protection regulation?",
    ),
    expected_verdict=VerdictType.REFUTED,
    expected_relationship=EpistemicRelationship.SUPPORTS,
    expected_lane=LaneType.LANE_A_GRAPH_LCI,
    expected_intent=IntentType.COMPARISON,
    is_synthetic=True,
    gold_doc_ids=["DATA_REG", "GUIDELINE_01"],
)

V5_SYNTH_03 = V5GroundTruthCase(
    case_id="v5-synth-03",
    description="GDPR vs DPA — retention period conflict",
    documents=[
        RawDocument(
            doc_id="GDPR_EXTRACT",
            content=(
                "GENERAL DATA PROTECTION REGULATION (EXCERPT)\n\n"
                "Article 5(1)(e) STORAGE LIMITATION\n"
                "Personal data shall be kept in a form which permits "
                "identification of data subjects for no longer than is "
                "necessary for the purposes for which the personal data "
                "are processed.\n\n"
                "Article 17. RIGHT TO ERASURE\n"
                "The data subject shall have the right to obtain from the "
                "controller the erasure of personal data without undue delay."
            ),
            metadata={"type": "regulation", "domain": "REGULATORY"},
        ),
        RawDocument(
            doc_id="DPA_01",
            content=(
                "DATA PROCESSING AGREEMENT\n\n"
                "Section 4. DATA RETENTION\n"
                "Processor shall retain all personal data for a minimum "
                "period of 10 years for audit and compliance purposes.\n\n"
                "Section 5. DELETION\n"
                "Upon termination, Processor shall delete personal data "
                "within 180 days."
            ),
            metadata={"type": "contract", "domain": "LEGAL"},
        ),
    ],
    query=QueryRequest(
        query_id="v5-synth-03-q",
        query_text=(
            "Does the DPA's 10-year retention period conflict with GDPR "
            "storage limitation?"
        ),
    ),
    expected_verdict=VerdictType.CONFIRMED,
    expected_relationship=EpistemicRelationship.CONTRADICTS,
    expected_lane=LaneType.LANE_A_GRAPH_LCI,
    expected_intent=IntentType.COMPARISON,
    is_synthetic=True,
    gold_doc_ids=["GDPR_EXTRACT", "DPA_01"],
)


# ============================================================================
# FIXTURE COLLECTIONS
# ============================================================================

V5_ALL_FIXTURES: List[V5GroundTruthCase] = [
    V5_SYNTH_01,
    V5_SYNTH_02,
    V5_SYNTH_03,
]

_V5_FIXTURE_INDEX: Dict[str, V5GroundTruthCase] = {
    c.case_id: c for c in V5_ALL_FIXTURES
}


def get_v5_fixture(case_id: str) -> V5GroundTruthCase:
    """Retrieve a v5 fixture by case_id."""
    if case_id not in _V5_FIXTURE_INDEX:
        raise KeyError(
            f"Unknown v5 fixture '{case_id}'. "
            f"Available: {sorted(_V5_FIXTURE_INDEX.keys())}"
        )
    return _V5_FIXTURE_INDEX[case_id]


# ============================================================================
# v5 GROUND TRUTH EVALUATION
# ============================================================================

def evaluate_v5_ground_truth(
    case: V5GroundTruthCase,
    reasoner_output: Optional[ReasonerOutput],
) -> V5GroundTruthResult:
    """Compare v5 pipeline output against ground truth expectations.

    Parameters
    ----------
    case:
        The labeled test case.
    reasoner_output:
        The ReasonerOutput from running the v5 pipeline.

    Returns
    -------
    V5GroundTruthResult
        Per-dimension match flags.
    """
    details: Dict[str, Any] = {}

    if reasoner_output is None:
        return V5GroundTruthResult(
            case_id=case.case_id,
            verdict_match=False,
            relationship_match=False,
            lane_match=False,
            intent_match=False,
            details={"error": "No reasoner output"},
        )

    finding = reasoner_output.finding

    # 1. Verdict match
    verdict_match = finding.verdict.verdict_type == case.expected_verdict
    details["found_verdict"] = finding.verdict.verdict_type.value
    details["expected_verdict"] = case.expected_verdict.value

    # 2. Relationship match: check if any inferred relationship matches
    relationship_match = False
    found_relationships = []
    for rel in finding.relationships:
        found_relationships.append(rel.relationship.value)
        if rel.relationship == case.expected_relationship:
            relationship_match = True
    details["found_relationships"] = found_relationships
    details["expected_relationship"] = case.expected_relationship.value

    # 3. Lane match
    lane_match = finding.lane_used == case.expected_lane.value
    details["found_lane"] = finding.lane_used
    details["expected_lane"] = case.expected_lane.value

    # 4. Intent match
    intent_match = finding.intent_classified == case.expected_intent.value
    details["found_intent"] = finding.intent_classified
    details["expected_intent"] = case.expected_intent.value

    return V5GroundTruthResult(
        case_id=case.case_id,
        verdict_match=verdict_match,
        relationship_match=relationship_match,
        lane_match=lane_match,
        intent_match=intent_match,
        details=details,
    )
