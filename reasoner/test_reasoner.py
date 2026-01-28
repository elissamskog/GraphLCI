"""
REASONER Unit Tests (v5)
Location: reasoner/test_reasoner.py
Owner: REASONER Instance

Tests for prompt building, epistemic extraction, evidence grounding,
verdict resolution, and the full pipeline.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from schemas.contracts import (
    ConfidenceFactors,
    EmbeddedNode,
    EpistemicRelationship,
    FocusedContext,
    GlobalContext,
    GroundedEvidence,
    InferredRelationship,
    IntentClassification,
    IntentType,
    LaneType,
    QueryRequest,
    RankedChunk,
    RawDocument,
    ReasonerInput,
    ReasonerOutput,
    RouterDecisionV5,
    Scope,
    ScopedNode,
    StructuralEdge,
    StructuralEdgeType,
    TargetedContext,
    TextChunk,
    VerdictType,
)

from reasoner.prompt_builder import (
    build_focused_prompt,
    build_global_prompt,
    build_targeted_prompt,
)
from reasoner.epistemic_extractor import extract_relationships
from reasoner.evidence_grounder import find_quote_in_documents, ground_evidence
from reasoner.verdict_resolver import calculate_confidence, determine_verdict_type
from reasoner.reasoner_controller import ReasonerController


# ============================================================================
# Mock LLM responses
# ============================================================================

MOCK_FOCUSED_RESPONSE = json.dumps({
    "relationships": [
        {
            "source_id": "doc_1",
            "target_id": "doc_2",
            "relationship": "CONTRADICTS",
            "confidence": 0.9,
            "reasoning": "Doc 1 caps liability while Doc 2 requires unlimited",
            "evidence_quote": "Liability shall not exceed annual fees"
        }
    ],
    "evidence": [
        {
            "claim": "Liability is capped",
            "source_document": "doc_1",
            "exact_quote": "Liability shall not exceed annual fees"
        }
    ],
    "verdict": {
        "verdict_type": "REFUTED",
        "answer": "The clauses contradict each other",
        "confidence": 0.9,
        "summary": "Doc 1 limits liability while Doc 2 requires unlimited."
    },
    "reasoning_trace": "Step 1: Compared clauses. Step 2: Found contradiction."
})

MOCK_GLOBAL_RESPONSE = json.dumps({
    "relationships": [
        {
            "source_id": "community_1",
            "target_id": "community_2",
            "relationship": "SUPPORTS",
            "confidence": 0.85,
            "reasoning": "Both communities agree on term definitions",
            "evidence_quote": "Term definitions are consistent"
        }
    ],
    "evidence": [
        {
            "claim": "Term consistency across documents",
            "source_document": "community_1",
            "exact_quote": "Term definitions are consistent"
        }
    ],
    "verdict": {
        "verdict_type": "CONFIRMED",
        "answer": "Definitions are consistent across communities",
        "confidence": 0.85,
        "summary": "Community summaries show consistent definitions."
    },
    "reasoning_trace": "Analyzed themes across communities."
})

MOCK_TARGETED_RESPONSE = json.dumps({
    "relationships": [
        {
            "source_id": "chunk_1",
            "target_id": "chunk_2",
            "relationship": "PARTIALLY_SUPPORTS",
            "confidence": 0.7,
            "reasoning": "Partial overlap in clause language",
            "evidence_quote": "The indemnification clause"
        }
    ],
    "evidence": [
        {
            "claim": "Indemnification applies",
            "source_document": "doc_1",
            "exact_quote": "The indemnification clause"
        }
    ],
    "verdict": {
        "verdict_type": "PARTIALLY_CONFIRMED",
        "answer": "Partial support found",
        "confidence": 0.7,
        "summary": "Some support found in chunk evidence."
    },
    "reasoning_trace": "Searched chunks for relevant text."
})


# ============================================================================
# Mock LLM client
# ============================================================================

class MockLLMClient:
    """Mock LLM client that returns canned responses based on prompt substrings."""

    def __init__(
        self,
        response_map: Optional[dict] = None,
        default_response: str = "",
    ) -> None:
        self._response_map = response_map or {}
        self._default = default_response

    def generate(self, prompt: str) -> str:
        for key, response in self._response_map.items():
            if key in prompt:
                return response
        return self._default


# ============================================================================
# Fixture helpers
# ============================================================================

def make_query_request() -> QueryRequest:
    return QueryRequest(
        query_id="test-query-001",
        query_text="Do the liability clauses in Doc 1 and Doc 2 contradict each other?",
    )


def make_scope() -> Scope:
    return Scope(
        nodes={
            "node_1": ScopedNode(
                node=EmbeddedNode(
                    node_id="node_1",
                    node_type="DOCUMENT",
                    canonical_text="MSA liability clause",
                    embedding=[0.1] * 10,
                    source_doc_id="doc_1",
                    source_chunks=["chunk_1"],
                    metadata={},
                ),
                ppr_score=0.8,
                rank=1,
            ),
            "node_2": ScopedNode(
                node=EmbeddedNode(
                    node_id="node_2",
                    node_type="DOCUMENT",
                    canonical_text="DORA liability requirement",
                    embedding=[0.2] * 10,
                    source_doc_id="doc_2",
                    source_chunks=["chunk_2"],
                    metadata={},
                ),
                ppr_score=0.6,
                rank=2,
            ),
        },
        edges={
            "edge_1": StructuralEdge(
                edge_id="edge_1",
                source="node_1",
                target="node_2",
                edge_type=StructuralEdgeType.REFERENCES,
                weight=0.9,
                metadata={},
                is_cross_document=True,
            ),
        },
        document_ids=["doc_1", "doc_2"],
        seed_node_ids=["node_1"],
        total_ppr_mass=1.4,
    )


def make_focused_context() -> FocusedContext:
    documents = {
        "doc_1": RawDocument(
            doc_id="doc_1",
            content=(
                "Liability shall not exceed annual fees paid under this agreement. "
                "The total cap applies to all claims combined."
            ),
            metadata={"type": "MSA"},
        ),
        "doc_2": RawDocument(
            doc_id="doc_2",
            content=(
                "The vendor accepts unlimited liability for all damages arising "
                "from breach. No cap shall apply to indemnification obligations."
            ),
            metadata={"type": "DORA"},
        ),
    }
    return FocusedContext(
        documents=documents,
        edge_grammar="doc_1 --[REFERENCES]--> doc_2: MSA references DORA requirements",
        node_context="node_1: MSA liability clause | node_2: DORA liability requirement",
        scope=make_scope(),
    )


def make_global_context() -> GlobalContext:
    scope = Scope(
        nodes={},
        edges={},
        document_ids=["doc_1", "doc_2"],
        seed_node_ids=[],
        total_ppr_mass=0.0,
    )
    return GlobalContext(
        community_summaries={
            "community_1": "Documents discuss liability caps and limitations.",
            "community_2": "Regulatory requirements mandate unlimited liability for certain breaches.",
        },
        scope=scope,
    )


def make_targeted_context() -> TargetedContext:
    chunks = [
        RankedChunk(
            chunk=TextChunk(
                chunk_id="chunk_1",
                source_doc_id="doc_1",
                text="Liability shall not exceed annual fees paid under this agreement.",
                token_count=12,
                position=0,
                page_number=1,
            ),
            relevance_score=0.95,
        ),
        RankedChunk(
            chunk=TextChunk(
                chunk_id="chunk_2",
                source_doc_id="doc_2",
                text="The vendor accepts unlimited liability for all damages arising from breach.",
                token_count=12,
                position=0,
                page_number=2,
            ),
            relevance_score=0.88,
        ),
    ]
    scope = Scope(
        nodes={},
        edges={},
        document_ids=["doc_1", "doc_2"],
        seed_node_ids=[],
        total_ppr_mass=0.0,
    )
    return TargetedContext(chunks=chunks, scope=scope)


def make_router_decision(lane: LaneType) -> RouterDecisionV5:
    return RouterDecisionV5(
        query_id="test-query-001",
        intent=IntentClassification(
            classification=IntentType.CONTRADICTION_DETECTION,
            confidence=0.95,
            reasoning="Query asks about contradictions between documents",
        ),
        selected_lane=lane,
        seed_nodes=[],
        confidence=0.95,
    )


def make_reasoner_input(lane: LaneType) -> ReasonerInput:
    return ReasonerInput(
        query=make_query_request(),
        query_id="test-query-001",
        router_decision=make_router_decision(lane),
        focused_context=make_focused_context() if lane == LaneType.LANE_A_GRAPH_LCI else None,
        global_context=make_global_context() if lane == LaneType.LANE_B_GRAPH_ONLY else None,
        targeted_context=make_targeted_context() if lane == LaneType.LANE_C_VECTOR_ONLY else None,
    )


# ============================================================================
# Test functions
# ============================================================================

def test_prompt_builder():
    """Verify each builder returns a non-empty string containing query text and context."""
    query = make_query_request()

    # Focused prompt
    focused = build_focused_prompt(query, make_focused_context())
    assert isinstance(focused, str) and len(focused) > 0
    assert query.query_text in focused
    assert "doc_1" in focused
    assert "Liability shall not exceed" in focused
    assert "REFERENCES" in focused  # edge grammar content

    # Global prompt
    global_prompt = build_global_prompt(query, make_global_context())
    assert isinstance(global_prompt, str) and len(global_prompt) > 0
    assert query.query_text in global_prompt
    assert "community_1" in global_prompt
    assert "liability caps" in global_prompt.lower()

    # Targeted prompt
    targeted = build_targeted_prompt(query, make_targeted_context())
    assert isinstance(targeted, str) and len(targeted) > 0
    assert query.query_text in targeted
    assert "chunk_1" in targeted
    assert "0.950" in targeted  # relevance score formatted

    print("  PASSED: test_prompt_builder")


def test_epistemic_extractor():
    """Verify extract_relationships correctly parses mock JSON."""
    relationships = extract_relationships(MOCK_FOCUSED_RESPONSE)

    assert len(relationships) == 1
    rel = relationships[0]
    assert rel.source_id == "doc_1"
    assert rel.target_id == "doc_2"
    assert rel.relationship == EpistemicRelationship.CONTRADICTS
    assert rel.confidence == 0.9
    assert "liability" in rel.reasoning.lower() or "cap" in rel.reasoning.lower()
    assert rel.evidence_quote == "Liability shall not exceed annual fees"

    print("  PASSED: test_epistemic_extractor")


def test_epistemic_extractor_malformed():
    """Verify graceful degradation on bad JSON."""
    # Completely invalid
    result = extract_relationships("This is not JSON at all")
    assert result == []

    # Valid JSON but no relationships key
    result = extract_relationships('{"foo": "bar"}')
    assert result == []

    # Invalid relationship type should default to UNRELATED
    bad_rel = json.dumps({
        "relationships": [{
            "source_id": "a",
            "target_id": "b",
            "relationship": "INVALID_TYPE",
            "confidence": 0.5,
            "reasoning": "test",
        }]
    })
    result = extract_relationships(bad_rel)
    assert len(result) == 1
    assert result[0].relationship == EpistemicRelationship.UNRELATED

    # Confidence out of range should be clamped
    high_conf = json.dumps({
        "relationships": [{
            "source_id": "a",
            "target_id": "b",
            "relationship": "SUPPORTS",
            "confidence": 1.5,
            "reasoning": "test",
        }]
    })
    result = extract_relationships(high_conf)
    assert result[0].confidence == 1.0

    print("  PASSED: test_epistemic_extractor_malformed")


def test_evidence_grounder():
    """Verify evidence grounding for exact match, no match, and chunk-level."""
    documents = {
        "doc_1": RawDocument(
            doc_id="doc_1",
            content="Liability shall not exceed annual fees paid under this agreement.",
            metadata={},
        ),
        "doc_2": RawDocument(
            doc_id="doc_2",
            content="The vendor accepts unlimited liability for all damages.",
            metadata={},
        ),
    }

    # Exact match
    result = find_quote_in_documents("Liability shall not exceed annual fees", documents)
    assert result is not None
    assert result.source_document == "doc_1"
    assert result.char_offset is not None

    # No match
    result = find_quote_in_documents("This text does not appear anywhere in docs", documents)
    assert result is None

    # ground_evidence with mix of groundable and ungroundable claims
    claims = [
        "Liability shall not exceed annual fees",
        "This text does not exist in any document at all whatsoever",
    ]
    grounded = ground_evidence(claims, documents)
    assert len(grounded) == 2
    assert grounded[0].source_document == "doc_1"
    assert grounded[0].exact_quote != ""
    assert grounded[1].exact_quote == ""  # Ungrounded

    # Chunk-level grounding
    chunks = [
        TextChunk(
            chunk_id="chunk_99",
            source_doc_id="doc_3",
            text="Special clause about indemnification limits.",
            token_count=6,
            position=0,
            page_number=5,
        ),
    ]
    claims2 = ["indemnification limits"]
    grounded2 = ground_evidence(claims2, {}, chunks)
    assert len(grounded2) == 1
    assert grounded2[0].source_chunk == "chunk_99"
    assert grounded2[0].page_number == 5

    print("  PASSED: test_evidence_grounder")


def test_verdict_resolver():
    """Verify verdict determination and confidence calculation."""
    evidence = [
        GroundedEvidence(claim="test", source_document="doc_1", exact_quote="quote"),
    ]

    # REFUTED: more contradictions with >= 2 strong contradicts
    rels_refuted = [
        InferredRelationship("a", "b", EpistemicRelationship.CONTRADICTS, 0.9, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.STRONGLY_CONTRADICTS, 0.95, "r2"),
        InferredRelationship("a", "d", EpistemicRelationship.CONTRADICTS, 0.85, "r3"),
    ]
    assert determine_verdict_type(rels_refuted, evidence) == VerdictType.REFUTED

    # PARTIALLY_REFUTED: more contradictions but < 2 strong contradicts
    rels_partial_refuted = [
        InferredRelationship("a", "b", EpistemicRelationship.PARTIALLY_CONTRADICTS, 0.7, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.CONTRADICTS, 0.8, "r2"),
    ]
    assert determine_verdict_type(rels_partial_refuted, evidence) == VerdictType.PARTIALLY_REFUTED

    # CONFIRMED: more supports with >= 2 strong supports
    rels_confirmed = [
        InferredRelationship("a", "b", EpistemicRelationship.SUPPORTS, 0.9, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.STRONGLY_SUPPORTS, 0.95, "r2"),
        InferredRelationship("a", "d", EpistemicRelationship.SUPPORTS, 0.85, "r3"),
    ]
    assert determine_verdict_type(rels_confirmed, evidence) == VerdictType.CONFIRMED

    # PARTIALLY_CONFIRMED: more supports but < 2 strong supports
    rels_partial_confirmed = [
        InferredRelationship("a", "b", EpistemicRelationship.PARTIALLY_SUPPORTS, 0.7, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.SUPPORTS, 0.8, "r2"),
    ]
    assert determine_verdict_type(rels_partial_confirmed, evidence) == VerdictType.PARTIALLY_CONFIRMED

    # NOT_FOUND: no relationships and no evidence
    assert determine_verdict_type([], []) == VerdictType.NOT_FOUND

    # NOT_FOUND: all neutral
    rels_neutral = [
        InferredRelationship("a", "b", EpistemicRelationship.UNRELATED, 0.5, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.INSUFFICIENT_EVIDENCE, 0.3, "r2"),
    ]
    assert determine_verdict_type(rels_neutral, evidence) == VerdictType.NOT_FOUND

    # INCONCLUSIVE: equal support and contradiction
    rels_mixed = [
        InferredRelationship("a", "b", EpistemicRelationship.SUPPORTS, 0.9, "r1"),
        InferredRelationship("a", "c", EpistemicRelationship.CONTRADICTS, 0.9, "r2"),
    ]
    assert determine_verdict_type(rels_mixed, evidence) == VerdictType.INCONCLUSIVE

    # Confidence calculation
    conf = calculate_confidence(rels_refuted, evidence)
    assert isinstance(conf, ConfidenceFactors)
    assert 0.0 <= conf.overall <= 1.0
    assert 0.0 <= conf.evidence_quality <= 1.0
    assert 0.0 <= conf.source_agreement <= 1.0
    assert 0.0 <= conf.coverage <= 1.0

    # Confidence with no relationships
    empty_conf = calculate_confidence([], [])
    assert empty_conf.overall == 0.0
    assert empty_conf.evidence_quality == 0.0

    print("  PASSED: test_verdict_resolver")


def test_full_pipeline():
    """End-to-end: build ReasonerInput, run controller, verify output."""
    ri = make_reasoner_input(LaneType.LANE_A_GRAPH_LCI)

    mock_llm = MockLLMClient(
        response_map={
            "liability": MOCK_FOCUSED_RESPONSE,
        },
        default_response=MOCK_FOCUSED_RESPONSE,
    )

    controller = ReasonerController(llm_client=mock_llm)
    output = controller.process(ri)

    assert isinstance(output, ReasonerOutput)

    finding = output.finding
    assert finding.finding_id  # Non-empty
    assert finding.query_id == "test-query-001"
    assert isinstance(finding.verdict.verdict_type, VerdictType)
    assert len(finding.relationships) >= 1
    assert all(
        isinstance(r.relationship, EpistemicRelationship)
        for r in finding.relationships
    )
    assert len(finding.evidence) >= 1
    assert any(e.source_document for e in finding.evidence)

    # Confidence factors
    cf = finding.confidence_factors
    assert 0.0 <= cf.overall <= 1.0
    assert 0.0 <= cf.evidence_quality <= 1.0
    assert 0.0 <= cf.source_agreement <= 1.0
    assert 0.0 <= cf.coverage <= 1.0

    # Reasoning trace
    assert finding.reasoning_trace  # Non-empty

    # ReasonerTrace
    trace = output.trace
    assert trace.trace_id
    assert trace.query_id == "test-query-001"
    assert trace.llm_time_ms >= 0
    assert trace.total_time_ms >= 0
    assert trace.relationships_inferred >= 1
    assert trace.lane_used == LaneType.LANE_A_GRAPH_LCI.value

    # Scope summary
    assert finding.scope.document_count == 2
    assert finding.scope.node_count == 2
    assert finding.scope.edge_count == 1
    assert finding.scope.bridge_edges_used == 1

    # Lane and intent
    assert finding.lane_used == "LANE_A_GRAPH_LCI"
    assert finding.intent_classified == "CONTRADICTION_DETECTION"

    print("  PASSED: test_full_pipeline")


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    print("Running REASONER tests...\n")

    tests = [
        test_prompt_builder,
        test_epistemic_extractor,
        test_epistemic_extractor_malformed,
        test_evidence_grounder,
        test_verdict_resolver,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed}/{passed + failed} tests passed")

    if failed > 0:
        sys.exit(1)
