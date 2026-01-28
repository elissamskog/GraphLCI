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
    # GlobalContext,
    GroundedEvidence,
    InferredRelationship,
    SearchStrategyClassification,
    SearchStrategyType,
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
    build_targeted_prompt,
    build_hybrid_prompt,
)
from reasoner.epistemic_extractor import extract_relationships
from reasoner.evidence_grounder import find_quote_in_documents, ground_evidence
from reasoner.verdict_resolver import calculate_confidence, determine_verdict_type
from reasoner.reasoner_controller import ReasonerController


# ============================================================================
# Mock LLM responses
# ============================================================================

MOCK_HYBRID_RESPONSE = json.dumps({
    "relationships": [
        {
            "source_id": "doc_1",
            "target_id": "doc_2",
            "relationship": "STRONGLY_SUPPORTS",
            "confidence": 0.95,
            "reasoning": "Structural and semantic evidence align.",
            "evidence_quote": "Liability shall not exceed annual fees"
        },
        {
            "source_id": "doc_1",
            "target_id": "doc_2",
            "relationship": "SUPPORTS",
            "confidence": 0.9,
            "reasoning": "Direct quote found in chunk.",
            "evidence_quote": "The total cap applies"
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
        "verdict_type": "CONFIRMED",
        "answer": "The clauses are consistent",
        "confidence": 0.95,
        "summary": "Confirmed via hybrid synthesis."
    },
    "reasoning_trace": "MONOLOGUE: The structural path in A supports the relevance of semantic chunk C. Step 1: Analytical review..."
})

MOCK_TEMPORAL_ONLY_C_RESPONSE = json.dumps({
    "relationships": [
        {
            "source_id": "chunk_1",
            "target_id": "chunk_2",
            "relationship": "SUPERSEDES",
            "confidence": 0.9,
            "reasoning": "Chunk 2 seems later than Chunk 1",
            "evidence_quote": "Revised version 2.0"
        }
    ],
    "evidence": [
        {
            "claim": "Revised version",
            "source_document": "doc_1",
            "exact_quote": "Revised version 2.0"
        }
    ],
    "verdict": {
        "verdict_type": "PARTIALLY_CONFIRMED",
        "answer": "Possible superseding",
        "confidence": 0.4,
        "summary": "Temporal claim with only semantic evidence."
    },
    "reasoning_trace": "Found temporal language in chunks."
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
        },
        edges={},
        document_ids=["doc_1"],
        seed_node_ids=["node_1"],
        total_ppr_mass=0.8,
    )


def make_focused_context() -> FocusedContext:
    documents = {
        "doc_1": RawDocument(
            doc_id="doc_1",
            content="Liability shall not exceed annual fees paid under this agreement.",
            metadata={"type": "MSA"},
        ),
    }
    return FocusedContext(
        documents=documents,
        edge_grammar="doc_1 --[REFERENCES]--> doc_2",
        node_context="node_1: MSA",
        scope=make_scope(),
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
            ),
            relevance_score=0.95,
        ),
    ]
    return TargetedContext(chunks=chunks, scope=make_scope())


def make_router_decision(lane: LaneType) -> RouterDecisionV5:
    return RouterDecisionV5(
        query_id="test-query-001",
        search_strategy=SearchStrategyClassification(
            classification=SearchStrategyType.CONTRADICTION_DETECTION,
            confidence=0.95,
            reasoning="test",
        ),
        selected_lane=lane,
        seed_nodes=[],
        confidence=0.95,
    )


# ============================================================================
# Test functions
# ============================================================================

def test_hybrid_prompt_builder():
    """Verify hybrid builder contains both stream headers and precedence rule."""
    query = make_query_request()
    focused = make_focused_context()
    targeted = make_targeted_context()

    prompt = build_hybrid_prompt(query, focused, targeted)
    assert "## STREAM 1: STRUCTURAL (LANE A)" in prompt
    assert "## STREAM 2: SEMANTIC (LANE C)" in prompt
    assert "THE PRECEDENCE RULE" in prompt
    assert "REASONING MONOLOGUE" in prompt
    assert "doc_1" in prompt
    assert "chunk_1" in prompt

    print("  PASSED: test_hybrid_prompt_builder")


def test_epistemic_calibration():
    """Verify confidence cap when temporal claim lacks Lane A evidence."""
    ri = ReasonerInput(
        query=make_query_request(),
        query_id="q1",
        router_decision=make_router_decision(LaneType.LANE_C_VECTOR_ONLY),
        focused_context=None, # Only Lane C
        targeted_context=make_targeted_context(),
    )

    mock_llm = MockLLMClient(default_response=MOCK_TEMPORAL_ONLY_C_RESPONSE)
    controller = ReasonerController(llm_client=mock_llm)
    output = controller.process(ri)

    # Even if LLM is confident, controller should cap it because SUPERSEDES is temporal
    # and we have no Lane A context.
    assert output.finding.verdict.confidence <= 0.4
    print("  PASSED: test_epistemic_calibration")


def test_verdict_resolver_strict_confirmed():
    """Verify CONFIRMED requires linked quote (C) to structural relationship (A)."""
    # 1. Multiple strong supports but NO grounded quotes
    rels = [
        InferredRelationship("doc1", "doc2", EpistemicRelationship.STRONGLY_SUPPORTS, 0.9, "r1"),
        InferredRelationship("doc1", "doc3", EpistemicRelationship.STRONGLY_SUPPORTS, 0.9, "r2"),
    ]
    evidence = [] # No grounded evidence
    
    verdict = determine_verdict_type(rels, evidence)
    assert verdict == VerdictType.PARTIALLY_CONFIRMED # Downgraded from CONFIRMED

    # 2. Multiple strong supports WITH grounded chunk quote
    evidence_grounded = [
        GroundedEvidence(claim="c1", source_document="doc1", source_chunk="chunk1", exact_quote="quote1")
    ]
    rels_with_quote = [
        InferredRelationship("doc1", "doc2", EpistemicRelationship.STRONGLY_SUPPORTS, 0.9, "r1", evidence_quote="quote1"),
        InferredRelationship("doc1", "doc3", EpistemicRelationship.STRONGLY_SUPPORTS, 0.9, "r2"),
    ]
    
    verdict_confirmed = determine_verdict_type(rels_with_quote, evidence_grounded)
    assert verdict_confirmed == VerdictType.CONFIRMED

    print("  PASSED: test_verdict_resolver_strict_confirmed")


def test_full_hybrid_pipeline():
    """Verify full synthesis pipeline with monologue and hybrid contexts."""
    ri = ReasonerInput(
        query=make_query_request(),
        query_id="q1",
        router_decision=make_router_decision(LaneType.LANE_A_GRAPH_LCI),
        focused_context=make_focused_context(),
        targeted_context=make_targeted_context(),
    )

    mock_llm = MockLLMClient(default_response=MOCK_HYBRID_RESPONSE)
    controller = ReasonerController(llm_client=mock_llm)
    output = controller.process(ri)

    assert output.finding.verdict.verdict_type == VerdictType.CONFIRMED
    assert "MONOLOGUE" in output.finding.reasoning_trace
    assert output.finding.search_strategy == "CONTRADICTION_DETECTION"
    
    print("  PASSED: test_full_hybrid_pipeline")


if __name__ == "__main__":
    print("Running REASONER Hybrid Synthesis tests...\n")

    tests = [
        test_hybrid_prompt_builder,
        test_epistemic_calibration,
        test_verdict_resolver_strict_confirmed,
        test_full_hybrid_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed}/{passed + failed} tests passed")

    if failed > 0:
        sys.exit(1)
