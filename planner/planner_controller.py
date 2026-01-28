"""
Planner Controller
Responsibility: Orchestrate intent classification, target extraction, and lane routing.

Receives PlannerRequest from ROOT, returns PlannerResponse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import sys
import os
# Add project root so 'schemas' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add this file's directory so 'planner' resolves to the inner package (must be first)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from schemas.contracts import (
    DomainType,
    FocusedContext,
    GlobalContext,
    IdentifiedTargets,
    IntentClassification,
    IntentType,
    LaneType,
    PlannerInput,
    PlannerOutput,
    PlannerRequest,
    PlannerResponse,
    QueryRequest,
    RawDocument,
    RouterDecision,
    RouterDecisionV5,
    Scope,
    SeedNode,
    TargetedContext,
    TextChunk,
)

from planner.intent_classifier import ClassificationInput, IntentClassifier
from planner.target_extractor import ExtractionInput, TargetExtractor
from planner.lane_router import LaneRouter, RoutingInput
from planner.query_embedder import embed_query
from planner.seed_finder import find_seed_nodes
from planner.ppr_expander import personalized_pagerank, extract_scope
from planner.context_assembler import (
    assemble_focused_context,
    assemble_global_context,
    assemble_targeted_context,
)


# ============================================================================
# CONTROLLER
# ============================================================================

class PlannerController:
    """
    Main controller for the PLANNER instance.

    Chains IntentClassifier -> TargetExtractor -> LaneRouter to produce
    a RouterDecision for each incoming PlannerRequest.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.intent_classifier = IntentClassifier(self.config)
        self.target_extractor = TargetExtractor(self.config)
        self.lane_router = LaneRouter(self.config)

    def process(self, request: PlannerRequest) -> PlannerResponse:
        """
        Main processing pipeline.

        Flow:
        1. Classify intent
        2. Extract targets
        3. Route to lane
        4. Compose RouterDecision and wrap in PlannerResponse
        """
        try:
            # Step 1: Classify intent
            classification = self.intent_classifier.classify(
                ClassificationInput(
                    query_text=request.query_request.query_text,
                    user_context=request.query_request.user_context,
                )
            )

            # Step 2: Extract targets
            targets = self.target_extractor.extract(
                ExtractionInput(
                    query_text=request.query_request.query_text,
                    available_documents=self._get_available_docs(),
                    known_entities=self._get_known_entities(),
                )
            )

            # Step 3: Route to lane
            routing = self.lane_router.route(
                RoutingInput(
                    intent=classification.intent,
                    targets=targets.identified_targets,
                    corpus_metadata=self._get_corpus_metadata(),
                )
            )

            # Step 4: Compose decision
            router_decision = RouterDecision(
                query_id=request.query_request.query_id,
                intent=classification.intent,
                selected_lane=routing.selected_lane,
                requires_graph_scope=routing.requires_graph_scope,
                identified_targets=targets.identified_targets,
                domain_expectation=routing.domain_expectation,
            )

            return PlannerResponse(
                success=True,
                router_decision=router_decision,
                errors=[],
            )

        except Exception as e:
            return PlannerResponse(
                success=False,
                router_decision=None,
                errors=[str(e)],
            )

    def _get_available_docs(self) -> List[str]:
        """Get available document IDs from config."""
        return self.config.get('available_documents', [])

    def _get_known_entities(self) -> List[str]:
        """Get known entities from config."""
        return self.config.get('known_entities', [])

    def _get_corpus_metadata(self) -> Dict[str, Any]:
        """Get corpus metadata from config."""
        return self.config.get(
            'corpus_metadata',
            {'has_graph': False, 'domain': 'LEGAL'},
        )


# ============================================================================
# V5 CONTROLLER
# ============================================================================

class PlannerControllerV5:
    """
    V5 controller for the PLANNER instance.

    Pipeline: embed query -> find seeds -> PPR expand -> classify intent
    -> route lane -> assemble context -> return PlannerOutput.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.intent_classifier = IntentClassifier(self.config)

    def process(self, planner_input: PlannerInput) -> PlannerOutput:
        """Run the full v5 planner pipeline."""
        query_text = planner_input.query.query_text
        query_id = planner_input.query.query_id

        # Step 1: Embed query
        embedding_client = self.config['embedding_client']
        query_embedding = embed_query(query_text, embedding_client)

        # Step 2: Find seeds
        seeds = find_seed_nodes(
            query_embedding,
            planner_input.similarity_index,
            graph=planner_input.graph,
            top_k=self.config.get('seed_top_k', 10),
            threshold=self.config.get('seed_threshold', 0.5),
        )

        # Step 3: PPR expand
        seed_ids = [s.node_id for s in seeds]
        ppr_scores = personalized_pagerank(
            planner_input.graph,
            seed_ids,
            alpha=self.config.get('ppr_alpha', 0.85),
            max_iterations=self.config.get('ppr_max_iter', 50),
            tolerance=self.config.get('ppr_tolerance', 1e-6),
        )
        scope = extract_scope(
            planner_input.graph,
            ppr_scores,
            seed_ids,
            top_k=self.config.get('scope_max_nodes', 50),
            min_score=self.config.get('scope_min_score', 0.01),
        )

        # Step 4: Classify intent (reuse v4 offline classifier)
        classification = self.intent_classifier.classify(
            ClassificationInput(query_text=query_text)
        )

        # Step 5: Route lane (v5 rules)
        lane = self._route_v5(classification.intent, scope)

        # Step 6: Build router decision
        router_decision = RouterDecisionV5(
            query_id=query_id,
            intent=classification.intent,
            selected_lane=lane,
            seed_nodes=seeds,
            scope=scope,
            confidence=classification.intent.confidence,
        )

        # Step 7: Assemble context per lane
        focused: Optional[FocusedContext] = None
        global_ctx: Optional[GlobalContext] = None
        targeted: Optional[TargetedContext] = None

        if lane == LaneType.LANE_A_GRAPH_LCI:
            focused = assemble_focused_context(
                scope,
                planner_input.graph,
                planner_input.document_store or {},
            )
        elif lane == LaneType.LANE_B_GRAPH_ONLY:
            if planner_input.community_structure is not None:
                summaries = {
                    cid: c.summary or ""
                    for cid, c in planner_input.community_structure.communities.items()
                }
                global_ctx = assemble_global_context(
                    scope,
                    planner_input.community_structure,
                    summaries,
                )
        elif lane == LaneType.LANE_C_VECTOR_ONLY:
            chunks = self._gather_chunks(planner_input)
            targeted = assemble_targeted_context(
                scope,
                chunks,
                query_embedding,
            )

        return PlannerOutput(
            router_decision=router_decision,
            focused_context=focused,
            global_context=global_ctx,
            targeted_context=targeted,
        )

    def _route_v5(
        self, intent: IntentClassification, scope: Scope,
    ) -> LaneType:
        """V5 routing rules.

        - COMPARISON -> LANE_A
        - SENSEMAKING -> LANE_B (override to LANE_A if document_count <= 5)
        - LOOKUP -> LANE_C
        - Override: if scope has < 5 nodes -> LANE_C
        - Low confidence fallback -> LANE_C
        """
        confidence_threshold = self.config.get('confidence_threshold', 0.7)
        if intent.confidence < confidence_threshold:
            return LaneType.LANE_C_VECTOR_ONLY

        node_count = len(scope.nodes)
        if node_count < 5:
            return LaneType.LANE_C_VECTOR_ONLY

        intent_type = intent.classification

        if intent_type == IntentType.COMPARISON:
            return LaneType.LANE_A_GRAPH_LCI

        if intent_type == IntentType.SENSEMAKING:
            doc_count = len(scope.document_ids)
            if doc_count <= 5:
                return LaneType.LANE_A_GRAPH_LCI
            return LaneType.LANE_B_GRAPH_ONLY

        if intent_type == IntentType.LOOKUP:
            return LaneType.LANE_C_VECTOR_ONLY

        return LaneType.LANE_C_VECTOR_ONLY

    def _gather_chunks(self, planner_input: PlannerInput) -> List[TextChunk]:
        """Gather TextChunks — prefer real chunks from chunker, fall back to document_store."""
        if planner_input.chunks:
            return planner_input.chunks
        # Fallback: build synthetic chunks from document_store (backward compat)
        chunks: List[TextChunk] = []
        doc_store = planner_input.document_store or {}
        for doc_id, doc in doc_store.items():
            # Build a single TextChunk per document as fallback
            chunks.append(TextChunk(
                chunk_id=f"chunk_{doc_id}_0",
                source_doc_id=doc_id,
                text=doc.content,
                token_count=len(doc.content.split()),
                position=0,
            ))
        return chunks


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_planner_controller() -> None:
    """
    End-to-end tests for the PlannerController pipeline.
    """

    # ---- Config with graph, docs, entities ----
    config_with_graph = {
        'available_documents': ['MSA', 'DORA', 'SOW-2024', 'NDA'],
        'known_entities': ['Liability Cap', 'Vendor', 'Customer', 'termination fee'],
        'corpus_metadata': {'has_graph': True, 'domain': 'LEGAL'},
    }

    # ---- Config without graph ----
    config_no_graph = {
        'available_documents': ['MSA', 'DORA', 'SOW-2024', 'NDA'],
        'known_entities': ['Liability Cap', 'Vendor', 'Customer', 'termination fee'],
        'corpus_metadata': {'has_graph': False, 'domain': 'LEGAL'},
    }

    test_cases = [
        # 1. COMPARISON with graph + targets -> LANE_A, graph_scope=True
        {
            "name": "COMPARISON with graph and targets -> LANE_A",
            "config": config_with_graph,
            "query": "Does the MSA contradict DORA?",
            "query_id": "q1",
            "expected_lane": LaneType.LANE_A_GRAPH_LCI,
            "expected_graph_scope": True,
        },
        # 2. SENSEMAKING with graph -> LANE_B, graph_scope=True
        {
            "name": "SENSEMAKING with graph -> LANE_B",
            "config": config_with_graph,
            "query": "What are the key risk themes across all documents?",
            "query_id": "q2",
            "expected_lane": LaneType.LANE_B_GRAPH_ONLY,
            "expected_graph_scope": True,
        },
        # 3. LOOKUP -> LANE_C, graph_scope=False
        {
            "name": "LOOKUP -> LANE_C",
            "config": config_with_graph,
            "query": "What is the termination fee?",
            "query_id": "q3",
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 4. Multi-doc comparison -> LANE_A
        {
            "name": "Multi-doc comparison -> LANE_A",
            "config": config_with_graph,
            "query": "Compare the provisions in MSA and SOW-2024.",
            "query_id": "q4",
            "expected_lane": LaneType.LANE_A_GRAPH_LCI,
            "expected_graph_scope": True,
        },
        # 5. No-graph fallback -> LANE_C for everything
        {
            "name": "COMPARISON without graph -> LANE_C fallback",
            "config": config_no_graph,
            "query": "Does the MSA contradict DORA?",
            "query_id": "q5",
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 6. Error handling -> success=False
        {
            "name": "Error handling: bad request -> success=False",
            "config": config_with_graph,
            "query": None,  # Will cause an error
            "query_id": "q6",
            "expect_failure": True,
        },
        # 7. Validate all RouterDecision fields populated
        {
            "name": "All RouterDecision fields populated",
            "config": config_with_graph,
            "query": "Summarize the liability provisions across all contracts",
            "query_id": "q7",
            "validate_fields": True,
            "expected_lane": LaneType.LANE_B_GRAPH_ONLY,
            "expected_graph_scope": True,
        },
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Planner Controller Test Suite")
    print("=" * 70)

    for case in test_cases:
        controller = PlannerController(case["config"])

        # Build request (may intentionally fail)
        try:
            request = PlannerRequest(
                query_request=QueryRequest(
                    query_id=case["query_id"],
                    query_text=case["query"],
                )
            )
            response = controller.process(request)
        except Exception:
            # If building the request itself fails, treat as error scenario
            response = PlannerResponse(success=False, router_decision=None, errors=["request build error"])

        if case.get("expect_failure"):
            ok = not response.success
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"\n[{status}] {case['name']}")
            print(f"  Expected success=False, got success={response.success}")
            if response.errors:
                print(f"  Errors: {response.errors}")
            continue

        # Normal case assertions
        all_ok = True
        issues = []

        if not response.success:
            all_ok = False
            issues.append(f"success=False, errors={response.errors}")
        else:
            rd = response.router_decision

            if rd.selected_lane != case["expected_lane"]:
                all_ok = False
                issues.append(
                    f"lane: expected {case['expected_lane'].value}, "
                    f"got {rd.selected_lane.value}"
                )

            if rd.requires_graph_scope != case["expected_graph_scope"]:
                all_ok = False
                issues.append(
                    f"graph_scope: expected {case['expected_graph_scope']}, "
                    f"got {rd.requires_graph_scope}"
                )

            if case.get("validate_fields"):
                # Verify every field of RouterDecision is populated
                if rd.query_id != case["query_id"]:
                    all_ok = False
                    issues.append(f"query_id mismatch: {rd.query_id}")
                if rd.intent is None:
                    all_ok = False
                    issues.append("intent is None")
                if rd.identified_targets is None:
                    all_ok = False
                    issues.append("identified_targets is None")
                if rd.domain_expectation is None:
                    all_ok = False
                    issues.append("domain_expectation is None")
                if rd.intent and not isinstance(rd.intent.classification, IntentType):
                    all_ok = False
                    issues.append("intent.classification not IntentType")
                if rd.domain_expectation and not isinstance(rd.domain_expectation, DomainType):
                    all_ok = False
                    issues.append("domain_expectation not DomainType")

        status = "PASS" if all_ok else "FAIL"
        if all_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {case['name']}")
        if response.success and response.router_decision:
            rd = response.router_decision
            print(f"  Lane:        {rd.selected_lane.value}")
            print(f"  Graph scope: {rd.requires_graph_scope}")
            print(f"  Intent:      {rd.intent.classification.value} "
                  f"(conf={rd.intent.confidence})")
            print(f"  Targets:     docs={rd.identified_targets.explicit_docs}, "
                  f"entities={rd.identified_targets.implicit_entities}")
            print(f"  Domain:      {rd.domain_expectation.value}")
        for issue in issues:
            print(f"  ** {issue}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    assert failed == 0, f"{failed} test(s) failed"
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_planner_controller()
