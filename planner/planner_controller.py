"""
Planner Controller
Responsibility: Orchestrate intent classification, target extraction, and lane routing.

Receives PlannerRequest from ROOT, returns PlannerResponse.
"""

from __future__ import annotations

import sys
import os
import logging
from typing import Any, Dict, List, Optional
# Add project root so 'schemas' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add this file's directory so 'planner' resolves to the inner package (must be first)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from schemas.contracts import (
    AliasMap,
    DomainType,
    FocusedContext,
    GlobalContext,  # Deprecated
    IdentifiedTargets,
    SearchStrategyClassification,
    SearchStrategyType,
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
    WeightedSeedSet,
    ClaimDecomposition,
)

from planner.search_strategy_classifier import ClassificationInput, SearchStrategyClassifier
from planner.target_extractor import ExtractionInput, TargetExtractor
from planner.lane_router import LaneRouter, RoutingInput
from planner.query_embedder import embed_query
from planner.seed_finder import find_seed_nodes, find_seeds_multi_query, find_balanced_seeds, find_subclaim_seeds
from planner.ppr_expander import personalized_pagerank, extract_scope
from planner.context_assembler import (
    assemble_focused_context,
    assemble_targeted_context,
)
from planner.claim_decomposer import decompose_claim

logger = logging.getLogger(__name__)


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
        self.classifier = SearchStrategyClassifier(self.config)
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
            # Step 1: Classify search strategy
            classification = self.classifier.classify(
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
                    strategy=classification.strategy,
                    targets=targets.identified_targets,
                    corpus_metadata=self._get_corpus_metadata(),
                )
            )

            # Step 4: Compose decision
            router_decision = RouterDecision(
                query_id=request.query_request.query_id,
                search_strategy=classification.strategy,
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

    Pipeline: embed query -> find seeds -> PPR expand -> classify strategy
    -> route lane -> assemble context -> return PlannerOutput.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.classifier = SearchStrategyClassifier(self.config)
        self.decomposer_config = self.config.get('decomposition', {})

    def process(self, planner_input: PlannerInput) -> PlannerOutput:
        """Run the full v5 planner pipeline with Entity-Balanced Seed Selection."""
        query_text = planner_input.query.query_text
        query_id = planner_input.query.query_id
        embedding_client = self.config['embedding_client']

        # Step 1: Embed query
        query_embedding = embed_query(query_text, embedding_client)

        # Step 1.2: DECOMPOSE CLAIM (NEW - Two-Stage Architecture)
        decomposition = None
        llm_client = self.config.get('llm_client')  # Get LLM client from config
        if self.decomposer_config.get('enable_decomposition', False):
            decomposition = decompose_claim(
                query_text=query_text,
                llm_client=llm_client,
                config=self.decomposer_config
            )
            logger.info(
                f"Decomposition: should_decompose={decomposition.should_decompose}, "
                f"subclaims={len(decomposition.subclaims)}, "
                f"confidence={decomposition.confidence:.2f}"
            )

        # Step 1.5: Entity-Aware Seed Discovery
        # Extract entities from query using TargetExtractor
        extractor = TargetExtractor(self.config)
        targets = extractor.extract(ExtractionInput(
            query_text=query_text,
            available_documents=[],
            known_entities=self.config.get("known_entities", []),
        ))
        
        logger.info(f"PLANNER Targets: explicit={targets.identified_targets.explicit_docs}, implicit={targets.identified_targets.implicit_entities}")

        # Multi-vector search: embed each entity separately
        entity_embeddings = []
        for entity in targets.identified_targets.implicit_entities:
            entity_embeddings.append(embed_query(entity, embedding_client))
        
        # Step 2: Find seeds with balanced seeding (Multi-Bucket Strategy)
        # NEW: Multi-source weighted seeding if decomposition enabled
        seed_weights = None
        if decomposition and decomposition.should_decompose:
            # Multi-source weighted seeding path
            subclaim_seeds = find_subclaim_seeds(
                subclaims=decomposition.subclaims,
                similarity_index=planner_input.similarity_index,
                graph=planner_input.graph,
                embedding_client=embedding_client,
                top_k_per_subclaim=5,
                threshold=0.70,
            )

            subclaim_priorities = {
                sc.subclaim_id: sc.priority for sc in decomposition.subclaims
            }

            weighted_seed_set = find_balanced_seeds(
                global_embedding=query_embedding,
                entity_embeddings=entity_embeddings,
                similarity_index=planner_input.similarity_index,
                graph=planner_input.graph,
                top_k=self.config.get('seed_top_k', 12),
                threshold=self.config.get('seed_threshold', 0.70),
                global_quota=self.config.get('seed_global_quota', 3),
                entity_quota=self.config.get('seed_entity_quota', 3),
                entity_texts=targets.identified_targets.implicit_entities,
                subclaim_seeds=subclaim_seeds,
                subclaim_priorities=subclaim_priorities,
            )

            seeds = weighted_seed_set.seeds
            seed_weights = self._compute_seed_weights(weighted_seed_set, decomposition)
        else:
            # Fallback to existing single-pass flow
            weighted_seed_set = find_balanced_seeds(
                global_embedding=query_embedding,
                entity_embeddings=entity_embeddings,
                similarity_index=planner_input.similarity_index,
                graph=planner_input.graph,
                top_k=self.config.get('seed_top_k', 12),
                threshold=self.config.get('seed_threshold', 0.70),
                global_quota=self.config.get('seed_global_quota', 3),
                entity_quota=self.config.get('seed_entity_quota', 3),
                entity_texts=targets.identified_targets.implicit_entities,
            )
            seeds = weighted_seed_set.seeds

        # Step 3: Classify strategy (moved before PPR to inform alpha)
        classification = self.classifier.classify(
            ClassificationInput(query_text=query_text)
        )

        # Step 4: PPR expand with query-adaptive damping
        # Multihop/relational queries need higher alpha (more diffusion)
        # to reach nodes 3-4 hops away. Lookup queries stay local.
        # P1b: Also force alpha=0.85 for multi-entity queries (2+ entities).
        seed_ids = [s.node_id for s in seeds]
        entity_count = len(targets.identified_targets.implicit_entities)
        ppr_alpha = self._select_ppr_alpha(classification.strategy, entity_count)
        ppr_scores = personalized_pagerank(
            planner_input.graph,
            seed_ids,
            alpha=ppr_alpha,
            max_iterations=self.config.get('ppr_max_iter', 50),
            tolerance=self.config.get('ppr_tolerance', 1e-6),
            degree_discount=False,  # Disabled: P2b edge-level discount is sufficient
            seed_weights=seed_weights,  # NEW: Pass weights for weighted PPR
        )

        # Extract scope with hardened thresholds: min_score=0.035, top_k=15
        scope = extract_scope(
            planner_input.graph,
            ppr_scores,
            seed_ids,
            top_k=self.config.get('scope_max_nodes', 25),
            min_score=self.config.get('scope_min_score', 0.005),
        )

        # Step 5: Route lane (v5 rules)
        lane = self._route_v5(classification.strategy, scope)

        # Step 6: Build router decision
        router_decision = RouterDecisionV5(
            query_id=query_id,
            search_strategy=classification.strategy,
            selected_lane=lane,
            seed_nodes=seeds,
            scope=scope,
            confidence=classification.strategy.confidence,
            claim_decomposition=decomposition,
        )

        # Step 7: Assemble context (Aggregator Pattern) with AliasMap
        focused: Optional[FocusedContext] = None
        global_ctx: Optional[GlobalContext] = None
        targeted: Optional[TargetedContext] = None
        focused_alias_map: Optional[AliasMap] = None
        targeted_alias_map: Optional[AliasMap] = None

        # --- Lane A: Graph Context ---
        if planner_input.graph and scope and len(scope.nodes) > 0:
            focused, focused_alias_map = assemble_focused_context(
                scope,
                planner_input.graph,
                planner_input.document_store or {},
            )

        # --- Lane C: Vector Context ---
        chunks = self._gather_chunks(planner_input)
        targeted, targeted_alias_map = assemble_targeted_context(
            scope,
            chunks,
            query_embedding,
        )

        # --- De-duplication: Prioritize Graph Context ---
        if focused and targeted and targeted.chunks:
            covered_chunk_ids = set()
            if planner_input.graph and getattr(planner_input.graph, "nodes", None):
                for node_id in scope.nodes:
                    graph_node = planner_input.graph.nodes.get(node_id)
                    if graph_node and hasattr(graph_node, "source_chunks"):
                        for cid in graph_node.source_chunks:
                            covered_chunk_ids.add(cid)
            
            if covered_chunk_ids:
                unique_chunks = [
                    c for c in targeted.chunks 
                    if c.chunk.chunk_id not in covered_chunk_ids
                ]
                targeted.chunks = unique_chunks

        # --- Merge alias maps ---
        merged_mappings = {}
        if focused_alias_map:
            merged_mappings.update(focused_alias_map.mappings)
        if targeted_alias_map:
            merged_mappings.update(targeted_alias_map.mappings)
        
        final_alias_map = AliasMap(mappings=merged_mappings) if merged_mappings else None

        return PlannerOutput(
            router_decision=router_decision,
            focused_context=focused,
            global_context=global_ctx,
            targeted_context=targeted,
            alias_map=final_alias_map,
        )

    # --- Multihop strategies that need longer PPR walks ---
    _MULTIHOP_STRATEGIES = frozenset({
        SearchStrategyType.COMPARISON,
        SearchStrategyType.EVIDENCE_SYNTHESIS,
        SearchStrategyType.METHODOLOGY_COMPARISON,
        SearchStrategyType.CONTRADICTION_DETECTION,
    })

    def _select_ppr_alpha(
        self,
        strategy: SearchStrategyClassification,
        entity_count: int = 0,
    ) -> float:
        """Choose PPR damping factor based on query strategy and entity count.

        Multi-entity queries (2+ entities) always use alpha=0.85 for longer
        diffusion, even if classified as LOOKUP. This handles cases like
        "Clara Vance was born in Oxford" (2 entities, spans 2 docs).

        Multihop/relational queries use alpha=0.85 (longer walks, ~52%
        chance of reaching 4-hop nodes). Lookup/verification queries
        use alpha=0.65 (tighter locality, ~18% at 4 hops).
        """
        if self.config.get('ppr_alpha') is not None:
            return self.config['ppr_alpha']  # explicit override wins

        # P1b: Force multihop alpha for multi-entity queries
        if entity_count >= 2:
            return self.config.get('ppr_alpha_multihop', 0.85)

        if strategy.classification in self._MULTIHOP_STRATEGIES:
            return self.config.get('ppr_alpha_multihop', 0.85)
        return self.config.get('ppr_alpha_local', 0.65)

    def _compute_seed_weights(
        self,
        weighted_seed_set: WeightedSeedSet,
        decomposition: ClaimDecomposition
    ) -> Dict[str, float]:
        """Compute PPR initial weights based on subclaim priority.

        Priority multipliers:
        - primary: 2.0x (main subject claims)
        - bridge: 1.5x (connecting entities)
        - secondary: 1.0x (supporting claims)

        Args:
            weighted_seed_set: The weighted seed set with node IDs
            decomposition: The claim decomposition with subclaims

        Returns:
            Dict mapping node_id to initial weight for PPR
        """
        PRIORITY_WEIGHTS = {
            "primary": self.decomposer_config.get('primary_weight_multiplier', 2.0),
            "bridge": self.decomposer_config.get('bridge_weight_multiplier', 1.5),
            "secondary": 1.0,
        }

        weights: Dict[str, float] = {}

        # Build node_id -> priority mapping
        node_to_priority: Dict[str, str] = {}

        # Map primary nodes
        for node_id in weighted_seed_set.primary_node_ids:
            node_to_priority[node_id] = "primary"

        # Map bridge nodes
        for node_id in weighted_seed_set.bridge_node_ids:
            node_to_priority[node_id] = "bridge"

        # Assign weights to all seeds
        for seed in weighted_seed_set.seeds:
            priority = node_to_priority.get(seed.node_id, "secondary")
            weights[seed.node_id] = PRIORITY_WEIGHTS[priority]

        return weights

    def _route_v5(
        self, strategy: SearchStrategyClassification, scope: Scope,
    ) -> LaneType:
        """V5 routing rules.

        - Low confidence or sparse scope -> LANE_C (Vector Only)
        - Otherwise -> LANE_A (Graph LCI)
        
        This enables structural reasoning for LOOKUP and CLAIM_VERIFICATION
        when the graph provides a sufficient scope.
        """
        confidence_threshold = self.config.get('confidence_threshold', 0.7)
        if strategy.confidence < confidence_threshold:
            return LaneType.LANE_C_VECTOR_ONLY

        node_count = len(scope.nodes)
        if node_count < 5:
            return LaneType.LANE_C_VECTOR_ONLY

        # Default to Lane A (Graph LCI) if we have a valid scope and high confidence,
        # regardless of strategy type. Graph structure aids all reasoning tasks.
        return LaneType.LANE_A_GRAPH_LCI


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
        # SENSEMAKING test case removed

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
            "query": "Does the MSA contradict DORA?",
            "query_id": "q7",
            "validate_fields": True,
            "expected_lane": LaneType.LANE_A_GRAPH_LCI,
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
                if rd.search_strategy is None:
                    all_ok = False
                    issues.append("search_strategy is None")
                if rd.identified_targets is None:
                    all_ok = False
                    issues.append("identified_targets is None")
                if rd.domain_expectation is None:
                    all_ok = False
                    issues.append("domain_expectation is None")
                if rd.search_strategy and not isinstance(rd.search_strategy.classification, SearchStrategyType):
                    all_ok = False
                    issues.append("search_strategy.classification not SearchStrategyType")
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
            print(f"  Strategy:    {rd.search_strategy.classification.value} "
                  f"(conf={rd.search_strategy.confidence})")
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
