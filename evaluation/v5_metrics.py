"""
Stage 1 (v5): Intrinsic Validation
Location: evaluation/v5_metrics.py
Owner: ROOT Instance

Quantitative scores computed independently from raw v5 module outputs.
No pre-computed values are read from module internals — all metrics are
derived by the evaluation framework itself.

Mirrors the v4 IntrinsicValidator / PipelineSnapshot pattern.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from schemas.contracts import (
    ChunkerOutput,
    ExplorerOutput,
    LibrarianOutput,
    PlannerOutput,
    QueryRequest,
    RawDocument,
    ReasonerOutput,
)
from schemas.evaluation_contracts import (
    ChunkerMetrics,
    ExplorerMetrics,
    LibrarianMetrics,
    PlannerMetrics,
    ReasonerMetrics,
    V5MetricScore,
    V5PipelineMetrics,
)


# ============================================================================
# V5 PIPELINE SNAPSHOT
# ============================================================================

@dataclass
class V5PipelineSnapshot:
    """All intermediate outputs from one v5 pipeline run."""
    documents: List[RawDocument]
    query: QueryRequest
    chunker_output: Optional[ChunkerOutput] = None
    explorer_output: Optional[ExplorerOutput] = None
    librarian_output: Optional[LibrarianOutput] = None
    planner_output: Optional[PlannerOutput] = None
    reasoner_output: Optional[ReasonerOutput] = None


# ============================================================================
# V5 INTRINSIC VALIDATOR
# ============================================================================

class V5IntrinsicValidator:
    """Validate v5 pipeline outputs without ground truth.

    Computes 13 metrics across 4 modules:

    Explorer (4):
      - node_coverage: fraction of input docs with >= 1 node
      - entity_yield: entities per chunk
      - claim_yield: claims per chunk
      - embedding_validity: fraction of embeddings that are non-zero finite

    Librarian (3):
      - graph_connectivity: fraction of nodes reachable via BFS from any node
      - bridge_edge_density: bridge_edges / total edges
      - document_coverage: fraction of input docs with >= 1 graph node

    Planner (3):
      - seed_discovery: number of seed nodes found
      - scope_coverage: fraction of input docs in scope
      - lane_consistency: intent matches expected lane routing

    Reasoner (3):
      - evidence_grounding_rate: fraction of evidence with non-empty exact_quote
      - relationship_completeness: number of relationships inferred
      - confidence_validity: overall confidence > 0
    """

    def __init__(self, snapshot: V5PipelineSnapshot) -> None:
        self._snap = snapshot

    def validate(self) -> V5PipelineMetrics:
        chunker = self._validate_chunker()
        explorer = self._validate_explorer()
        librarian = self._validate_librarian()
        planner = self._validate_planner()
        reasoner = self._validate_reasoner()

        overall = (
            explorer.all_passed
            and librarian.all_passed
            and planner.all_passed
            and reasoner.all_passed
        )
        if chunker is not None:
            overall = overall and chunker.all_passed

        return V5PipelineMetrics(
            explorer=explorer,
            librarian=librarian,
            planner=planner,
            reasoner=reasoner,
            overall_pass=overall,
            chunker=chunker,
        )

    # ------------------------------------------------------------------
    # CHUNKER metrics
    # ------------------------------------------------------------------

    def _validate_chunker(self) -> Optional[ChunkerMetrics]:
        """Validate chunker output. Returns None if no chunker output."""
        out = self._snap.chunker_output
        if out is None:
            return None

        input_doc_ids = {d.doc_id for d in self._snap.documents}
        num_docs = len(input_doc_ids)

        # chunk_count: at least 1 chunk produced
        cc_value = float(len(out.chunks))
        chunk_count = V5MetricScore(
            name="chunk_count",
            value=cc_value,
            threshold=1.0,
            passed=cc_value >= 1.0,
            detail=f"{int(cc_value)} chunks produced",
        )

        # doc_coverage: every input doc appears in source_doc_ids
        chunked_doc_ids = set(out.source_doc_ids)
        dc_value = len(chunked_doc_ids & input_doc_ids) / num_docs if num_docs > 0 else 0.0
        doc_coverage = V5MetricScore(
            name="doc_coverage",
            value=dc_value,
            threshold=1.0,
            passed=dc_value >= 1.0,
            detail=f"{len(chunked_doc_ids & input_doc_ids)}/{num_docs} docs chunked",
        )

        return ChunkerMetrics(
            chunk_count=chunk_count,
            doc_coverage=doc_coverage,
        )

    # ------------------------------------------------------------------
    # EXPLORER metrics
    # ------------------------------------------------------------------

    def _validate_explorer(self) -> ExplorerMetrics:
        out = self._snap.explorer_output
        input_doc_ids = {d.doc_id for d in self._snap.documents}
        num_docs = len(input_doc_ids)

        if out is None:
            return ExplorerMetrics(
                node_coverage=V5MetricScore("node_coverage", 0.0, 0.8, False, "No explorer output"),
                entity_yield=V5MetricScore("entity_yield", 0.0, 1.0, False, "No explorer output"),
                claim_yield=V5MetricScore("claim_yield", 0.0, 0.5, False, "No explorer output"),
                embedding_validity=V5MetricScore("embedding_validity", 0.0, 0.95, False, "No explorer output"),
            )

        ns = out.node_set

        # node_coverage: docs with >= 1 node
        docs_with_nodes = set(ns.source_doc_ids)
        cov_value = len(docs_with_nodes & input_doc_ids) / num_docs if num_docs > 0 else 0.0
        node_coverage = V5MetricScore(
            name="node_coverage",
            value=cov_value,
            threshold=0.8,
            passed=cov_value >= 0.8,
            detail=f"{len(docs_with_nodes & input_doc_ids)}/{num_docs} docs have nodes",
        )

        # entity_yield: entities / chunks
        chunk_count = ns.chunk_count if ns.chunk_count > 0 else 1
        ey_value = ns.entity_count / chunk_count
        entity_yield = V5MetricScore(
            name="entity_yield",
            value=ey_value,
            threshold=1.0,
            passed=ey_value >= 1.0,
            detail=f"{ns.entity_count} entities / {chunk_count} chunks",
        )

        # claim_yield: claims / chunks
        cy_value = ns.claim_count / chunk_count
        claim_yield = V5MetricScore(
            name="claim_yield",
            value=cy_value,
            threshold=0.5,
            passed=cy_value >= 0.5,
            detail=f"{ns.claim_count} claims / {chunk_count} chunks",
        )

        # embedding_validity: non-zero, finite embeddings
        total_nodes = len(ns.nodes)
        valid_embeddings = 0
        for node in ns.nodes:
            if node.embedding and len(node.embedding) > 0:
                all_finite = all(math.isfinite(v) for v in node.embedding)
                any_nonzero = any(v != 0.0 for v in node.embedding)
                if all_finite and any_nonzero:
                    valid_embeddings += 1

        ev_value = valid_embeddings / total_nodes if total_nodes > 0 else 0.0
        embedding_validity = V5MetricScore(
            name="embedding_validity",
            value=ev_value,
            threshold=0.95,
            passed=ev_value >= 0.95,
            detail=f"{valid_embeddings}/{total_nodes} embeddings valid",
        )

        return ExplorerMetrics(
            node_coverage=node_coverage,
            entity_yield=entity_yield,
            claim_yield=claim_yield,
            embedding_validity=embedding_validity,
        )

    # ------------------------------------------------------------------
    # LIBRARIAN metrics
    # ------------------------------------------------------------------

    def _validate_librarian(self) -> LibrarianMetrics:
        out = self._snap.librarian_output
        input_doc_ids = {d.doc_id for d in self._snap.documents}
        num_docs = len(input_doc_ids)

        if out is None:
            return LibrarianMetrics(
                graph_connectivity=V5MetricScore("graph_connectivity", 0.0, 0.5, False, "No librarian output"),
                bridge_edge_density=V5MetricScore("bridge_edge_density", 0.0, 0.05, False, "No librarian output"),
                document_coverage=V5MetricScore("document_coverage", 0.0, 0.8, False, "No librarian output"),
            )

        graph = out.graph

        # graph_connectivity: BFS reachability from any node
        gc_value = self._compute_bfs_reachability(graph.adjacency)
        graph_connectivity = V5MetricScore(
            name="graph_connectivity",
            value=gc_value,
            threshold=0.5,
            passed=gc_value >= 0.5,
            detail=f"{gc_value:.2f} fraction reachable via BFS",
        )

        # bridge_edge_density
        total_edges = len(graph.edges)
        bridge_count = len(graph.bridge_edges) if graph.bridge_edges else 0
        bed_value = bridge_count / total_edges if total_edges > 0 else 0.0
        bridge_edge_density = V5MetricScore(
            name="bridge_edge_density",
            value=bed_value,
            threshold=0.05,
            passed=bed_value >= 0.05,
            detail=f"{bridge_count}/{total_edges} bridge edges",
        )

        # document_coverage: docs represented in graph
        docs_in_graph: Set[str] = set()
        for doc_id, node_ids in graph.node_by_doc.items():
            if node_ids:
                docs_in_graph.add(doc_id)
        dc_value = len(docs_in_graph & input_doc_ids) / num_docs if num_docs > 0 else 0.0
        document_coverage = V5MetricScore(
            name="document_coverage",
            value=dc_value,
            threshold=0.8,
            passed=dc_value >= 0.8,
            detail=f"{len(docs_in_graph & input_doc_ids)}/{num_docs} docs in graph",
        )

        return LibrarianMetrics(
            graph_connectivity=graph_connectivity,
            bridge_edge_density=bridge_edge_density,
            document_coverage=document_coverage,
        )

    @staticmethod
    def _compute_bfs_reachability(adjacency: Dict[str, List[str]]) -> float:
        """Fraction of nodes reachable from the node with highest degree."""
        if not adjacency:
            return 0.0

        all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            all_nodes.update(neighbors)

        if not all_nodes:
            return 0.0

        # Start BFS from node with highest degree
        start = max(adjacency, key=lambda n: len(adjacency.get(n, [])))

        visited: Set[str] = set()
        queue: deque[str] = deque([start])
        visited.add(start)

        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) / len(all_nodes)

    # ------------------------------------------------------------------
    # PLANNER metrics
    # ------------------------------------------------------------------

    def _validate_planner(self) -> PlannerMetrics:
        out = self._snap.planner_output
        input_doc_ids = {d.doc_id for d in self._snap.documents}
        num_docs = len(input_doc_ids)

        if out is None:
            return PlannerMetrics(
                seed_discovery=V5MetricScore("seed_discovery", 0.0, 1.0, False, "No planner output"),
                scope_coverage=V5MetricScore("scope_coverage", 0.0, 0.3, False, "No planner output"),
                lane_consistency=V5MetricScore("lane_consistency", 0.0, 1.0, False, "No planner output"),
            )

        rd = out.router_decision

        # seed_discovery: number of seed nodes
        seed_count = len(rd.seed_nodes)
        sd_value = float(seed_count)
        seed_discovery = V5MetricScore(
            name="seed_discovery",
            value=sd_value,
            threshold=1.0,
            passed=sd_value >= 1.0,
            detail=f"{seed_count} seed nodes found",
        )

        # scope_coverage: docs in scope / input docs
        scope = rd.scope
        if scope is not None:
            scope_doc_ids = set(scope.document_ids)
            sc_value = len(scope_doc_ids & input_doc_ids) / num_docs if num_docs > 0 else 0.0
        else:
            sc_value = 0.0
        scope_coverage = V5MetricScore(
            name="scope_coverage",
            value=sc_value,
            threshold=0.3,
            passed=sc_value >= 0.3,
            detail=f"{sc_value:.2f} doc coverage in scope",
        )

        # lane_consistency: intent→lane routing rules
        lc_value = self._check_lane_consistency(rd)
        lane_consistency = V5MetricScore(
            name="lane_consistency",
            value=lc_value,
            threshold=1.0,
            passed=lc_value == 1.0,
            detail="lane matches intent routing rules" if lc_value == 1.0 else "lane/intent mismatch",
        )

        return PlannerMetrics(
            seed_discovery=seed_discovery,
            scope_coverage=scope_coverage,
            lane_consistency=lane_consistency,
        )

    @staticmethod
    def _check_lane_consistency(rd) -> float:
        """Check that strategy classification maps to the correct lane.

        - COMPARISON, CONTRADICTION_DETECTION, CLAIM_VERIFICATION -> LANE_A
        - LOOKUP -> LANE_C
        """
        from schemas.contracts import SearchStrategyType, LaneType

        strategy = rd.search_strategy.classification
        lane = rd.selected_lane

        lane_a_strategies = {
            SearchStrategyType.COMPARISON,
            SearchStrategyType.CONTRADICTION_DETECTION,
            SearchStrategyType.CLAIM_VERIFICATION,
        }
        
        lane_c_strategies = {
            SearchStrategyType.LOOKUP,
        }

        if strategy in lane_a_strategies and lane == LaneType.LANE_A_GRAPH_LCI:
            return 1.0
        
        if strategy in lane_c_strategies and lane == LaneType.LANE_C_VECTOR_ONLY:
            return 1.0

        # Not a strict failure for strategies that could reasonably go either way
        return 0.0

    # ------------------------------------------------------------------
    # REASONER metrics
    # ------------------------------------------------------------------

    def _validate_reasoner(self) -> ReasonerMetrics:
        out = self._snap.reasoner_output

        if out is None:
            return ReasonerMetrics(
                evidence_grounding_rate=V5MetricScore("evidence_grounding_rate", 0.0, 0.5, False, "No reasoner output"),
                relationship_completeness=V5MetricScore("relationship_completeness", 0.0, 1.0, False, "No reasoner output"),
                confidence_validity=V5MetricScore("confidence_validity", 0.0, 0.0, False, "No reasoner output"),
            )

        finding = out.finding

        # evidence_grounding_rate: fraction with non-empty exact_quote
        total_evidence = len(finding.evidence)
        grounded = sum(
            1 for e in finding.evidence if e.exact_quote and e.exact_quote.strip()
        )
        egr_value = grounded / total_evidence if total_evidence > 0 else 0.0
        evidence_grounding_rate = V5MetricScore(
            name="evidence_grounding_rate",
            value=egr_value,
            threshold=0.5,
            passed=egr_value >= 0.5,
            detail=f"{grounded}/{total_evidence} evidence items grounded",
        )

        # relationship_completeness: at least 1 relationship
        rel_count = len(finding.relationships)
        rc_value = float(rel_count)
        relationship_completeness = V5MetricScore(
            name="relationship_completeness",
            value=rc_value,
            threshold=1.0,
            passed=rc_value >= 1.0,
            detail=f"{rel_count} relationships inferred",
        )

        # confidence_validity: overall confidence > 0
        conf = finding.confidence_factors.overall
        cv_passed = conf > 0
        confidence_validity = V5MetricScore(
            name="confidence_validity",
            value=conf,
            threshold=0.0,
            passed=cv_passed,
            detail=f"overall confidence = {conf:.3f}",
        )

        return ReasonerMetrics(
            evidence_grounding_rate=evidence_grounding_rate,
            relationship_completeness=relationship_completeness,
            confidence_validity=confidence_validity,
        )
