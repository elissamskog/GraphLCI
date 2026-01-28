"""
Tests for the v5 PLANNER pipeline.
Location: planner/tests/test_v5_pipeline.py

Tests cover: query_embedder, seed_finder, ppr_expander, context_assembler,
and the full PlannerControllerV5 pipeline.
"""

from __future__ import annotations

import hashlib
import math
import sys
import os

# Ensure project root and planner dir are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any, Dict, List, Optional, Tuple

from schemas.contracts import (
    Community,
    CommunityStructure,
    EmbeddedNode,
    FocusedContext,
    GlobalContext,  # Deprecated
    IdentifiedTargets,
    SearchStrategyClassification,
    SearchStrategyType,
    KnowledgeGraph,
    LaneType,
    PlannerInput,
    PlannerOutput,
    QueryRequest,
    RankedChunk,
    RawDocument,
    RouterDecisionV5,
    Scope,
    ScopedNode,
    SeedNode,
    StructuralEdge,
    StructuralEdgeType,
    TargetedContext,
    TextChunk,
)

from planner.search_strategy_classifier import MockLLMClient
from planner.query_embedder import embed_query
from planner.seed_finder import find_seed_nodes
from planner.ppr_expander import personalized_pagerank, extract_scope
from planner.context_assembler import (
    assemble_focused_context,
    # assemble_global_context,
    assemble_targeted_context,
)
from planner_controller import PlannerControllerV5


# ============================================================================
# TEST INFRASTRUCTURE
# ============================================================================

EMBEDDING_DIM = 8


class MockEmbeddingClient:
    """Deterministic embedding client for tests."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:EMBEDDING_DIM]]
            # Normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results


class MockSimilarityIndex:
    """In-memory similarity index for tests.

    Stores (node_id, embedding) pairs and returns cosine-similar results.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, List[float]]] = []

    def add(self, node_id: str, embedding: List[float]) -> None:
        self._entries.append((node_id, embedding))

    def query(
        self, embedding: List[float], top_k: int,
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for node_id, node_emb in self._entries:
            sim = _cosine_similarity(embedding, node_emb)
            scored.append((node_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _make_embedding(text: str) -> List[float]:
    """Generate a deterministic embedding from text."""
    client = MockEmbeddingClient()
    return client.embed([text])[0]


def build_test_graph() -> KnowledgeGraph:
    """Build a small test knowledge graph with ~10 nodes across 2 docs."""
    nodes: Dict[str, EmbeddedNode] = {}
    edges: Dict[str, StructuralEdge] = {}
    adjacency: Dict[str, List[str]] = {}
    node_by_doc: Dict[str, List[str]] = {}

    # Document 1: MSA nodes
    for i, (text, ntype) in enumerate([
        ("Liability Cap", "ENTITY"),
        ("Termination Fee", "ENTITY"),
        ("Governing Law", "CLAUSE"),
        ("Indemnification", "CLAUSE"),
        ("Vendor", "ENTITY"),
    ]):
        nid = f"msa_n{i}"
        nodes[nid] = EmbeddedNode(
            node_id=nid,
            node_type=ntype,
            canonical_text=text,
            embedding=_make_embedding(f"MSA {text}"),
            source_doc_id="MSA",
            source_chunks=[f"chunk_MSA_{i}"],
            metadata={},
        )
        node_by_doc.setdefault("MSA", []).append(nid)
        adjacency.setdefault(nid, [])

    # Document 2: DORA nodes
    for i, (text, ntype) in enumerate([
        ("Liability Limitation", "ENTITY"),
        ("Compliance Requirement", "CLAUSE"),
        ("Data Protection", "CLAUSE"),
        ("Vendor Obligation", "ENTITY"),
        ("Penalty Clause", "CLAUSE"),
    ]):
        nid = f"dora_n{i}"
        nodes[nid] = EmbeddedNode(
            node_id=nid,
            node_type=ntype,
            canonical_text=text,
            embedding=_make_embedding(f"DORA {text}"),
            source_doc_id="DORA",
            source_chunks=[f"chunk_DORA_{i}"],
            metadata={},
        )
        node_by_doc.setdefault("DORA", []).append(nid)
        adjacency.setdefault(nid, [])

    # Edges within MSA
    _add_edge(edges, adjacency, "e1", "msa_n0", "msa_n1",
              StructuralEdgeType.SAME_SOURCE, 1.0)
    _add_edge(edges, adjacency, "e2", "msa_n2", "msa_n3",
              StructuralEdgeType.SAME_SOURCE, 1.0)
    _add_edge(edges, adjacency, "e3", "msa_n3", "msa_n4",
              StructuralEdgeType.REFERENCES, 0.8)

    # Edges within DORA
    _add_edge(edges, adjacency, "e4", "dora_n0", "dora_n1",
              StructuralEdgeType.SAME_SOURCE, 1.0)
    _add_edge(edges, adjacency, "e5", "dora_n2", "dora_n3",
              StructuralEdgeType.REFERENCES, 0.7)

    # Cross-document edges (bridges)
    _add_edge(edges, adjacency, "e6", "msa_n0", "dora_n0",
              StructuralEdgeType.SYNONYMY, 0.9,
              is_cross_document=True, similarity=0.92)
    _add_edge(edges, adjacency, "e7", "msa_n4", "dora_n3",
              StructuralEdgeType.SYNONYMY, 0.85,
              is_cross_document=True, similarity=0.88)

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        adjacency=adjacency,
        node_by_doc=node_by_doc,
        node_count=len(nodes),
        edge_count=len(edges),
        document_count=2,
        bridge_count=2,
        bridge_edges=["e6", "e7"],
    )


def _add_edge(
    edges: Dict[str, StructuralEdge],
    adjacency: Dict[str, List[str]],
    edge_id: str,
    source: str,
    target: str,
    edge_type: StructuralEdgeType,
    weight: float,
    is_cross_document: bool = False,
    similarity: Optional[float] = None,
) -> None:
    edges[edge_id] = StructuralEdge(
        edge_id=edge_id,
        source=source,
        target=target,
        edge_type=edge_type,
        weight=weight,
        metadata={},
        similarity=similarity,
        is_cross_document=is_cross_document,
    )
    adjacency.setdefault(source, []).append(edge_id)
    adjacency.setdefault(target, []).append(edge_id)


def build_test_similarity_index(graph: KnowledgeGraph) -> MockSimilarityIndex:
    """Build a similarity index from graph nodes."""
    index = MockSimilarityIndex()
    for node in graph.nodes.values():
        index.add(node.node_id, node.embedding)
    return index


def build_test_community_structure(
    graph: KnowledgeGraph,
) -> CommunityStructure:
    """Build a simple 2-community structure: one per document."""
    communities: Dict[str, Community] = {}
    node_to_community: Dict[str, str] = {}

    msa_nodes = [nid for nid, n in graph.nodes.items()
                 if n.source_doc_id == "MSA"]
    dora_nodes = [nid for nid, n in graph.nodes.items()
                  if n.source_doc_id == "DORA"]

    communities["comm_msa"] = Community(
        community_id="comm_msa",
        level=0,
        node_ids=msa_nodes,
        summary="MSA community: covers liability, termination, governance, indemnification.",
    )
    communities["comm_dora"] = Community(
        community_id="comm_dora",
        level=0,
        node_ids=dora_nodes,
        summary="DORA community: covers compliance, data protection, vendor obligations.",
    )

    for nid in msa_nodes:
        node_to_community[nid] = "comm_msa"
    for nid in dora_nodes:
        node_to_community[nid] = "comm_dora"

    return CommunityStructure(
        communities=communities,
        levels={0: ["comm_msa", "comm_dora"]},
        node_to_community=node_to_community,
    )


def build_test_document_store() -> Dict[str, RawDocument]:
    """Build a test document store."""
    return {
        "MSA": RawDocument(
            doc_id="MSA",
            content="Master Service Agreement content with liability cap and termination provisions.",
            metadata={"type": "contract"},
        ),
        "DORA": RawDocument(
            doc_id="DORA",
            content="Digital Operational Resilience Act with compliance requirements.",
            metadata={"type": "regulation"},
        ),
    }


def build_test_chunks(graph: KnowledgeGraph) -> List[TextChunk]:
    """Build TextChunks that correspond to graph nodes' source_chunks."""
    chunks: List[TextChunk] = []
    seen: set = set()
    position = 0
    for node in graph.nodes.values():
        for chunk_id in node.source_chunks:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                source_doc_id=node.source_doc_id,
                text=f"Chunk text for {node.canonical_text}",
                token_count=10,
                position=position,
            ))
            position += 1
    return chunks


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_embed_query():
    """Verify embed_query returns a list of floats matching embedding dim."""
    client = MockEmbeddingClient()
    result = embed_query("What is the liability cap?", client)
    assert isinstance(result, list)
    assert len(result) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in result)


def test_find_seed_nodes():
    """Verify seeds found, sorted by similarity, filtered by threshold."""
    graph = build_test_graph()
    index = build_test_similarity_index(graph)
    # Use an embedding close to a known node
    query_emb = _make_embedding("MSA Liability Cap")
    seeds = find_seed_nodes(query_emb, index, graph, top_k=5, threshold=0.0)
    assert len(seeds) > 0
    assert len(seeds) <= 5
    # Sorted descending
    for i in range(len(seeds) - 1):
        assert seeds[i].similarity >= seeds[i + 1].similarity
    # First seed should be the exact match
    assert seeds[0].node_id == "msa_n0"
    assert seeds[0].similarity > 0.99  # near-exact embedding match


def test_find_seed_nodes_threshold_filter():
    """All below threshold -> empty list."""
    graph = build_test_graph()
    index = build_test_similarity_index(graph)
    query_emb = _make_embedding("MSA Liability Cap")
    # Set threshold impossibly high
    seeds = find_seed_nodes(query_emb, index, graph, top_k=5, threshold=1.01)
    assert seeds == []


def test_personalized_pagerank():
    """Verify seed nodes get highest scores, scores sum to ~1.0."""
    graph = build_test_graph()
    seed_ids = ["msa_n0", "dora_n0"]
    scores = personalized_pagerank(graph, seed_ids, alpha=0.85, max_iterations=100)
    assert len(scores) == len(graph.nodes)
    # Seeds should have high scores
    for sid in seed_ids:
        assert scores[sid] > 0
    # Scores should roughly sum to 1.0 (PPR is a probability distribution)
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.1, f"PPR total={total}, expected ~1.0"
    # Seeds should be among the top scorers
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = {nid for nid, _ in sorted_scores[:4]}
    assert "msa_n0" in top_ids
    assert "dora_n0" in top_ids


def test_personalized_pagerank_no_valid_seeds():
    """Empty/invalid seeds -> all-zero scores."""
    graph = build_test_graph()
    scores = personalized_pagerank(graph, ["nonexistent_node"])
    assert all(s == 0.0 for s in scores.values())


def test_extract_scope():
    """Verify correct node count, edges between selected nodes, doc IDs."""
    graph = build_test_graph()
    seed_ids = ["msa_n0", "dora_n0"]
    scores = personalized_pagerank(graph, seed_ids)
    scope = extract_scope(graph, scores, seed_ids, top_k=6, min_score=0.001)

    assert len(scope.nodes) <= 6
    assert len(scope.nodes) > 0
    # Seed nodes should be in scope
    assert "msa_n0" in scope.nodes
    assert "dora_n0" in scope.nodes
    # Nodes should have ranks
    for sn in scope.nodes.values():
        assert sn.rank >= 1
    # Edges should only connect nodes within scope
    for edge in scope.edges.values():
        assert edge.source in scope.nodes
        assert edge.target in scope.nodes
    # Document IDs should be populated
    assert len(scope.document_ids) > 0
    assert scope.total_ppr_mass > 0


def test_assemble_focused_context():
    """Verify documents, edge grammar, node context for Lane A."""
    graph = build_test_graph()
    seed_ids = ["msa_n0", "dora_n0"]
    scores = personalized_pagerank(graph, seed_ids)
    scope = extract_scope(graph, scores, seed_ids, top_k=10, min_score=0.001)
    doc_store = build_test_document_store()

    ctx = assemble_focused_context(scope, graph, doc_store)
    assert isinstance(ctx, FocusedContext)
    assert len(ctx.documents) > 0
    assert isinstance(ctx.edge_grammar, str)
    assert len(ctx.edge_grammar) > 0
    assert isinstance(ctx.node_context, str)
    assert len(ctx.node_context) > 0
    assert ctx.scope is scope


# test_assemble_global_context removed



def test_assemble_targeted_context():
    """Verify chunks ranked and capped at 20."""
    graph = build_test_graph()
    seed_ids = ["msa_n0", "dora_n0"]
    scores = personalized_pagerank(graph, seed_ids)
    scope = extract_scope(graph, scores, seed_ids, top_k=10, min_score=0.001)
    chunks = build_test_chunks(graph)
    query_emb = _make_embedding("liability")

    ctx = assemble_targeted_context(scope, chunks, query_emb)
    assert isinstance(ctx, TargetedContext)
    assert len(ctx.chunks) <= 20
    assert len(ctx.chunks) > 0
    # Ranked descending
    for i in range(len(ctx.chunks) - 1):
        assert ctx.chunks[i].relevance_score >= ctx.chunks[i + 1].relevance_score
    assert ctx.scope is scope


# ============================================================================
# FULL PIPELINE TESTS
# ============================================================================

def _build_pipeline_config() -> Dict[str, Any]:
    graph = build_test_graph()
    return {
        'embedding_client': MockEmbeddingClient(),
        'llm_client': MockLLMClient(),  # LLM client for classifier
        'seed_top_k': 5,
        'seed_threshold': 0.0,  # low threshold for tests
        'ppr_alpha': 0.85,
        'ppr_max_iter': 50,
        'ppr_tolerance': 1e-6,
        'scope_max_nodes': 50,
        'scope_min_score': 0.001,
        'confidence_threshold': 0.5,
        '_test_graph': graph,
    }


def _build_planner_input(
    query_text: str,
    config: Dict[str, Any],
    include_community: bool = True,
    include_docs: bool = True,
) -> PlannerInput:
    graph = config['_test_graph']
    index = build_test_similarity_index(graph)
    community_structure = (
        build_test_community_structure(graph) if include_community else None
    )
    doc_store = build_test_document_store() if include_docs else None

    return PlannerInput(
        query=QueryRequest(query_id="test_q", query_text=query_text),
        graph=graph,
        similarity_index=index,
        community_structure=community_structure,
        document_store=doc_store,
    )


def test_full_pipeline():
    """End-to-end PlannerControllerV5.process() produces valid PlannerOutput."""
    config = _build_pipeline_config()
    controller = PlannerControllerV5(config)
    planner_input = _build_planner_input(
        "Does the MSA contradict DORA?", config
    )

    output = controller.process(planner_input)
    assert isinstance(output, PlannerOutput)
    assert isinstance(output.router_decision, RouterDecisionV5)
    assert output.router_decision.query_id == "test_q"
    assert isinstance(output.router_decision.search_strategy, SearchStrategyClassification)
    assert isinstance(output.router_decision.selected_lane, LaneType)
    assert len(output.router_decision.seed_nodes) > 0
    assert output.router_decision.scope is not None

    # Aggregator Pattern: Both contexts should be populated if data exists
    # Lane A query
    if output.router_decision.selected_lane == LaneType.LANE_A_GRAPH_LCI:
        assert output.focused_context is not None
        assert output.targeted_context is not None
    # Lane C query
    elif output.router_decision.selected_lane == LaneType.LANE_C_VECTOR_ONLY:
        assert output.targeted_context is not None
        # focused_context populated if graph data available



def test_pipeline_lane_a():
    """COMPARISON query -> LANE_A -> focused_context populated."""
    config = _build_pipeline_config()
    controller = PlannerControllerV5(config)
    planner_input = _build_planner_input(
        "Does the MSA contradict DORA?", config
    )

    output = controller.process(planner_input)
    output = controller.process(planner_input)
    assert output.router_decision.search_strategy.classification == SearchStrategyType.COMPARISON
    assert output.router_decision.selected_lane == LaneType.LANE_A_GRAPH_LCI
    assert output.focused_context is not None
    assert output.focused_context.documents is not None
    assert output.global_context is None
    # Aggregator: Targeted context should also be populated
    assert output.targeted_context is not None



# Lane B test removed



def test_pipeline_lane_c():
    """LOOKUP query -> LANE_C -> targeted_context populated."""
    config = _build_pipeline_config()
    controller = PlannerControllerV5(config)
    planner_input = _build_planner_input(
        "What is the termination fee?", config
    )

    output = controller.process(planner_input)
    output = controller.process(planner_input)
    assert output.router_decision.search_strategy.classification == SearchStrategyType.LOOKUP
    assert output.router_decision.selected_lane == LaneType.LANE_C_VECTOR_ONLY
    assert output.targeted_context is not None
    # focused_context can be populated if graph exists, even if Lane C chosen
    assert output.global_context is None



def _build_large_test_graph(doc_count: int = 7) -> KnowledgeGraph:
    """Build a graph with many documents to trigger LANE_B routing."""
    nodes: Dict[str, EmbeddedNode] = {}
    edges: Dict[str, StructuralEdge] = {}
    adjacency: Dict[str, List[str]] = {}
    node_by_doc: Dict[str, List[str]] = {}

    edge_counter = 0
    prev_node_id: Optional[str] = None

    for d in range(doc_count):
        doc_id = f"DOC_{d}"
        for j in range(2):
            nid = f"doc{d}_n{j}"
            nodes[nid] = EmbeddedNode(
                node_id=nid,
                node_type="ENTITY",
                canonical_text=f"Entity {d}-{j}",
                embedding=_make_embedding(f"{doc_id} Entity {j}"),
                source_doc_id=doc_id,
                source_chunks=[f"chunk_{doc_id}_{j}"],
                metadata={},
            )
            node_by_doc.setdefault(doc_id, []).append(nid)
            adjacency.setdefault(nid, [])

            if prev_node_id is not None:
                eid = f"e_large_{edge_counter}"
                edge_counter += 1
                _add_edge(edges, adjacency, eid, prev_node_id, nid,
                          StructuralEdgeType.SAME_SOURCE, 0.8)
            prev_node_id = nid

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        adjacency=adjacency,
        node_by_doc=node_by_doc,
        node_count=len(nodes),
        edge_count=len(edges),
        document_count=doc_count,
    )


def test_aggregator_pattern_deduplication():
    """Verify that chunks present in FocusedContext are removed from TargetedContext."""
    config = _build_pipeline_config()
    controller = PlannerControllerV5(config)
    
    # Setup graph and chunks such that there is overlap.
    # build_test_graph() has nodes with source_chunks=['chunk_MSA_0', 'chunk_MSA_1', ...]
    # _build_planner_input creates chunks from document_store manually in _gather_chunks 
    # if planner_input.chunks is None.
    # PlannerControllerV5._gather_chunks builds synthetic chunks: "chunk_{doc_id}_0"
    
    # To test deduplication effectively, we need the *targeted* chunks to match the *graph* chunks.
    # Graph nodes in build_test_graph have source_chunks like "chunk_MSA_0".
    # _gather_chunks builds "chunk_MSA_0" (position 0). 
    
    # Let's verify what _gather_chunks produces for 'MSA':
    # chunk_id=f"chunk_{doc_id}_0" -> "chunk_MSA_0"
    # graph node msa_n0 has source_chunks=["chunk_MSA_0"]
    
    # So overlap should happen automatically for chunk_MSA_0!
    
    planner_input = _build_planner_input(
        "Does the MSA contradict DORA?", config
    )
    
    output = controller.process(planner_input)
    assert output.focused_context is not None
    assert output.targeted_context is not None
    
    # Check deduplication
    # Graph has msa_n0 which sources chunk_MSA_0.
    # Targeted context would normally include chunk_MSA_0 because it's in the doc store.
    
    # Inspect targeted chunks
    t_chunk_ids = [c.chunk.chunk_id for c in output.targeted_context.chunks]
    
    # Verify that 'chunk_MSA_0' is NOT in targeted chunks (deduplicated)
    # But wait, does scope include msa_n0?
    # Scope depends on PPR. msa_n0 is a seed (exact match for "MSA Liability Cap" logic in tests? no query is "Does MSA contradict...")
    # In test_full_pipeline, we see seeds found.
    # Let's ensure msa_n0 is in scope. 
    # The query "Does the MSA contradict DORA?" might not hit msa_n0 ("Liability Cap") directly unless embedding matches.
    # But PPR expands.
    
    # If msa_n0 is in scope, then chunk_MSA_0 must be removed from targeted.
    
    # Let's verify if *any* deduplication happened.
    # If no deduplication, targeted chunks would be all chunks.
    
    # Manually check scope nodes to see what should be removed
    scope_node_ids = set(output.router_decision.scope.nodes.keys())
    # Retrieve covering chunks from graph
    covered_chunks = set()
    graph = planner_input.graph
    for nid in scope_node_ids:
        node = graph.nodes.get(nid)
        if node and hasattr(node, "source_chunks"):
             for cid in node.source_chunks:
                 covered_chunks.add(cid)
                 
    # Assert that none of covered_chunks are in targeted_context
    for tc in output.targeted_context.chunks:
        assert tc.chunk.chunk_id not in covered_chunks, f"Chunk {tc.chunk.chunk_id} should have been deduplicated!"
        
    print(f"\\nAggregator Deduplication Verified: Removed {len(covered_chunks)} overlapping chunks from targeted context.")



# ============================================================================
# RUNNER
# ============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
