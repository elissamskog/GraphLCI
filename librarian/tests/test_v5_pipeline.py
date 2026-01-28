"""
Tests for the v5 LIBRARIAN pipeline.
Location: librarian/tests/test_v5_pipeline.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from schemas.contracts import (
    CoreferenceGroup,
    CommunityStructure,
    EmbeddedNode,
    KnowledgeGraph,
    LibrarianInput,
    LibrarianOutput,
    NodeSet,
    StructuralEdge,
    StructuralEdgeType,
)

from librarian.similarity_index import SimilarityIndex
from librarian.edge_builder import extract_explicit_edges, create_synonymy_edges
from librarian.graph_assembler import assemble_graph
from librarian.community_detector import detect_communities
from librarian.community_summarizer import summarize_communities
from librarian.librarian_controller import LibrarianControllerV5


# ============================================================================
# Helpers
# ============================================================================


def make_node(
    node_id: str,
    text: str,
    doc_id: str,
    embedding: List[float],
    node_type: str = "ENTITY",
    chunks: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> EmbeddedNode:
    return EmbeddedNode(
        node_id=node_id,
        node_type=node_type,
        canonical_text=text,
        embedding=embedding,
        source_doc_id=doc_id,
        source_chunks=chunks or [],
        metadata=metadata or {},
    )


def make_node_set(
    nodes: List[EmbeddedNode],
    coreference_groups: Optional[List[CoreferenceGroup]] = None,
) -> NodeSet:
    doc_ids = list({n.source_doc_id for n in nodes})
    return NodeSet(
        nodes=nodes,
        source_doc_ids=doc_ids,
        chunk_count=0,
        entity_count=len(nodes),
        claim_count=0,
        coreference_groups=coreference_groups,
    )


def _unit_vec(dim: int, idx: int) -> List[float]:
    """Create a unit vector with 1.0 at position idx."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


# ============================================================================
# 1. test_similarity_index_add_and_query
# ============================================================================


def test_similarity_index_add_and_query():
    n1 = make_node("a", "alpha", "d1", [1.0, 0.0, 0.0])
    n2 = make_node("b", "beta", "d1", [0.0, 1.0, 0.0])
    n3 = make_node("c", "gamma", "d1", [1.0, 0.0, 0.0])

    idx = SimilarityIndex()
    idx.add_nodes({"a": n1, "b": n2, "c": n3})

    results = idx.query([1.0, 0.0, 0.0], top_k=3)
    assert len(results) >= 1
    # The node with identical embedding should be top
    ids = [r[0] for r in results]
    assert ids[0] in ("a", "c")  # both have the same embedding
    # Top similarity should be ~1.0
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# 2. test_similarity_index_threshold_filtering
# ============================================================================


def test_similarity_index_threshold_filtering():
    n1 = make_node("a", "alpha", "d1", [1.0, 0.0, 0.0])
    n2 = make_node("b", "beta", "d1", [0.0, 1.0, 0.0])

    idx = SimilarityIndex()
    idx.add_nodes({"a": n1, "b": n2})

    results = idx.query([1.0, 0.0, 0.0], top_k=10, threshold=0.99)
    # Only node 'a' should pass the threshold
    ids = [r[0] for r in results]
    assert "a" in ids
    assert "b" not in ids


# ============================================================================
# 3. test_extract_explicit_edges_same_source
# ============================================================================


def test_extract_explicit_edges_same_source():
    nodes = {
        "n1": make_node("n1", "A", "doc1", [1, 0, 0]),
        "n2": make_node("n2", "B", "doc1", [0, 1, 0]),
        "n3": make_node("n3", "C", "doc1", [0, 0, 1]),
    }

    edges = extract_explicit_edges(nodes)
    same_source = [e for e in edges if e.edge_type == StructuralEdgeType.SAME_SOURCE]
    # Star topology: hub connects to 2 spokes
    assert len(same_source) == 2
    for e in same_source:
        assert not e.is_cross_document


# ============================================================================
# 4. test_extract_explicit_edges_same_chunk
# ============================================================================


def test_extract_explicit_edges_same_chunk():
    nodes = {
        "n1": make_node("n1", "A", "doc1", [1, 0, 0], chunks=["chunk_x"]),
        "n2": make_node("n2", "B", "doc1", [0, 1, 0], chunks=["chunk_x"]),
    }

    edges = extract_explicit_edges(nodes)
    same_chunk = [e for e in edges if e.edge_type == StructuralEdgeType.SAME_CHUNK]
    assert len(same_chunk) >= 1
    assert same_chunk[0].metadata.get("chunk_id") == "chunk_x"


# ============================================================================
# 5. test_extract_explicit_edges_cites
# ============================================================================


def test_extract_explicit_edges_cites():
    nodes = {
        "n1": make_node("n1", "Source", "doc1", [1, 0], metadata={"cites": ["n2"]}),
        "n2": make_node("n2", "Target", "doc2", [0, 1]),
    }

    edges = extract_explicit_edges(nodes)
    cites = [e for e in edges if e.edge_type == StructuralEdgeType.CITES]
    assert len(cites) == 1
    assert cites[0].source == "n1"
    assert cites[0].target == "n2"
    assert cites[0].is_cross_document


# ============================================================================
# 6. test_create_synonymy_edges_high_similarity
# ============================================================================


def test_create_synonymy_edges_high_similarity():
    # Nearly identical embeddings
    nodes = {
        "n1": make_node("n1", "Treaty", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "Treaty", "doc2", [0.99, 0.1, 0.0]),
    }

    edges = create_synonymy_edges(nodes, similarity_threshold=0.85)
    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    assert len(synonymy) >= 1
    assert synonymy[0].similarity is not None
    assert synonymy[0].similarity > 0.85


# ============================================================================
# 7. test_create_synonymy_edges_low_similarity
# ============================================================================


def test_create_synonymy_edges_low_similarity():
    # Orthogonal embeddings → cosine sim = 0
    nodes = {
        "n1": make_node("n1", "X", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "Y", "doc2", [0.0, 1.0, 0.0]),
    }

    edges = create_synonymy_edges(nodes, similarity_threshold=0.85)
    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    assert len(synonymy) == 0


# ============================================================================
# 8. test_create_synonymy_edges_coreference
# ============================================================================


def test_create_synonymy_edges_coreference():
    nodes = {
        "n1": make_node("n1", "EU", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "European Union", "doc2", [0.0, 1.0, 0.0]),
    }
    coref = [CoreferenceGroup(canonical_form="EU", variants=["European Union"])]

    edges = create_synonymy_edges(nodes, coreference_groups=coref, similarity_threshold=0.99)
    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    # Should have coreference edge even though embeddings are orthogonal
    coref_edges = [e for e in synonymy if e.similarity == 1.0]
    assert len(coref_edges) == 1
    assert coref_edges[0].metadata.get("origin") == "coreference"


# ============================================================================
# 9. test_assemble_graph_structure
# ============================================================================


def test_assemble_graph_structure():
    nodes = {
        "n1": make_node("n1", "A", "doc1", [1, 0]),
        "n2": make_node("n2", "B", "doc1", [0, 1]),
        "n3": make_node("n3", "C", "doc2", [1, 1]),
        "n4": make_node("n4", "D", "doc2", [0, 0]),
    }
    edges = [
        StructuralEdge(
            edge_id="e1", source="n1", target="n2",
            edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0,
            metadata={},
        ),
        StructuralEdge(
            edge_id="e2", source="n3", target="n4",
            edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0,
            metadata={},
        ),
        StructuralEdge(
            edge_id="e3", source="n1", target="n3",
            edge_type=StructuralEdgeType.SYNONYMY, weight=0.9,
            metadata={}, similarity=0.9,
        ),
    ]

    graph = assemble_graph(nodes, edges)

    assert graph.node_count == 4
    assert graph.edge_count == 3
    assert graph.document_count == 2
    # n1-n3 is cross-document
    assert "e3" in graph.bridge_edges
    assert graph.bridge_count >= 1
    # Adjacency: n1 should have e1 and e3
    assert "e1" in graph.adjacency["n1"]
    assert "e3" in graph.adjacency["n1"]
    # node_by_doc
    assert set(graph.node_by_doc["doc1"]) == {"n1", "n2"}
    assert set(graph.node_by_doc["doc2"]) == {"n3", "n4"}


# ============================================================================
# 10. test_detect_communities_basic
# ============================================================================


def test_detect_communities_basic():
    # Two dense clusters with no inter-cluster edges
    nodes = {}
    for i in range(6):
        nodes[f"n{i}"] = make_node(f"n{i}", f"Node{i}", "doc1", _unit_vec(6, i))

    # Cluster A: n0-n1-n2 (fully connected)
    cluster_a_edges = [
        StructuralEdge(edge_id="ea01", source="n0", target="n1",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
        StructuralEdge(edge_id="ea02", source="n0", target="n2",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
        StructuralEdge(edge_id="ea12", source="n1", target="n2",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
    ]
    # Cluster B: n3-n4-n5 (fully connected)
    cluster_b_edges = [
        StructuralEdge(edge_id="eb34", source="n3", target="n4",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
        StructuralEdge(edge_id="eb35", source="n3", target="n5",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
        StructuralEdge(edge_id="eb45", source="n4", target="n5",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
    ]

    all_edges = cluster_a_edges + cluster_b_edges
    graph = assemble_graph(nodes, all_edges)
    cs = detect_communities(graph, max_levels=3)

    # Should have level 0
    assert 0 in cs.levels
    level_0_comms = cs.levels[0]

    # Each original node should be assigned
    for nid in nodes:
        assert nid in cs.node_to_community

    # Should detect 2 communities at level 0
    assert len(level_0_comms) == 2

    # Verify cluster membership: nodes 0-2 in one community, 3-5 in another
    comm_a = cs.node_to_community["n0"]
    comm_b = cs.node_to_community["n3"]
    assert comm_a != comm_b
    assert cs.node_to_community["n1"] == comm_a
    assert cs.node_to_community["n2"] == comm_a
    assert cs.node_to_community["n4"] == comm_b
    assert cs.node_to_community["n5"] == comm_b


# ============================================================================
# 11. test_summarize_communities_mock_llm
# ============================================================================


def test_summarize_communities_mock_llm():
    nodes = {}
    for i in range(4):
        nodes[f"n{i}"] = make_node(f"n{i}", f"Concept {i}", "doc1", _unit_vec(4, i))

    edges = [
        StructuralEdge(edge_id="e01", source="n0", target="n1",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
        StructuralEdge(edge_id="e23", source="n2", target="n3",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={}),
    ]

    graph = assemble_graph(nodes, edges)
    cs = detect_communities(graph, max_levels=1)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"summary": "Test summary", "key_themes": ["theme1", "theme2"]}'

    summaries = summarize_communities(graph, cs, mock_llm)

    assert isinstance(summaries, dict)
    assert len(summaries) > 0
    for comm_id, summary in summaries.items():
        assert isinstance(summary, str)
        assert len(summary) > 0
        comm = cs.communities[comm_id]
        assert comm.summary is not None


# ============================================================================
# 12. test_full_pipeline
# ============================================================================


def test_full_pipeline():
    # 6 nodes across 2 documents, with some high-similarity pairs
    nodes_list = [
        make_node("n0", "Contract law", "doc1", [0.9, 0.1, 0.0, 0.0]),
        make_node("n1", "Tort law", "doc1", [0.1, 0.9, 0.0, 0.0]),
        make_node("n2", "Contract law principles", "doc1", [0.88, 0.12, 0.0, 0.0]),
        make_node("n3", "Negligence", "doc2", [0.0, 0.8, 0.2, 0.0]),
        make_node("n4", "Strict liability", "doc2", [0.0, 0.1, 0.9, 0.0]),
        make_node("n5", "Product liability", "doc2", [0.0, 0.15, 0.85, 0.0]),
    ]

    node_set = make_node_set(nodes_list)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"summary": "Legal concepts", "key_themes": ["law"]}'

    controller = LibrarianControllerV5(
        llm_client=mock_llm,
        config={
            "synonymy": {"threshold": 0.80},
            "community_detection": {"enabled": True, "max_levels": 2},
            "summarization": {"enabled": True},
        },
    )

    inp = LibrarianInput(node_set=node_set)
    output = controller.process(inp)

    assert isinstance(output, LibrarianOutput)
    assert output.graph is not None
    assert output.graph.node_count == 6
    assert output.graph.edge_count > 0
    assert output.community_structure is not None
    assert len(output.community_structure.communities) > 0


# ============================================================================
# 13. test_empty_node_set
# ============================================================================


def test_empty_node_set():
    node_set = make_node_set([])

    mock_llm = MagicMock()

    controller = LibrarianControllerV5(llm_client=mock_llm)
    inp = LibrarianInput(node_set=node_set)
    output = controller.process(inp)

    assert isinstance(output, LibrarianOutput)
    assert output.graph.node_count == 0
    assert output.graph.edge_count == 0
    assert output.community_structure is not None
    assert len(output.community_structure.communities) == 0


# ============================================================================
# 14. test_create_synonymy_edges_default_threshold
# ============================================================================


def test_create_synonymy_edges_default_threshold():
    """Verify the default threshold is 0.80 (spec §5.1)."""
    # Cosine sim ≈ 0.82 — should pass 0.80 but not 0.85
    nodes = {
        "n1": make_node("n1", "A", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "B", "doc1", [0.82, 0.57, 0.0]),
    }

    # Use default threshold (should be 0.80)
    edges = create_synonymy_edges(nodes)
    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    assert len(synonymy) >= 1, "Default threshold 0.80 should accept similarity ~0.82"


# ============================================================================
# 15. test_extract_explicit_edges_chunk_node_map
# ============================================================================


def test_extract_explicit_edges_chunk_node_map():
    """When chunk_node_map is provided, it should be used for SAME_CHUNK edges
    instead of node.source_chunks."""
    nodes = {
        "n1": make_node("n1", "A", "doc1", [1, 0, 0], chunks=[]),  # no source_chunks
        "n2": make_node("n2", "B", "doc1", [0, 1, 0], chunks=[]),
        "n3": make_node("n3", "C", "doc1", [0, 0, 1], chunks=["unrelated_chunk"]),
    }

    # Provide an external chunk_node_map grouping n1 and n2 together
    chunk_map = {"chunk_alpha": ["n1", "n2"]}

    edges = extract_explicit_edges(nodes, chunk_node_map=chunk_map)
    same_chunk = [e for e in edges if e.edge_type == StructuralEdgeType.SAME_CHUNK]

    # Should have a SAME_CHUNK edge from the external map
    assert len(same_chunk) == 1
    edge = same_chunk[0]
    assert {edge.source, edge.target} == {"n1", "n2"}
    assert edge.metadata.get("chunk_id") == "chunk_alpha"


# ============================================================================
# 16. test_create_synonymy_edges_via_index
# ============================================================================


def test_create_synonymy_edges_via_index():
    """When a SimilarityIndex is passed, it should be used for neighbor lookup."""
    nodes = {
        "n1": make_node("n1", "Treaty", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "Treaty v2", "doc2", [0.99, 0.1, 0.0]),
        "n3": make_node("n3", "Unrelated", "doc2", [0.0, 1.0, 0.0]),
    }

    idx = SimilarityIndex()
    idx.add_nodes(nodes)

    edges = create_synonymy_edges(
        nodes,
        similarity_threshold=0.80,
        similarity_index=idx,
    )

    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    # n1 and n2 are very similar; n3 is orthogonal to n1
    pairs = {(e.source, e.target) for e in synonymy}
    # n1-n2 should be connected
    assert ("n1", "n2") in pairs or ("n2", "n1") in pairs
    # n1-n3 should NOT be connected (cosine sim ≈ 0)
    assert ("n1", "n3") not in pairs and ("n3", "n1") not in pairs


# ============================================================================
# 17. test_detect_communities_min_size
# ============================================================================


def test_detect_communities_min_size():
    """Communities smaller than min_community_size should be merged."""
    nodes = {}
    # Cluster A: 4 nodes fully connected
    for i in range(4):
        nodes[f"a{i}"] = make_node(f"a{i}", f"A{i}", "doc1", _unit_vec(8, i))
    # Cluster B: 1 node weakly connected to cluster A
    nodes["b0"] = make_node("b0", "B0", "doc2", _unit_vec(8, 4))

    edges = []
    # Fully connect cluster A
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(StructuralEdge(
                edge_id=f"ea{i}{j}", source=f"a{i}", target=f"a{j}",
                edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={},
            ))
    # Weak connection from b0 to a0
    edges.append(StructuralEdge(
        edge_id="eb0a0", source="b0", target="a0",
        edge_type=StructuralEdgeType.REFERENCES, weight=0.1, metadata={},
    ))

    graph = assemble_graph(nodes, edges)

    # With min_community_size=2, the singleton b0 community should be merged
    cs = detect_communities(graph, max_levels=1, min_community_size=2)

    # All nodes should end up in the same community since b0's singleton
    # gets merged into cluster A
    level_0 = cs.levels[0]
    assert len(level_0) == 1, f"Expected 1 community after merge, got {len(level_0)}"


# ============================================================================
# 18. test_summarize_bottom_up
# ============================================================================


def test_summarize_bottom_up():
    """Level-1 summary should use level-0 summaries as input, not raw nodes."""
    from schemas.contracts import Community

    nodes = {}
    for i in range(6):
        nodes[f"n{i}"] = make_node(f"n{i}", f"Concept {i}", "doc1", _unit_vec(6, i))

    # Cluster A: n0-n1-n2, Cluster B: n3-n4-n5
    cluster_a_edges = [
        StructuralEdge(edge_id=f"ea{i}{j}", source=f"n{i}", target=f"n{j}",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={})
        for i in range(3) for j in range(i + 1, 3)
    ]
    cluster_b_edges = [
        StructuralEdge(edge_id=f"eb{i}{j}", source=f"n{i}", target=f"n{j}",
                       edge_type=StructuralEdgeType.SAME_SOURCE, weight=1.0, metadata={})
        for i in range(3, 6) for j in range(i + 1, 6)
    ]
    all_edges = cluster_a_edges + cluster_b_edges

    graph = assemble_graph(nodes, all_edges)
    cs = detect_communities(graph, max_levels=2)

    # Track prompts sent to LLM
    prompts: List[str] = []

    def capture_generate(prompt: str) -> str:
        prompts.append(prompt)
        return '{"summary": "Test summary", "key_themes": ["t1"], "tensions": null}'

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = capture_generate

    summarize_communities(graph, cs, mock_llm)

    # If there are level-1 communities, their prompts should reference
    # "sub-community summaries" rather than raw node content
    if 1 in cs.levels:
        level_1_ids = set(cs.levels[1])
        # Find prompts for level-1 communities: they should mention sub-community
        # We check that at least one prompt contains "sub-community" or "Sub-community"
        has_bottom_up = any("sub-community" in p.lower() for p in prompts)
        assert has_bottom_up, "Level-1 summarization should use child summaries (bottom-up)"


# ============================================================================
# 19. test_cross_doc_diversity_reserved_slots
# ============================================================================


def test_cross_doc_diversity_reserved_slots():
    """Cross-doc edges get reserved slots even when intra-doc similarity is higher."""
    # doc1 has many highly similar nodes; doc2 has one node moderately similar
    nodes = {
        "d1_a": make_node("d1_a", "A", "doc1", [1.0, 0.0, 0.0, 0.0]),
        "d1_b": make_node("d1_b", "B", "doc1", [0.99, 0.1, 0.0, 0.0]),
        "d1_c": make_node("d1_c", "C", "doc1", [0.98, 0.15, 0.0, 0.0]),
        "d1_d": make_node("d1_d", "D", "doc1", [0.97, 0.2, 0.0, 0.0]),
        "d2_x": make_node("d2_x", "X", "doc2", [0.85, 0.3, 0.0, 0.0]),
    }

    # Use brute-force path (no similarity_index) with reserved cross-doc slots
    edges = create_synonymy_edges(
        nodes,
        similarity_threshold=0.80,
        cross_doc_threshold=0.70,
        max_edges_per_node=3,
        min_cross_doc_edges=1,
    )

    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    cross_doc = [e for e in synonymy if e.is_cross_document]

    # d2_x should have at least one cross-doc edge due to reserved slot
    assert len(cross_doc) >= 1, (
        f"Expected at least 1 cross-doc edge, got {len(cross_doc)}. "
        f"All edges: {[(e.source, e.target, e.similarity, e.is_cross_document) for e in synonymy]}"
    )


# ============================================================================
# 20. test_cross_doc_threshold_separate
# ============================================================================


def test_cross_doc_threshold_separate():
    """Cross-doc pairs below 0.80 but above 0.70 are connected when
    cross_doc_threshold is set."""
    # Build vectors where cross-doc sim lands in 0.70-0.80 range
    nodes = {
        "d1_a": make_node("d1_a", "A", "doc1", [1.0, 0.0, 0.0]),
        "d2_b": make_node("d2_b", "B", "doc2", [0.75, 0.66, 0.0]),
    }

    # Without cross_doc_threshold: 0.80 threshold should reject this pair
    edges_strict = create_synonymy_edges(
        nodes,
        similarity_threshold=0.80,
        min_cross_doc_edges=0,
    )
    synonymy_strict = [e for e in edges_strict if e.edge_type == StructuralEdgeType.SYNONYMY]
    assert len(synonymy_strict) == 0, "Strict threshold 0.80 should reject sim ~0.75"

    # With cross_doc_threshold=0.70 and reserved slots: should accept
    edges_relaxed = create_synonymy_edges(
        nodes,
        similarity_threshold=0.80,
        cross_doc_threshold=0.70,
        min_cross_doc_edges=1,
        max_edges_per_node=5,
    )
    synonymy_relaxed = [e for e in edges_relaxed if e.edge_type == StructuralEdgeType.SYNONYMY]
    assert len(synonymy_relaxed) >= 1, (
        f"Relaxed cross_doc_threshold 0.70 should accept sim ~0.75. "
        f"Got {len(synonymy_relaxed)} edges."
    )


# ============================================================================
# 21. test_coreference_case_insensitive
# ============================================================================


def test_coreference_case_insensitive():
    """Case differences should not block coreference matching."""
    nodes = {
        "n1": make_node("n1", "Contract Law", "doc1", [1.0, 0.0, 0.0]),
        "n2": make_node("n2", "contract law", "doc2", [0.0, 1.0, 0.0]),
        "n3": make_node("n3", "CONTRACT LAW", "doc3", [0.0, 0.0, 1.0]),
    }
    coref = [CoreferenceGroup(canonical_form="Contract Law", variants=["contract law"])]

    edges = create_synonymy_edges(
        nodes,
        coreference_groups=coref,
        similarity_threshold=0.99,  # High threshold so only coref edges form
    )

    coref_edges = [
        e for e in edges
        if e.edge_type == StructuralEdgeType.SYNONYMY
        and e.metadata.get("origin") == "coreference"
    ]

    # n1 matches "Contract Law", n2 matches "contract law", n3 matches "CONTRACT LAW"
    # (case-insensitive). All three should be linked pairwise.
    connected_nodes = set()
    for e in coref_edges:
        connected_nodes.add(e.source)
        connected_nodes.add(e.target)

    assert "n1" in connected_nodes, "n1 ('Contract Law') should match canonical"
    assert "n2" in connected_nodes, "n2 ('contract law') should match variant"
    assert "n3" in connected_nodes, "n3 ('CONTRACT LAW') should match case-insensitively"
    # 3 nodes → 3 pairwise edges
    assert len(coref_edges) == 3, f"Expected 3 pairwise coref edges, got {len(coref_edges)}"


# ============================================================================
# 22. test_max_edges_per_node_configurable
# ============================================================================


def test_max_edges_per_node_configurable():
    """Controller reads max_edges_per_node from config."""
    nodes_list = [
        make_node("n0", "A", "doc1", [1.0, 0.0, 0.0, 0.0]),
        make_node("n1", "B", "doc1", [0.99, 0.1, 0.0, 0.0]),
        make_node("n2", "C", "doc1", [0.98, 0.15, 0.0, 0.0]),
        make_node("n3", "D", "doc1", [0.97, 0.2, 0.0, 0.0]),
    ]

    node_set = make_node_set(nodes_list)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"summary": "Test", "key_themes": ["t"]}'

    # Set max_edges_per_node=1 so each node gets at most 1 synonymy edge
    controller = LibrarianControllerV5(
        llm_client=mock_llm,
        config={
            "synonymy": {
                "threshold": 0.80,
                "max_edges_per_node": 1,
                "min_cross_doc_edges": 0,
            },
            "community_detection": {"enabled": False},
        },
    )

    inp = LibrarianInput(node_set=node_set)
    output = controller.process(inp)

    # With max_edges_per_node=1, each node can have at most 1 synonymy edge
    synonymy = [
        e for e in output.graph.edges.values()
        if e.edge_type == StructuralEdgeType.SYNONYMY
    ]
    # Count per node
    from collections import Counter
    node_counts: Counter = Counter()
    for e in synonymy:
        node_counts[e.source] += 1
        node_counts[e.target] += 1

    for nid, count in node_counts.items():
        assert count <= 1, f"Node {nid} has {count} synonymy edges, expected <= 1"


# ============================================================================
# 23. test_surplus_rollover
# ============================================================================


def test_surplus_rollover():
    """When no cross-doc neighbors exist, reserved slots roll over to intra-doc."""
    # All nodes in the same document — no cross-doc candidates exist
    nodes = {
        "n0": make_node("n0", "A", "doc1", [1.0, 0.0, 0.0]),
        "n1": make_node("n1", "B", "doc1", [0.99, 0.1, 0.0]),
        "n2": make_node("n2", "C", "doc1", [0.98, 0.15, 0.0]),
        "n3": make_node("n3", "D", "doc1", [0.97, 0.2, 0.0]),
    }

    # Request 3 reserved cross-doc slots, but there are no cross-doc nodes
    edges = create_synonymy_edges(
        nodes,
        similarity_threshold=0.80,
        cross_doc_threshold=0.70,
        max_edges_per_node=4,
        min_cross_doc_edges=3,
    )

    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]

    # Despite reserving 3 cross-doc slots, intra-doc edges should still form
    # because surplus rolls over. Without rollover, each node would only get
    # max_edges(4) - reserved(3) = 1 intra-doc slot.
    # With rollover, unused cross-doc slots become available for intra-doc.
    assert len(synonymy) >= 2, (
        f"Surplus rollover should allow more intra-doc edges. Got {len(synonymy)} edges."
    )
