"""
Graph Assembler (v5)
Location: librarian/graph_assembler.py
Owner: LIBRARIAN Instance

Assembles the full KnowledgeGraph from nodes and edges.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from schemas.contracts import (
    EmbeddedNode,
    KnowledgeGraph,
    StructuralEdge,
)


def assemble_graph(
    nodes: Dict[str, EmbeddedNode],
    edges: List[StructuralEdge],
) -> KnowledgeGraph:
    """Assemble a KnowledgeGraph from nodes and edges.

    Builds adjacency lists, groups nodes by document and type,
    identifies bridge edges (cross-document), and computes counts.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    edges:
        All structural edges.

    Returns
    -------
    KnowledgeGraph
        Complete knowledge graph with adjacency and metadata.
    """
    # Build edges dict
    edges_dict: Dict[str, StructuralEdge] = {e.edge_id: e for e in edges}

    # Build adjacency lists
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        adjacency[e.source].append(e.edge_id)
        adjacency[e.target].append(e.edge_id)

    # Group nodes by document
    node_by_doc: Dict[str, List[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        node_by_doc[node.source_doc_id].append(node_id)

    # Group nodes by type
    node_by_type: Dict[str, List[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        node_by_type[node.node_type].append(node_id)

    # Identify bridge edges (cross-document)
    bridge_edge_ids: List[str] = []
    for e in edges:
        src = nodes.get(e.source)
        tgt = nodes.get(e.target)
        if src is not None and tgt is not None and src.source_doc_id != tgt.source_doc_id:
            e.is_cross_document = True
            bridge_edge_ids.append(e.edge_id)

    # Counts
    document_ids = set(node.source_doc_id for node in nodes.values())

    return KnowledgeGraph(
        nodes=dict(nodes),
        edges=edges_dict,
        adjacency=dict(adjacency),
        node_by_doc=dict(node_by_doc),
        node_by_type=dict(node_by_type),
        bridge_edges=bridge_edge_ids,
        node_count=len(nodes),
        edge_count=len(edges),
        document_count=len(document_ids),
        bridge_count=len(bridge_edge_ids),
    )
