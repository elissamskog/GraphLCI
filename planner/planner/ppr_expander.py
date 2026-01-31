"""
PPR Expander (v5)
Location: planner/planner/ppr_expander.py
Owner: PLANNER Instance

Personalized PageRank expansion from seed nodes to define query scope.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from schemas.contracts import (
    KnowledgeGraph,
    Scope,
    ScopedNode,
    SeedNode,
    StructuralEdge,
)


def personalized_pagerank(
    graph: KnowledgeGraph,
    seed_node_ids: List[str],
    alpha: float = 0.85,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    degree_discount: bool = True,
    seed_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Run Personalized PageRank from seed nodes.

    Parameters
    ----------
    graph:
        The knowledge graph with adjacency lists.
    seed_node_ids:
        Node IDs to use as teleport targets.
    alpha:
        Damping factor (probability of following an edge).
    max_iterations:
        Maximum PPR iterations.
    tolerance:
        Convergence tolerance.
    degree_discount:
        If True, apply log2 normalization on out-weights to prevent
        high-degree hub nodes from absorbing probability mass.
    seed_weights:
        Optional dict mapping node_id to priority weight multiplier.
        Used for weighted seeding (primary=2.0x, bridge=1.5x, secondary=1.0x).
        If None, all seeds get equal weight.

    Returns
    -------
    Dict[str, float]
        Node ID to PPR score mapping.
    """
    all_nodes = list(graph.nodes.keys())
    if not all_nodes:
        return {}

    # Filter seeds to those actually in the graph
    valid_seeds = [s for s in seed_node_ids if s in graph.nodes]
    if not valid_seeds:
        return {n: 0.0 for n in all_nodes}

    # Build bidirectional adjacency: node_id -> [(neighbour_id, weight)]
    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for node_id, edge_ids in graph.adjacency.items():
        for edge_id in edge_ids:
            edge = graph.edges.get(edge_id)
            if edge is None:
                continue
            w = max(edge.weight, 0.01)
            # Add both directions (undirected treatment)
            if edge.source == node_id:
                neighbour = edge.target
            else:
                neighbour = edge.source
            if neighbour in graph.nodes:
                adj[node_id].append((neighbour, w))

    # Compute out-weights (optionally degree-discounted)
    node_degree: Dict[str, int] = {n: len(adj.get(n, [])) for n in all_nodes}
    out_weight: Dict[str, float] = {}
    for node_id in all_nodes:
        neighbors = adj.get(node_id, [])
        if degree_discount:
            # Discount each neighbor's contribution by its own degree
            total = sum(
                w / math.log2(node_degree.get(nbr, 1) + 2)
                for nbr, w in neighbors
            )
        else:
            total = sum(w for _, w in neighbors)
        out_weight[node_id] = total if total > 0 else 1.0

    # Personalization vector: weighted by node specificity s_i = 1 / |P_i|
    # and priority weight (from subclaim decomposition).
    # Rare nodes (fewer chunks) get higher initial weight.
    personal: Dict[str, float] = defaultdict(float)
    total_specificity = 0.0

    for s in valid_seeds:
        node = graph.nodes.get(s)
        chunk_count = len(node.source_chunks) if node else 1

        # Base specificity (anti-hub bias)
        w_specificity = 1.0 / max(chunk_count, 1)

        # Priority weight (from subclaim decomposition)
        w_priority = seed_weights.get(s, 1.0) if seed_weights else 1.0

        # Combined weight
        w = w_specificity * w_priority
        personal[s] = w
        total_specificity += w

    # Normalize so sum(personal) = 1.0
    if total_specificity > 0:
        for s in valid_seeds:
            personal[s] /= total_specificity
    else:
        # Fallback to uniform if something went wrong
        uniform_w = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            personal[s] = uniform_w

    # Initialize scores to personalization vector
    scores: Dict[str, float] = {n: personal.get(n, 0.0) for n in all_nodes}

    teleport = 1.0 - alpha

    for _iteration in range(max_iterations):
        new_scores: Dict[str, float] = {}

        for node in all_nodes:
            # Teleport component
            rank = teleport * personal.get(node, 0.0)

            # Diffusion component: sum over neighbours that point to this node
            for neighbour, weight in adj.get(node, []):
                if degree_discount:
                     weight /= math.log2(node_degree.get(node, 1) + 2)
                
                rank += alpha * scores[neighbour] * (weight / out_weight[neighbour])

            new_scores[node] = rank

        # Check convergence (L1 norm)
        delta = sum(abs(new_scores[n] - scores[n]) for n in all_nodes)
        scores = new_scores

        if delta < tolerance:
            break

    return scores


def extract_scope(
    graph: KnowledgeGraph,
    ppr_scores: Dict[str, float],
    seed_node_ids: List[str],
    top_k: int = 50,
    min_score: float = 0.01,
    prefer_claims: bool = True,
) -> Scope:
    """Extract a scope subgraph from PPR scores.

    Parameters
    ----------
    graph:
        The full knowledge graph.
    ppr_scores:
        PPR scores for all nodes.
    seed_node_ids:
        Original seed node IDs.
    top_k:
        Maximum nodes in scope.
    min_score:
        Minimum PPR score to include.
    prefer_claims:
        If True, prioritize CLAIM nodes over ENTITY nodes by boosting
        CLAIM scores. This ensures factual assertions are included
        in the scope rather than just entity labels.

    Returns
    -------
    Scope
        The scoped subgraph with nodes, edges, and document IDs.
    """
    # Sort nodes by PPR score descending, filter by min_score

    sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
    candidates: List[Tuple[str, float]] = []
    for node_id, score in sorted_nodes:
        if score < min_score:
            break
        if node_id in graph.nodes:
            candidates.append((node_id, score))
    
    # If prefer_claims, re-rank to prioritize CLAIM nodes
    if prefer_claims:
        def node_priority(item: Tuple[str, float]) -> Tuple[int, float]:
            """Sort key: (type_priority, -ppr_score). Lower priority = better."""
            node_id, score = item
            node = graph.nodes.get(node_id)
            if node is None:
                return (2, -score)  # Unknown nodes last
            # node_type is a string: "ENTITY", "CLAIM", etc.
            if node.node_type == "CLAIM":
                return (0, -score)  # Claims first
            elif node.node_type == "ENTITY":
                return (1, -score)  # Entities second
            else:
                return (2, -score)  # Others last
        
        candidates.sort(key=node_priority)
    
    selected: List[Tuple[str, float]] = candidates[:top_k]



    selected_ids: Set[str] = {node_id for node_id, _ in selected}

    # Build ScopedNodes with ranks
    scoped_nodes: Dict[str, ScopedNode] = {}
    for rank, (node_id, score) in enumerate(selected, start=1):
        scoped_nodes[node_id] = ScopedNode(
            node=graph.nodes[node_id],
            ppr_score=score,
            rank=rank,
        )

    # Collect edges between selected nodes
    scoped_edges: Dict[str, StructuralEdge] = {}
    seen_edge_ids: Set[str] = set()
    for node_id in selected_ids:
        for edge_id in graph.adjacency.get(node_id, []):
            if edge_id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge_id)
            edge = graph.edges.get(edge_id)
            if edge is None:
                continue
            if edge.source in selected_ids and edge.target in selected_ids:
                scoped_edges[edge_id] = edge

    # Collect unique document IDs from selected nodes
    doc_ids: List[str] = list({
        graph.nodes[node_id].source_doc_id
        for node_id in selected_ids
        if graph.nodes[node_id].source_doc_id
    })

    # Compute total PPR mass of selected nodes
    total_ppr_mass = sum(score for _, score in selected)

    return Scope(
        nodes=scoped_nodes,
        edges=scoped_edges,
        document_ids=doc_ids,
        seed_node_ids=seed_node_ids,
        total_ppr_mass=total_ppr_mass,
    )
