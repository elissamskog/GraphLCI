"""
Seed Finder (v5)
Location: planner/planner/seed_finder.py
Owner: PLANNER Instance

Finds seed nodes in the graph by embedding similarity to the query.
"""

from __future__ import annotations

from typing import Any, List

from schemas.contracts import KnowledgeGraph, SeedNode


def find_seed_nodes(
    query_embedding: List[float],
    similarity_index: Any,  # SimilarityIndex instance
    graph: KnowledgeGraph,
    top_k: int = 10,
    threshold: float = 0.5,
) -> List[SeedNode]:
    """Find seed nodes most similar to the query embedding.

    Uses similarity_index.query() to get (node_id, score) pairs,
    then enriches each result with metadata from graph.nodes.

    Parameters
    ----------
    query_embedding:
        The query's embedding vector.
    similarity_index:
        Similarity index built by LIBRARIAN.
    graph:
        The knowledge graph, used to look up node metadata
        (node_type, canonical_text, source_doc_id) for each hit.
    top_k:
        Maximum number of seed nodes.
    threshold:
        Minimum similarity threshold.

    Returns
    -------
    List[SeedNode]
        Seed nodes ranked by similarity, with full metadata.
    """
    # Over-fetch to leave room for threshold filtering
    raw_results = similarity_index.query(query_embedding, top_k * 2)

    seeds: List[SeedNode] = []
    for node_id, score in raw_results:
        if score < threshold:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        seeds.append(SeedNode(
            node_id=node_id,
            similarity=score,
            node_type=node.node_type,
            canonical_text=node.canonical_text,
            source_doc_id=node.source_doc_id,
        ))

    # Sort descending by similarity, cap at top_k
    seeds.sort(key=lambda s: s.similarity, reverse=True)
    return seeds[:top_k]


def find_seeds_multi_query(
    query_embeddings: List[List[float]],
    similarity_index: Any,
    graph: KnowledgeGraph,
    top_k: int = 10,
    threshold: float = 0.5,
) -> List[SeedNode]:
    """Find seed nodes using multiple query embeddings, merging results.

    Calls find_seed_nodes for each embedding, then merges by keeping
    the highest similarity score per node_id.

    Parameters
    ----------
    query_embeddings:
        List of embedding vectors (e.g. per-entity + raw query).
    similarity_index:
        Similarity index built by LIBRARIAN.
    graph:
        The knowledge graph.
    top_k:
        Maximum number of seed nodes in final result.
    threshold:
        Minimum similarity threshold per individual query.

    Returns
    -------
    List[SeedNode]
        Merged seed nodes ranked by similarity, capped at top_k.
    """
    best_by_node: dict[str, SeedNode] = {}

    for embedding in query_embeddings:
        seeds = find_seed_nodes(
            embedding, similarity_index, graph,
            top_k=top_k, threshold=threshold,
        )
        for seed in seeds:
            existing = best_by_node.get(seed.node_id)
            if existing is None or seed.similarity > existing.similarity:
                best_by_node[seed.node_id] = seed

    merged = list(best_by_node.values())
    merged.sort(key=lambda s: s.similarity, reverse=True)
    return merged[:top_k]
