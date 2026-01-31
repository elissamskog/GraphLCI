"""
Seed Finder (v5)
Location: planner/planner/seed_finder.py
Owner: PLANNER Instance

Finds seed nodes in the graph by embedding similarity to the query.
"""

from __future__ import annotations

from typing import Any, List, Dict, Optional

from schemas.contracts import KnowledgeGraph, SeedNode, Subclaim, WeightedSeedSet


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


def _are_similar(text1: str, text2: str) -> bool:
    """Simple heuristic for duplicate detection based on word overlap."""
    s1 = set(text1.lower().split())
    s2 = set(text2.lower().split())
    if not s1 or not s2:
        return False
    intersection = s1.intersection(s2)
    overlap = len(intersection) / max(len(s1), len(s2))
    return overlap > 0.70


def find_subclaim_seeds(
    subclaims: List[Subclaim],
    similarity_index: Any,
    graph: KnowledgeGraph,
    embedding_client: Any,
    top_k_per_subclaim: int = 5,
    threshold: float = 0.70,
) -> Dict[str, List[SeedNode]]:
    """Find seeds for each subclaim objective.

    For each subclaim:
    1. Embed the objective text
    2. Embed each entity separately
    3. Find seeds, merge by max similarity
    4. Return bucketed by subclaim_id

    Parameters
    ----------
    subclaims:
        List of Subclaim objects from decomposition
    similarity_index:
        Similarity index built by LIBRARIAN
    graph:
        Knowledge graph for node metadata
    embedding_client:
        Client for embedding text (supports embed() method)
    top_k_per_subclaim:
        Maximum seeds per subclaim
    threshold:
        Minimum similarity threshold

    Returns
    -------
    Dict[str, List[SeedNode]]
        Mapping from subclaim_id to list of seed nodes
    """
    subclaim_seeds: Dict[str, List[SeedNode]] = {}

    for sc in subclaims:
        # Embed the objective text
        objective_embedding = embedding_client.embed(sc.objective)

        # Embed each entity separately
        entity_embeddings = []
        for entity in sc.entities:
            entity_embeddings.append(embedding_client.embed(entity))

        # Combine objective + entity embeddings
        all_embeddings = [objective_embedding] + entity_embeddings

        # Find seeds using multi-query search
        seeds = find_seeds_multi_query(
            query_embeddings=all_embeddings,
            similarity_index=similarity_index,
            graph=graph,
            top_k=top_k_per_subclaim,
            threshold=threshold,
        )

        subclaim_seeds[sc.subclaim_id] = seeds

    return subclaim_seeds


def find_balanced_seeds(
    global_embedding: List[float],
    entity_embeddings: List[List[float]],
    similarity_index: Any,
    graph: KnowledgeGraph,
    top_k: int = 12,
    threshold: float = 0.5,
    global_quota: int = 3,
    entity_quota: int = 3,
    entity_texts: List[str] = None,
    # NEW for weighted seeding:
    subclaim_seeds: Optional[Dict[str, List[SeedNode]]] = None,
    subclaim_priorities: Optional[Dict[str, str]] = None,
) -> WeightedSeedSet:
    """Find seed nodes with diversity and deduplication.

    Priority (when subclaim_seeds is provided):
    0. Hard Seeding (Exact or fuzzy string matches for query entities)
    1. Primary Subclaim Seeds (priority="primary", quota=4 per subclaim)
    2. Bridge Subclaim Seeds (priority="bridge", quota=3 per subclaim)
    3. Entity Buckets (quota=2)
    4. Global Bucket (quota=2)
    5. Fill remaining by score

    Priority (original, when subclaim_seeds is None):
    0. Hard Seeding
    1. Entity Buckets
    2. Global Bucket
    3. Fill remaining by score

    Returns:
        WeightedSeedSet with seeds and weights for weighted PPR
    """
    all_buckets: List[List[SeedNode]] = []

    # Search Global Bucket
    all_buckets.append(find_seed_nodes(
        global_embedding, similarity_index, graph, top_k=top_k, threshold=threshold
    ))

    # Search Entity Buckets
    for emb in entity_embeddings:
        all_buckets.append(find_seed_nodes(
            emb, similarity_index, graph, top_k=top_k, threshold=threshold
        ))

    final_seeds: dict[str, SeedNode] = {}
    primary_node_ids: List[str] = []
    bridge_node_ids: List[str] = []

    def is_duplicate(new_seed: SeedNode) -> bool:
        for existing in final_seeds.values():
            if _are_similar(new_seed.canonical_text, existing.canonical_text):
                return True
        return False

    # Round 0: Hard Seeding (unchanged)
    if entity_texts:
        for bucket in all_buckets:
            for s in bucket:
                if s.node_id not in final_seeds:
                    if any(t.lower() in s.canonical_text.lower() for t in entity_texts):
                        final_seeds[s.node_id] = s

    # NEW: Multi-source weighted seeding path
    if subclaim_seeds is not None and subclaim_priorities is not None:
        # Round 1: Primary Subclaim Seeds
        for subclaim_id, seeds in subclaim_seeds.items():
            priority = subclaim_priorities.get(subclaim_id, 'secondary')
            if priority == 'primary':
                added = 0
                for s in seeds:
                    if added >= 4:  # Primary quota
                        break
                    if s.node_id not in final_seeds:
                        if not is_duplicate(s):
                            final_seeds[s.node_id] = s
                            primary_node_ids.append(s.node_id)
                            added += 1

        # Round 2: Bridge Subclaim Seeds
        for subclaim_id, seeds in subclaim_seeds.items():
            priority = subclaim_priorities.get(subclaim_id, 'secondary')
            if priority == 'bridge':
                added = 0
                for s in seeds:
                    if added >= 3:  # Bridge quota
                        break
                    if s.node_id not in final_seeds:
                        if not is_duplicate(s):
                            final_seeds[s.node_id] = s
                            bridge_node_ids.append(s.node_id)
                            added += 1

        # Round 3: Entity Buckets (reduced quota)
        for bucket in all_buckets[1:]:
            added = 0
            for s in bucket:
                if added >= 2:  # Reduced from entity_quota
                    break
                if s.node_id not in final_seeds:
                    if not is_duplicate(s):
                        final_seeds[s.node_id] = s
                        added += 1

        # Round 4: Global Bucket (reduced quota)
        global_added = 0
        for s in all_buckets[0]:
            if global_added >= 2:  # Reduced from global_quota
                break
            if s.node_id not in final_seeds:
                if not is_duplicate(s):
                    final_seeds[s.node_id] = s
                    global_added += 1

    else:
        # Original path: No subclaim seeds
        # Round 1: Entity Buckets
        for bucket in all_buckets[1:]:
            added = 0
            for s in bucket:
                if added >= entity_quota:
                    break
                if s.node_id not in final_seeds:
                    if not is_duplicate(s):
                        final_seeds[s.node_id] = s
                        added += 1
                else:
                    added += 1

        # Round 2: Global Bucket
        global_added = 0
        for s in all_buckets[0]:
            if global_added >= global_quota:
                break
            if s.node_id not in final_seeds:
                if not is_duplicate(s):
                    final_seeds[s.node_id] = s
                    global_added += 1
            else:
                global_added += 1

    # Round 5 (or 3): Fill remaining up to top_k by similarity
    remaining_candidates = []
    for bucket in all_buckets:
        for s in bucket:
            if s.node_id not in final_seeds:
                remaining_candidates.append(s)

    remaining_candidates.sort(key=lambda x: x.similarity, reverse=True)
    for s in remaining_candidates:
        if len(final_seeds) >= top_k:
            break
        if s.node_id not in final_seeds and not is_duplicate(s):
            final_seeds[s.node_id] = s

    # Final sort
    merged = list(final_seeds.values())
    merged.sort(key=lambda s: s.similarity, reverse=True)
    final_list = merged[:top_k]

    # Compute weights (uniform by default)
    weights: Dict[str, float] = {seed.node_id: 1.0 for seed in final_list}

    return WeightedSeedSet(
        seeds=final_list,
        weights=weights,
        primary_node_ids=primary_node_ids,
        bridge_node_ids=bridge_node_ids,
    )
