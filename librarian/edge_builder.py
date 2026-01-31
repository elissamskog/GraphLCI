"""
Edge Builder (v5)
Location: librarian/edge_builder.py
Owner: LIBRARIAN Instance

Creates structural edges from node embeddings and explicit relationships.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from schemas.contracts import (
    EmbeddedNode,
    StructuralEdge,
    StructuralEdgeType,
    CoreferenceGroup,
)

if TYPE_CHECKING:
    from librarian.similarity_index import SimilarityIndex


def _make_edge_id() -> str:
    return uuid.uuid4().hex[:12]


def _is_cross_document(
    source_id: str, target_id: str, nodes: Dict[str, EmbeddedNode]
) -> bool:
    src = nodes.get(source_id)
    tgt = nodes.get(target_id)
    if src is None or tgt is None:
        return False
    return src.source_doc_id != tgt.source_doc_id


def _normalize_for_coref(text: str) -> str:
    """Normalize text for case-insensitive coreference matching."""
    return text.strip().lower()


def extract_explicit_edges(
    nodes: Dict[str, EmbeddedNode],
    chunk_node_map: Optional[Dict[str, List[str]]] = None,
) -> List[StructuralEdge]:
    """Extract explicit edges (CITES, REFERENCES, DEFINED_BY, etc.)
    from node metadata and text.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    chunk_node_map:
        Optional mapping of chunk_id → list of node_ids from EXPLORER.
        When provided, used for SAME_CHUNK edge grouping instead of
        ``node.source_chunks``.

    Returns
    -------
    List[StructuralEdge]
        Explicit structural edges found in the corpus.
    """
    edges: List[StructuralEdge] = []

    # --- Metadata-based edges ---
    for node_id, node in nodes.items():
        meta = node.metadata or {}

        # CITES
        cites = meta.get("cites")
        if isinstance(cites, list):
            for target_id in cites:
                if target_id in nodes:
                    edges.append(StructuralEdge(
                        edge_id=_make_edge_id(),
                        source=node_id,
                        target=target_id,
                        edge_type=StructuralEdgeType.CITES,
                        weight=1.0,
                        metadata={"origin": "metadata"},
                        similarity=None,
                        is_cross_document=_is_cross_document(node_id, target_id, nodes),
                    ))

        # REFERENCES
        references = meta.get("references")
        if isinstance(references, list):
            for target_id in references:
                if target_id in nodes:
                    edges.append(StructuralEdge(
                        edge_id=_make_edge_id(),
                        source=node_id,
                        target=target_id,
                        edge_type=StructuralEdgeType.REFERENCES,
                        weight=1.0,
                        metadata={"origin": "metadata"},
                        similarity=None,
                        is_cross_document=_is_cross_document(node_id, target_id, nodes),
                    ))

        # DEFINED_BY
        defined_by = meta.get("defined_by")
        if isinstance(defined_by, str) and defined_by in nodes:
            edges.append(StructuralEdge(
                edge_id=_make_edge_id(),
                source=node_id,
                target=defined_by,
                edge_type=StructuralEdgeType.DEFINED_BY,
                weight=1.0,
                metadata={"origin": "metadata"},
                similarity=None,
                is_cross_document=_is_cross_document(node_id, defined_by, nodes),
            ))

    # --- SAME_SOURCE edges (star topology per document) ---
    doc_groups: Dict[str, List[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        doc_groups[node.source_doc_id].append(node_id)

    for doc_id, group in doc_groups.items():
        if len(group) < 2:
            continue
        hub = group[0]
        # Log-normalize weight to penalize high-degree hubs
        hub_weight = 1.0 / math.log2(len(group) + 1)
        for spoke in group[1:]:
            edges.append(StructuralEdge(
                edge_id=_make_edge_id(),
                source=hub,
                target=spoke,
                edge_type=StructuralEdgeType.SAME_SOURCE,
                weight=hub_weight,
                metadata={"origin": "same_source", "doc_id": doc_id},
                similarity=None,
                is_cross_document=False,
            ))

    # --- SAME_CHUNK edges (star topology per chunk) ---
    if chunk_node_map is not None:
        # Use EXPLORER-provided chunk→node mapping
        chunk_groups = {
            cid: [nid for nid in nids if nid in nodes]
            for cid, nids in chunk_node_map.items()
        }
    else:
        # Fall back to node.source_chunks
        chunk_groups_dd: Dict[str, List[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            for chunk_id in (node.source_chunks or []):
                chunk_groups_dd[chunk_id].append(node_id)
        chunk_groups = dict(chunk_groups_dd)

    for chunk_id, group in chunk_groups.items():
        if len(group) < 2:
            continue
        hub = group[0]
        # Log-normalize weight to penalize high-degree hubs
        chunk_hub_weight = 1.0 / math.log2(len(group) + 1)
        for spoke in group[1:]:
            if hub == spoke:
                continue
            edges.append(StructuralEdge(
                edge_id=_make_edge_id(),
                source=hub,
                target=spoke,
                edge_type=StructuralEdgeType.SAME_CHUNK,
                weight=chunk_hub_weight,
                metadata={"origin": "same_chunk", "chunk_id": chunk_id},
                similarity=None,
                is_cross_document=_is_cross_document(hub, spoke, nodes),
            ))
            # Bidirectional: Spoke -> Hub
            edges.append(StructuralEdge(
                edge_id=_make_edge_id(),
                source=spoke,
                target=hub,
                edge_type=StructuralEdgeType.SAME_CHUNK,
                weight=chunk_hub_weight,
                metadata={"origin": "same_chunk", "chunk_id": chunk_id},
                similarity=None,
                is_cross_document=_is_cross_document(spoke, hub, nodes),
            ))

    return edges


def create_synonymy_edges(
    nodes: Dict[str, EmbeddedNode],
    coreference_groups: Optional[List[CoreferenceGroup]] = None,
    similarity_threshold: float = 0.80,
    cross_doc_threshold: Optional[float] = None,
    similarity_index: Optional[SimilarityIndex] = None,
    max_edges_per_node: int = 10,
    min_cross_doc_edges: int = 0,
) -> List[StructuralEdge]:
    """Create synonymy edges between nodes with high embedding similarity
    or shared coreference groups.

    Uses a two-tier allocation when ``min_cross_doc_edges > 0``:

    * **Tier 1 (intra-doc):** Fill up to ``max_edges_per_node - min_cross_doc_edges``
      slots using ``similarity_threshold``.
    * **Tier 2 (cross-doc):** Fill reserved ``min_cross_doc_edges`` slots using
      ``cross_doc_threshold`` (defaults to ``similarity_threshold - 0.10``).
    * **Surplus rollover:** Unused slots from either tier transfer to the other.

    Backward-compatible: with ``min_cross_doc_edges=0`` and
    ``cross_doc_threshold=None``, behavior is identical to the original.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    coreference_groups:
        Optional coreference groups from EXPLORER.
    similarity_threshold:
        Minimum cosine similarity for intra-doc synonymy edges.
    cross_doc_threshold:
        Minimum cosine similarity for cross-doc synonymy edges.
        Defaults to ``similarity_threshold - 0.10`` when ``min_cross_doc_edges > 0``.
    similarity_index:
        Optional pre-built ``SimilarityIndex``. When provided, used for
        per-node neighbor lookup instead of brute-force cosine matrix.
    max_edges_per_node:
        Maximum synonymy edges per node (cap).
    min_cross_doc_edges:
        Minimum reserved slots per node for cross-document edges.

    Returns
    -------
    List[StructuralEdge]
        Synonymy edges between similar/coreferent nodes.
    """
    # Resolve effective cross-doc threshold
    if cross_doc_threshold is None:
        effective_cross_threshold = (
            similarity_threshold - 0.10 if min_cross_doc_edges > 0
            else similarity_threshold
        )
    else:
        effective_cross_threshold = cross_doc_threshold

    edges: List[StructuralEdge] = []
    seen_pairs: set = set()

    # --- Coreference-based synonymy ---
    if coreference_groups:
        for group in coreference_groups:
            all_variants_normalized = {
                _normalize_for_coref(group.canonical_form)
            } | {_normalize_for_coref(v) for v in group.variants}
            # Find nodes whose canonical_text matches any variant (case-insensitive)
            matching_ids: List[str] = []
            for node_id, node in nodes.items():
                if _normalize_for_coref(node.canonical_text) in all_variants_normalized:
                    matching_ids.append(node_id)

            # Create edges between each pair
            for i in range(len(matching_ids)):
                for j in range(i + 1, len(matching_ids)):
                    a, b = matching_ids[i], matching_ids[j]
                    pair = (min(a, b), max(a, b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    edges.append(StructuralEdge(
                        edge_id=_make_edge_id(),
                        source=a,
                        target=b,
                        edge_type=StructuralEdgeType.SYNONYMY,
                        weight=1.0,
                        metadata={"origin": "coreference", "canonical": group.canonical_form},
                        similarity=1.0,
                        is_cross_document=_is_cross_document(a, b, nodes),
                    ))

    # --- Embedding-based synonymy ---
    query_threshold = min(similarity_threshold, effective_cross_threshold)

    if similarity_index is not None:
        # Use pre-built index for per-node neighbor lookup (two-pass)
        edge_count_per_node: Dict[str, int] = defaultdict(int)
        cross_count_per_node: Dict[str, int] = defaultdict(int)

        for node_id, node in nodes.items():
            if not node.embedding:
                continue
            if edge_count_per_node[node_id] >= max_edges_per_node:
                continue

            # Query with wider threshold and more candidates to capture cross-doc
            neighbors = similarity_index.query(
                node.embedding,
                top_k=max_edges_per_node * 3,
                threshold=query_threshold,
            )

            # Separate into intra-doc and cross-doc candidates
            intra_candidates: List[tuple] = []
            cross_candidates: List[tuple] = []
            for neighbor_id, sim in neighbors:
                if neighbor_id == node_id:
                    continue
                pair = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if pair in seen_pairs:
                    continue
                is_cross = _is_cross_document(node_id, neighbor_id, nodes)
                if is_cross:
                    if sim >= effective_cross_threshold:
                        cross_candidates.append((neighbor_id, sim, pair))
                else:
                    if sim >= similarity_threshold:
                        intra_candidates.append((neighbor_id, sim, pair))

            # Two-tier allocation
            remaining = max_edges_per_node - edge_count_per_node[node_id]
            reserved_cross = min(min_cross_doc_edges - cross_count_per_node[node_id], remaining)
            reserved_cross = max(reserved_cross, 0)
            intra_budget = remaining - reserved_cross

            # Tier 1: fill intra-doc slots
            added_intra = 0
            for neighbor_id, sim, pair in intra_candidates:
                if added_intra >= intra_budget:
                    break
                if edge_count_per_node[neighbor_id] >= max_edges_per_node:
                    continue
                seen_pairs.add(pair)
                edge_count_per_node[node_id] += 1
                edge_count_per_node[neighbor_id] += 1
                added_intra += 1
                edges.append(StructuralEdge(
                    edge_id=_make_edge_id(),
                    source=node_id,
                    target=neighbor_id,
                    edge_type=StructuralEdgeType.SYNONYMY,
                    weight=sim,
                    metadata={"origin": "embedding_similarity"},
                    similarity=sim,
                    is_cross_document=False,
                ))

            # Surplus rollover: unused intra slots go to cross-doc
            cross_budget = reserved_cross + (intra_budget - added_intra)

            # Tier 2: fill cross-doc slots
            added_cross = 0
            for neighbor_id, sim, pair in cross_candidates:
                if added_cross >= cross_budget:
                    break
                if edge_count_per_node[neighbor_id] >= max_edges_per_node:
                    continue
                seen_pairs.add(pair)
                edge_count_per_node[node_id] += 1
                edge_count_per_node[neighbor_id] += 1
                cross_count_per_node[node_id] += 1
                cross_count_per_node[neighbor_id] += 1
                added_cross += 1
                edges.append(StructuralEdge(
                    edge_id=_make_edge_id(),
                    source=node_id,
                    target=neighbor_id,
                    edge_type=StructuralEdgeType.SYNONYMY,
                    weight=sim,
                    metadata={"origin": "embedding_similarity"},
                    similarity=sim,
                    is_cross_document=True,
                ))
    else:
        # Brute-force cosine similarity matrix (two-pass)
        node_ids: List[str] = []
        embedding_list: List[List[float]] = []
        for node_id, node in nodes.items():
            if node.embedding:
                node_ids.append(node_id)
                embedding_list.append(node.embedding)

        if len(node_ids) < 2:
            return edges

        mat = np.array(embedding_list, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms

        # Full cosine similarity matrix
        sim_matrix = mat @ mat.T

        # Build per-node candidate pools separated by document membership
        intra_candidates_bf: Dict[int, List[tuple]] = defaultdict(list)
        cross_candidates_bf: Dict[int, List[tuple]] = defaultdict(list)

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                sim = float(sim_matrix[i, j])
                pair = (min(node_ids[i], node_ids[j]), max(node_ids[i], node_ids[j]))
                if pair in seen_pairs:
                    continue
                is_cross = _is_cross_document(node_ids[i], node_ids[j], nodes)
                if is_cross:
                    if sim >= effective_cross_threshold:
                        cross_candidates_bf[i].append((j, sim, pair))
                        cross_candidates_bf[j].append((i, sim, pair))
                else:
                    if sim > similarity_threshold:
                        intra_candidates_bf[i].append((j, sim, pair))
                        intra_candidates_bf[j].append((i, sim, pair))

        # Two-tier greedy allocation
        accepted_pairs: set = set()
        bf_edge_count: Dict[int, int] = defaultdict(int)
        bf_cross_count: Dict[int, int] = defaultdict(int)

        # Sort intra candidates globally by similarity descending
        all_intra = []
        seen_intra_pairs: set = set()
        for i, cands in intra_candidates_bf.items():
            for j, sim, pair in cands:
                if pair not in seen_intra_pairs:
                    all_intra.append((sim, i, j, pair))
                    seen_intra_pairs.add(pair)
        all_intra.sort(key=lambda x: -x[0])

        # Sort cross candidates globally by similarity descending
        all_cross = []
        seen_cross_pairs: set = set()
        for i, cands in cross_candidates_bf.items():
            for j, sim, pair in cands:
                if pair not in seen_cross_pairs:
                    all_cross.append((sim, i, j, pair))
                    seen_cross_pairs.add(pair)
        all_cross.sort(key=lambda x: -x[0])

        # Tier 1: intra-doc edges (up to max - reserved per node)
        for sim, i, j, pair in all_intra:
            if pair in accepted_pairs:
                continue
            i_remaining = max_edges_per_node - bf_edge_count[i]
            j_remaining = max_edges_per_node - bf_edge_count[j]
            i_reserved = max(min_cross_doc_edges - bf_cross_count[i], 0)
            j_reserved = max(min_cross_doc_edges - bf_cross_count[j], 0)
            if i_remaining <= i_reserved or j_remaining <= j_reserved:
                continue
            accepted_pairs.add(pair)
            bf_edge_count[i] += 1
            bf_edge_count[j] += 1

        # Tier 2: cross-doc edges (reserved slots + surplus)
        for sim, i, j, pair in all_cross:
            if pair in accepted_pairs:
                continue
            if (bf_edge_count[i] >= max_edges_per_node or
                    bf_edge_count[j] >= max_edges_per_node):
                continue
            accepted_pairs.add(pair)
            bf_edge_count[i] += 1
            bf_edge_count[j] += 1
            bf_cross_count[i] += 1
            bf_cross_count[j] += 1

        # Build index lookup for fast pair → similarity
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        for pair in accepted_pairs:
            a, b = pair
            idx_a = node_id_to_idx.get(a, -1)
            idx_b = node_id_to_idx.get(b, -1)
            sim = float(sim_matrix[idx_a, idx_b]) if idx_a >= 0 and idx_b >= 0 else 0.0
            edges.append(StructuralEdge(
                edge_id=_make_edge_id(),
                source=a,
                target=b,
                edge_type=StructuralEdgeType.SYNONYMY,
                weight=sim,
                metadata={"origin": "embedding_similarity"},
                similarity=sim,
                is_cross_document=_is_cross_document(a, b, nodes),
            ))

    return edges


def boost_bridge_edges(
    edges: List[StructuralEdge],
    multiplier: float = 1.5,
) -> List[StructuralEdge]:
    """Apply a weight multiplier to cross-document (bridge) edges.

    This encourages PPR mass to preferentially flow across document
    boundaries, counteracting the Semantic Sink effect where intra-
    document hub nodes absorb disproportionate mass.

    Parameters
    ----------
    edges:
        All structural edges (modified in-place and returned).
    multiplier:
        Weight multiplier for ``is_cross_document=True`` edges.

    Returns
    -------
    List[StructuralEdge]
        The same list with bridge edge weights boosted.
    """
    for edge in edges:
        if edge.is_cross_document:
            edge.weight *= multiplier
    return edges


def discount_synonymy_by_degree(
    edges: List[StructuralEdge],
    nodes: Dict[str, EmbeddedNode],
) -> List[StructuralEdge]:
    """Apply degree-based discount to SYNONYMY edges to prevent similarity clusters.

    High-degree nodes in similarity clusters (e.g., multiple claims about "Elias
    Thorne graduated") create a "Similarity Web" that traps PPR mass. This function
    reduces SYNONYMY edge weights based on the average degree of the endpoints,
    preventing high-weight clusters from dominating scope selection.

    Formula: discounted_weight = original_weight / log2(avg_degree + 2)

    Parameters
    ----------
    edges:
        All structural edges (SYNONYMY edges modified in-place).
    nodes:
        All nodes in the graph (used to validate endpoints exist).

    Returns
    -------
    List[StructuralEdge]
        The same list with SYNONYMY edge weights discounted by degree.
    """
    # Compute degree for each node
    from collections import defaultdict
    degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        if edge.source in nodes:
            degree[edge.source] += 1
        if edge.target in nodes:
            degree[edge.target] += 1

    # Discount SYNONYMY edges based on average endpoint degree
    for edge in edges:
        if edge.edge_type != StructuralEdgeType.SYNONYMY:
            continue
        src_degree = degree.get(edge.source, 0)
        tgt_degree = degree.get(edge.target, 0)
        avg_degree = (src_degree + tgt_degree) / 2.0
        discount_factor = math.log2(avg_degree + 2)
        edge.weight /= discount_factor

    return edges
