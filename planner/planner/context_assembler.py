"""
Context Assembler (v5)
Location: planner/planner/context_assembler.py
Owner: PLANNER Instance

Assembles context payloads for each lane type from the scoped subgraph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from schemas.contracts import (
    CommunityStructure,
    FocusedContext,
    GlobalContext,
    KnowledgeGraph,
    RankedChunk,
    RawDocument,
    Scope,
    StructuralEdgeType,
    TargetedContext,
    TextChunk,
)


# ============================================================================
# Edge type display names
# ============================================================================

_EDGE_TYPE_LABELS = {
    StructuralEdgeType.SYNONYMY: "Semantic Bridges",
    StructuralEdgeType.CITES: "Citations",
    StructuralEdgeType.REFERENCES: "References",
    StructuralEdgeType.SAME_SOURCE: "Same-Source Links",
    StructuralEdgeType.SAME_CHUNK: "Same-Chunk Links",
    StructuralEdgeType.DEFINED_BY: "Definitions",
}


def assemble_focused_context(
    scope: Scope,
    graph: KnowledgeGraph,
    document_store: Dict[str, RawDocument],
) -> FocusedContext:
    """Assemble focused context for Lane A (Graph-LCI).

    Builds edge grammar, node context, and includes full documents.

    Parameters
    ----------
    scope:
        The PPR-expanded scope.
    graph:
        The full knowledge graph.
    document_store:
        Available documents keyed by doc_id.

    Returns
    -------
    FocusedContext
        Context with documents, edge grammar, and node context.
    """
    # 1. Collect documents referenced by scope
    documents: Dict[str, RawDocument] = {}
    for doc_id in scope.document_ids:
        if doc_id in document_store:
            documents[doc_id] = document_store[doc_id]

    # 2. Build edge grammar string
    edge_grammar = _build_edge_grammar(scope, graph)

    # 3. Build node context string
    node_context = _build_node_context(scope)

    return FocusedContext(
        documents=documents,
        edge_grammar=edge_grammar,
        node_context=node_context,
        scope=scope,
    )


def _build_edge_grammar(scope: Scope, graph: KnowledgeGraph) -> str:
    """Build human-readable edge grammar from scope edges."""
    if not scope.edges:
        return "No structural edges in scope."

    # Group edges by type
    grouped: Dict[StructuralEdgeType, list] = defaultdict(list)
    for edge in scope.edges.values():
        grouped[edge.edge_type].append(edge)

    lines: List[str] = []
    for edge_type, edges in grouped.items():
        label = _EDGE_TYPE_LABELS.get(edge_type, edge_type.value)
        lines.append(f"=== {label} ===")

        for edge in edges:
            source_node = graph.nodes.get(edge.source)
            target_node = graph.nodes.get(edge.target)
            source_text = source_node.canonical_text if source_node else edge.source
            target_text = target_node.canonical_text if target_node else edge.target
            source_doc = source_node.source_doc_id if source_node else "?"
            target_doc = target_node.source_doc_id if target_node else "?"

            arrow = "-->" if not edge.is_cross_document else "==>"
            line = (
                f'  "{source_text}" (Doc: {source_doc}) '
                f'{arrow} "{target_text}" (Doc: {target_doc}) '
                f'[weight: {edge.weight:.2f}]'
            )

            if edge.edge_type == StructuralEdgeType.SYNONYMY and edge.similarity is not None:
                line += f" (similarity: {edge.similarity:.3f})"

            if edge.is_cross_document:
                line += " [CROSS-DOC]"

            lines.append(line)

        lines.append("")  # blank line between groups

    return "\n".join(lines).strip()


def _build_node_context(scope: Scope) -> str:
    """Build human-readable node context from scoped nodes."""
    if not scope.nodes:
        return "No nodes in scope."

    # Sort by rank
    sorted_nodes = sorted(scope.nodes.values(), key=lambda sn: sn.rank)

    lines: List[str] = []
    display_count = min(10, len(sorted_nodes))
    for sn in sorted_nodes[:display_count]:
        lines.append(
            f'{sn.rank}. [{sn.node.node_type}] "{sn.node.canonical_text}" '
            f'(Doc: {sn.node.source_doc_id}, relevance: {sn.ppr_score:.4f})'
        )

    remaining = len(sorted_nodes) - display_count
    if remaining > 0:
        lines.append(f"... and {remaining} more nodes")

    return "\n".join(lines)


def assemble_global_context(
    scope: Scope,
    community_structure: CommunityStructure,
    community_summaries: Dict[str, str],
) -> GlobalContext:
    """Assemble global context for Lane B (Graph-only).

    Uses community summaries relevant to the scope.

    Parameters
    ----------
    scope:
        The PPR-expanded scope.
    community_structure:
        Community assignments.
    community_summaries:
        Pre-computed community summaries.

    Returns
    -------
    GlobalContext
        Context with relevant community summaries.
    """
    # Find communities containing scoped nodes
    relevant_community_ids: set = set()
    for node_id in scope.nodes:
        community_id = community_structure.node_to_community.get(node_id)
        if community_id is not None:
            relevant_community_ids.add(community_id)

    # Filter summaries to only relevant communities
    filtered_summaries: Dict[str, str] = {
        cid: summary
        for cid, summary in community_summaries.items()
        if cid in relevant_community_ids
    }

    return GlobalContext(
        community_summaries=filtered_summaries,
        scope=scope,
    )


def assemble_targeted_context(
    scope: Scope,
    chunks: List[TextChunk],
    query_embedding: List[float],
) -> TargetedContext:
    """Assemble targeted context for Lane C (Vector-only).

    Ranks chunks by relevance using PPR scores of nodes that reference them.

    Parameters
    ----------
    scope:
        The PPR-expanded scope.
    chunks:
        Available text chunks.
    query_embedding:
        The query's embedding vector (reserved for future cosine-sim extension).

    Returns
    -------
    TargetedContext
        Context with ranked chunks.
    """
    # Map chunk_id -> max PPR score across scope nodes referencing that chunk
    chunk_scores: Dict[str, float] = {}
    for scoped_node in scope.nodes.values():
        node = scoped_node.node
        for chunk_id in node.source_chunks:
            current = chunk_scores.get(chunk_id, 0.0)
            if scoped_node.ppr_score > current:
                chunk_scores[chunk_id] = scoped_node.ppr_score

    # Build ranked chunks from provided chunks list
    ranked: List[RankedChunk] = []
    for chunk in chunks:
        score = chunk_scores.get(chunk.chunk_id, 0.0)
        if score > 0.0:
            ranked.append(RankedChunk(chunk=chunk, relevance_score=score))

    # Sort by relevance descending, use position as tiebreaker
    ranked.sort(key=lambda rc: (-rc.relevance_score, rc.chunk.position))

    # Cap at 20
    ranked = ranked[:20]

    return TargetedContext(
        chunks=ranked,
        scope=scope,
    )
