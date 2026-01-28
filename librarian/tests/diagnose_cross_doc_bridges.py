"""
Diagnostic script for cross-document bridge analysis.
Location: librarian/tests/diagnose_cross_doc_bridges.py
Owner: LIBRARIAN Instance

Provides three diagnostic functions to investigate sparse cross-doc bridges:
1. Similarity distribution across document boundaries
2. Per-node edge cap saturation
3. Coreference matching effectiveness

Usage:
    python -m librarian.tests.diagnose_cross_doc_bridges
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from schemas.contracts import (
    CoreferenceGroup,
    EmbeddedNode,
    StructuralEdge,
    StructuralEdgeType,
)

from librarian.edge_builder import create_synonymy_edges, _is_cross_document


def diagnose_similarity_distribution(
    nodes: Dict[str, EmbeddedNode],
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Compute cosine similarity for all cross-doc entity pairs.

    Reports counts at various thresholds and basic statistics.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    thresholds:
        Similarity thresholds to report counts for.

    Returns
    -------
    dict with keys: counts_at_threshold, mean, median, max, total_cross_doc_pairs
    """
    if thresholds is None:
        thresholds = [0.80, 0.75, 0.70, 0.65]

    # Group nodes by document
    doc_groups: Dict[str, List[str]] = defaultdict(list)
    for nid, node in nodes.items():
        if node.embedding:
            doc_groups[node.source_doc_id].append(nid)

    doc_ids = sorted(doc_groups.keys())
    if len(doc_ids) < 2:
        return {
            "counts_at_threshold": {t: 0 for t in thresholds},
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "total_cross_doc_pairs": 0,
        }

    # Compute all cross-doc similarities
    cross_doc_sims: List[float] = []

    for di in range(len(doc_ids)):
        for dj in range(di + 1, len(doc_ids)):
            nodes_i = doc_groups[doc_ids[di]]
            nodes_j = doc_groups[doc_ids[dj]]
            for ni in nodes_i:
                emb_i = np.array(nodes[ni].embedding, dtype=np.float32)
                norm_i = np.linalg.norm(emb_i)
                if norm_i == 0:
                    continue
                emb_i = emb_i / norm_i
                for nj in nodes_j:
                    emb_j = np.array(nodes[nj].embedding, dtype=np.float32)
                    norm_j = np.linalg.norm(emb_j)
                    if norm_j == 0:
                        continue
                    emb_j = emb_j / norm_j
                    sim = float(emb_i @ emb_j)
                    cross_doc_sims.append(sim)

    if not cross_doc_sims:
        return {
            "counts_at_threshold": {t: 0 for t in thresholds},
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "total_cross_doc_pairs": 0,
        }

    counts_at_threshold = {
        t: sum(1 for s in cross_doc_sims if s >= t) for t in thresholds
    }

    return {
        "counts_at_threshold": counts_at_threshold,
        "mean": statistics.mean(cross_doc_sims),
        "median": statistics.median(cross_doc_sims),
        "max": max(cross_doc_sims),
        "total_cross_doc_pairs": len(cross_doc_sims),
    }


def diagnose_cap_saturation(
    nodes: Dict[str, EmbeddedNode],
    coreference_groups: Optional[List[CoreferenceGroup]] = None,
    similarity_threshold: float = 0.80,
    max_edges_per_node: int = 10,
) -> Dict:
    """Run current create_synonymy_edges() and analyze cap saturation.

    Reports how many nodes hit the edge cap, with intra- vs cross-doc
    breakdown.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    coreference_groups:
        Optional coreference groups.
    similarity_threshold:
        Minimum similarity for synonymy edges.
    max_edges_per_node:
        Cap on edges per node.

    Returns
    -------
    dict with keys: total_edges, cross_doc_edges, intra_doc_edges,
    nodes_at_cap, per_node_breakdown (dict: node_id -> {total, intra, cross})
    """
    edges = create_synonymy_edges(
        nodes,
        coreference_groups=coreference_groups,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_node,
    )

    synonymy = [e for e in edges if e.edge_type == StructuralEdgeType.SYNONYMY]
    cross_doc = [e for e in synonymy if e.is_cross_document]
    intra_doc = [e for e in synonymy if not e.is_cross_document]

    # Per-node breakdown
    node_edges: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "intra": 0, "cross": 0})
    for e in synonymy:
        for nid in (e.source, e.target):
            node_edges[nid]["total"] += 1
            if e.is_cross_document:
                node_edges[nid]["cross"] += 1
            else:
                node_edges[nid]["intra"] += 1

    nodes_at_cap = sum(
        1 for info in node_edges.values() if info["total"] >= max_edges_per_node
    )

    return {
        "total_edges": len(synonymy),
        "cross_doc_edges": len(cross_doc),
        "intra_doc_edges": len(intra_doc),
        "nodes_at_cap": nodes_at_cap,
        "per_node_breakdown": dict(node_edges),
    }


def diagnose_coreference(
    nodes: Dict[str, EmbeddedNode],
    coreference_groups: List[CoreferenceGroup],
) -> Dict:
    """Analyze coreference matching effectiveness.

    Reports how many coreference groups produce matches and how many
    of those are cross-document.

    Parameters
    ----------
    nodes:
        All nodes keyed by node_id.
    coreference_groups:
        Coreference groups to analyze.

    Returns
    -------
    dict with keys: groups_passed, groups_with_matches, cross_doc_coref_edges,
    total_coref_edges, unmatched_variants
    """
    groups_with_matches = 0
    total_coref_edges = 0
    cross_doc_coref_edges = 0
    unmatched_variants: List[Dict] = []

    for group in coreference_groups:
        all_variants = {group.canonical_form} | set(group.variants)
        matching_ids: List[str] = []
        matched_texts: set = set()

        for node_id, node in nodes.items():
            if node.canonical_text in all_variants:
                matching_ids.append(node_id)
                matched_texts.add(node.canonical_text)

        if len(matching_ids) >= 2:
            groups_with_matches += 1

        # Count edges that would be created
        for i in range(len(matching_ids)):
            for j in range(i + 1, len(matching_ids)):
                total_coref_edges += 1
                if _is_cross_document(matching_ids[i], matching_ids[j], nodes):
                    cross_doc_coref_edges += 1

        # Track unmatched variants
        unmatched = all_variants - matched_texts
        if unmatched:
            unmatched_variants.append({
                "canonical": group.canonical_form,
                "unmatched": list(unmatched),
            })

    return {
        "groups_passed": len(coreference_groups),
        "groups_with_matches": groups_with_matches,
        "cross_doc_coref_edges": cross_doc_coref_edges,
        "total_coref_edges": total_coref_edges,
        "unmatched_variants": unmatched_variants,
    }


def run_diagnostics(
    nodes: Dict[str, EmbeddedNode],
    coreference_groups: Optional[List[CoreferenceGroup]] = None,
) -> None:
    """Run all diagnostics and print results."""
    print("=" * 60)
    print("CROSS-DOCUMENT BRIDGE DIAGNOSTICS")
    print("=" * 60)

    # 1. Similarity distribution
    print("\n--- 1. Similarity Distribution (cross-doc pairs) ---")
    sim_report = diagnose_similarity_distribution(nodes)
    print(f"  Total cross-doc pairs: {sim_report['total_cross_doc_pairs']}")
    print(f"  Mean similarity:  {sim_report['mean']:.4f}")
    print(f"  Median similarity: {sim_report['median']:.4f}")
    print(f"  Max similarity:   {sim_report['max']:.4f}")
    for t, count in sorted(sim_report["counts_at_threshold"].items(), reverse=True):
        print(f"  Pairs >= {t:.2f}: {count}")

    # 2. Cap saturation
    print("\n--- 2. Cap Saturation (max_edges_per_node=10) ---")
    cap_report = diagnose_cap_saturation(
        nodes,
        coreference_groups=coreference_groups,
    )
    print(f"  Total synonymy edges: {cap_report['total_edges']}")
    print(f"  Cross-doc edges:      {cap_report['cross_doc_edges']}")
    print(f"  Intra-doc edges:      {cap_report['intra_doc_edges']}")
    print(f"  Nodes at cap (10):    {cap_report['nodes_at_cap']}")

    # 3. Coreference
    if coreference_groups:
        print("\n--- 3. Coreference Matching ---")
        coref_report = diagnose_coreference(nodes, coreference_groups)
        print(f"  Groups passed:           {coref_report['groups_passed']}")
        print(f"  Groups with matches:     {coref_report['groups_with_matches']}")
        print(f"  Total coref edges:       {coref_report['total_coref_edges']}")
        print(f"  Cross-doc coref edges:   {coref_report['cross_doc_coref_edges']}")
        if coref_report["unmatched_variants"]:
            print(f"  Unmatched variants ({len(coref_report['unmatched_variants'])} groups):")
            for entry in coref_report["unmatched_variants"][:5]:
                print(f"    {entry['canonical']}: {entry['unmatched']}")
    else:
        print("\n--- 3. Coreference Matching ---")
        print("  No coreference groups provided.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # When run standalone, create synthetic data for testing
    from librarian.tests.test_v5_pipeline import make_node

    nodes = {
        "n1": make_node("n1", "Contract Law", "doc1", [0.9, 0.1, 0.0, 0.0]),
        "n2": make_node("n2", "Tort Law", "doc1", [0.1, 0.9, 0.0, 0.0]),
        "n3": make_node("n3", "contract law", "doc2", [0.85, 0.15, 0.0, 0.0]),
        "n4": make_node("n4", "Negligence", "doc2", [0.0, 0.8, 0.2, 0.0]),
        "n5": make_node("n5", "Strict Liability", "doc2", [0.0, 0.1, 0.9, 0.0]),
        "n6": make_node("n6", "Product Liability", "doc3", [0.0, 0.15, 0.85, 0.0]),
        "n7": make_node("n7", "Contract Breach", "doc3", [0.8, 0.2, 0.0, 0.0]),
    }

    coref_groups = [
        CoreferenceGroup(
            canonical_form="Contract Law",
            variants=["contract law", "Contract Breach"],
        ),
    ]

    run_diagnostics(nodes, coref_groups)
