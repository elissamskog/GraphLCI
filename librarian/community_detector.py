"""
Community Detector (v5)
Location: librarian/community_detector.py
Owner: LIBRARIAN Instance

Detects hierarchical communities in the knowledge graph.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from schemas.contracts import (
    Community,
    CommunityStructure,
    KnowledgeGraph,
)


def _build_weighted_adjacency(
    graph: KnowledgeGraph,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Build weighted undirected adjacency from graph edges.

    Returns adjacency dict and total edge weight (2m).
    """
    adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_weight = 0.0

    for edge in graph.edges.values():
        s, t = edge.source, edge.target
        if s not in graph.nodes or t not in graph.nodes:
            continue
        w = edge.weight
        adj[s][t] += w
        adj[t][s] += w
        total_weight += w

    return dict(adj), total_weight


def _louvain_phase(
    nodes: List[str],
    adj: Dict[str, Dict[str, float]],
    total_weight: float,
    resolution: float = 1.0,
) -> Dict[str, str]:
    """Run one phase of Louvain: greedy modularity optimisation.

    Parameters
    ----------
    resolution:
        Resolution parameter that scales the degree penalty term in the
        modularity gain formula. Higher values produce more communities.

    Returns mapping of node_id -> community_id.
    """
    if total_weight == 0:
        # No edges: each node is its own community
        return {n: n for n in nodes}

    m2 = 2.0 * total_weight  # 2m

    # Initial assignment: each node in its own community
    node_to_comm: Dict[str, str] = {n: n for n in nodes}

    # Degree of each node (sum of edge weights)
    k: Dict[str, float] = {}
    for n in nodes:
        k[n] = sum(adj.get(n, {}).values())

    # Community aggregates
    # sigma_tot: total degree of community
    sigma_tot: Dict[str, float] = {n: k[n] for n in nodes}
    # sigma_in: internal edge weight of community (initially 0 — no self-loops)
    sigma_in: Dict[str, float] = {n: 0.0 for n in nodes}
    # Members of each community
    comm_members: Dict[str, Set[str]] = {n: {n} for n in nodes}

    improved = True
    max_passes = 50
    pass_count = 0

    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1

        for node in nodes:
            current_comm = node_to_comm[node]

            # Compute k_{i,in} for current community (edges from node to members of its community)
            ki_in_current = 0.0
            for neighbor, w in adj.get(node, {}).items():
                if node_to_comm.get(neighbor) == current_comm:
                    ki_in_current += w

            # Remove node from its community
            sigma_tot[current_comm] -= k[node]
            sigma_in[current_comm] -= ki_in_current
            comm_members[current_comm].discard(node)

            # Evaluate moving to each neighbour's community
            best_comm = current_comm
            best_gain = 0.0

            # Compute set of candidate communities from neighbours
            candidate_comms: Set[str] = {current_comm}
            for neighbor in adj.get(node, {}):
                candidate_comms.add(node_to_comm[neighbor])

            for candidate in candidate_comms:
                # k_{i,in} for candidate community
                ki_in_cand = 0.0
                for neighbor, w in adj.get(node, {}).items():
                    if node_to_comm.get(neighbor) == candidate:
                        ki_in_cand += w

                # Modularity gain:
                # ΔQ = ki_in_cand/m2 - sigma_tot[candidate] * k[node] / (m2^2/2)
                # Simplified: ΔQ = (ki_in_cand - sigma_tot[candidate] * k[node] / m2) / m2
                # But we compare relative to returning to current_comm, which would be 0
                # so just compute the absolute gain for moving to candidate.
                delta_q = (ki_in_cand / m2) - resolution * (sigma_tot[candidate] * k[node]) / (m2 * m2 / 2.0)

                if delta_q > best_gain:
                    best_gain = delta_q
                    best_comm = candidate

            # Move node to best community
            node_to_comm[node] = best_comm

            ki_in_best = 0.0
            for neighbor, w in adj.get(node, {}).items():
                if node_to_comm.get(neighbor) == best_comm:
                    ki_in_best += w

            sigma_tot[best_comm] += k[node]
            sigma_in[best_comm] += ki_in_best
            comm_members[best_comm].add(node)

            if best_comm != current_comm:
                improved = True

                # Clean up empty community
                if not comm_members[current_comm]:
                    del comm_members[current_comm]
                    del sigma_tot[current_comm]
                    del sigma_in[current_comm]

    return node_to_comm


def _coarsen(
    node_to_comm: Dict[str, str],
    adj: Dict[str, Dict[str, float]],
) -> Tuple[List[str], Dict[str, Dict[str, float]], float, Dict[str, List[str]]]:
    """Coarsen the graph by merging communities into super-nodes.

    Returns (new_nodes, new_adj, new_total_weight, comm_to_members).
    """
    # Group nodes by community
    comm_members: Dict[str, List[str]] = defaultdict(list)
    for node, comm in node_to_comm.items():
        comm_members[comm].append(node)

    new_nodes = list(comm_members.keys())

    # Build coarsened adjacency
    new_adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    new_total_weight = 0.0

    seen_edges: Set[Tuple[str, str]] = set()
    for node, neighbors in adj.items():
        comm_s = node_to_comm[node]
        for neighbor, w in neighbors.items():
            comm_t = node_to_comm[neighbor]
            if comm_s == comm_t:
                continue  # internal edge in coarsened graph
            pair = (min(comm_s, comm_t), max(comm_s, comm_t))
            new_adj[comm_s][comm_t] += w
            new_adj[comm_t][comm_s] += w
            if pair not in seen_edges:
                seen_edges.add(pair)
                new_total_weight += w

    return new_nodes, dict(new_adj), new_total_weight, dict(comm_members)


def _merge_small_communities(
    node_to_comm: Dict[str, str],
    adj: Dict[str, Dict[str, float]],
    min_community_size: int,
) -> Dict[str, str]:
    """Merge communities smaller than *min_community_size* into the
    most-connected neighbouring community.

    Mutates nothing — returns a new mapping.
    """
    # Group by community
    comm_members: Dict[str, List[str]] = defaultdict(list)
    for node, comm in node_to_comm.items():
        comm_members[comm].append(node)

    small_comms = [c for c, members in comm_members.items() if len(members) < min_community_size]
    if not small_comms:
        return node_to_comm

    result = dict(node_to_comm)

    for small_comm in small_comms:
        members = comm_members.get(small_comm)
        if members is None:
            continue
        # Already merged if members have been reassigned
        current_members = [n for n in members if result[n] == small_comm]
        if not current_members:
            continue

        # Find the neighbouring community with the strongest connection
        neighbor_weights: Dict[str, float] = defaultdict(float)
        for node in current_members:
            for neighbor, w in adj.get(node, {}).items():
                nc = result[neighbor]
                if nc != small_comm:
                    neighbor_weights[nc] += w

        if not neighbor_weights:
            # No external neighbors — leave as-is
            continue

        best_neighbor = max(neighbor_weights, key=neighbor_weights.get)  # type: ignore[arg-type]

        # Reassign all members of this small community
        for node in current_members:
            result[node] = best_neighbor

        # Update comm_members for the target so subsequent iterations see them
        comm_members.setdefault(best_neighbor, []).extend(current_members)

    return result


def detect_communities(
    graph: KnowledgeGraph,
    max_levels: int = 3,
    resolution: float = 1.0,
    min_community_size: int = 3,
) -> CommunityStructure:
    """Detect hierarchical communities in the knowledge graph.

    Uses a Louvain-inspired greedy modularity optimisation.

    Parameters
    ----------
    graph:
        The knowledge graph to partition.
    max_levels:
        Maximum hierarchy depth for community detection.
    resolution:
        Resolution parameter for modularity gain. Higher values produce
        more (smaller) communities.
    min_community_size:
        Communities smaller than this are merged into their most-connected
        neighbour after each Louvain phase.

    Returns
    -------
    CommunityStructure
        Hierarchical community assignments.
    """
    all_communities: Dict[str, Community] = {}
    levels: Dict[int, List[str]] = {}
    node_to_community: Dict[str, str] = {}

    node_ids = list(graph.nodes.keys())

    if not node_ids:
        return CommunityStructure(
            communities={},
            levels={},
            node_to_community={},
        )

    adj, total_weight = _build_weighted_adjacency(graph)

    current_nodes = node_ids
    current_adj = adj
    current_total = total_weight

    # Track which original nodes map to current super-nodes
    # Initially each original node maps to itself
    supernode_to_originals: Dict[str, List[str]] = {n: [n] for n in node_ids}

    for level in range(max_levels):
        # Phase 1: greedy modularity optimisation
        node_to_comm = _louvain_phase(current_nodes, current_adj, current_total, resolution=resolution)

        # Merge small communities
        if min_community_size > 1:
            node_to_comm = _merge_small_communities(node_to_comm, current_adj, min_community_size)

        # Group into communities
        comm_groups: Dict[str, List[str]] = defaultdict(list)
        for node, comm in node_to_comm.items():
            comm_groups[comm].append(node)

        # If no change (each node stayed in its own community), stop
        if len(comm_groups) >= len(current_nodes):
            # Still create level-0 communities if this is the first level
            if level == 0:
                level_comm_ids: List[str] = []
                for comm_label, members in comm_groups.items():
                    comm_id = uuid.uuid4().hex[:12]
                    original_nodes: List[str] = []
                    for m in members:
                        original_nodes.extend(supernode_to_originals.get(m, [m]))

                    community = Community(
                        community_id=comm_id,
                        level=level,
                        node_ids=original_nodes,
                    )
                    all_communities[comm_id] = community
                    level_comm_ids.append(comm_id)

                    for orig in original_nodes:
                        node_to_community[orig] = comm_id

                levels[level] = level_comm_ids
            break

        # Create Community objects for this level
        level_comm_ids = []
        new_supernode_to_originals: Dict[str, List[str]] = {}

        for comm_label, members in comm_groups.items():
            comm_id = uuid.uuid4().hex[:12]

            # Collect all original node_ids that belong to this community
            original_nodes = []
            for m in members:
                original_nodes.extend(supernode_to_originals.get(m, [m]))

            community = Community(
                community_id=comm_id,
                level=level,
                node_ids=original_nodes,
            )
            all_communities[comm_id] = community
            level_comm_ids.append(comm_id)

            # For level 0, map original nodes to their community
            if level == 0:
                for orig in original_nodes:
                    node_to_community[orig] = comm_id

            new_supernode_to_originals[comm_id] = original_nodes

        levels[level] = level_comm_ids

        # Set parent/child relationships with previous level
        if level > 0:
            prev_level_ids = levels.get(level - 1, [])
            for comm_id in level_comm_ids:
                parent_comm = all_communities[comm_id]
                parent_nodes = set(parent_comm.node_ids)
                children: List[str] = []
                for prev_comm_id in prev_level_ids:
                    prev_comm = all_communities[prev_comm_id]
                    if set(prev_comm.node_ids).issubset(parent_nodes):
                        children.append(prev_comm_id)
                        prev_comm.parent_community_id = comm_id
                parent_comm.child_community_ids = children

        # Phase 2: coarsen for next level
        if len(comm_groups) <= 1:
            break

        current_nodes, current_adj, current_total, _ = _coarsen(
            node_to_comm, current_adj
        )
        supernode_to_originals = new_supernode_to_originals

    return CommunityStructure(
        communities=all_communities,
        levels=levels,
        node_to_community=node_to_community,
    )
