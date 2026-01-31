"""
Personalized PageRank Traversal
Location: librarian/resolver/ppr_traversal.py
Owner: LIBRARIAN Instance

Responsibility: Execute Personalized PageRank over the knowledge graph to
find the subgraph most relevant to a set of seed (target) nodes.  The
algorithm is a pure-Python iterative power-iteration implementation that
operates on the in-memory ``GraphState`` — no external database required
at this layer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.contracts import GraphEdge, GraphNode, GraphState

logger = logging.getLogger(__name__)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class PPRInput:
    """Input to the PPR traversal."""
    seed_nodes: List[str]
    graph_state: GraphState
    damping_factor: float = 0.85
    max_iterations: int = 100
    min_score_threshold: float = 0.01
    max_hops: Optional[int] = None
    forward_weight_factor: float = 1.0   # v5: weight multiplier for Source -> Target
    backward_weight_factor: float = 0.5  # v5: weight multiplier for Target -> Source


@dataclass
class PPROutput:
    """Output of the PPR traversal."""
    ranked_nodes: List[Tuple[str, float]]
    traversed_edges: List[str]
    convergence_iterations: int


# ============================================================================
# GRAPH ADJACENCY BUILDER
# ============================================================================

class _AdjacencyGraph:
    def __init__(self, graph_state: GraphState, config: Optional[Dict[str, Any]] = None) -> None:
        self.node_ids: Set[str] = {n.node_id for n in graph_state.nodes}
        self.adj: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._build(graph_state.edges, config or {})

    def _build(self, edges: List[GraphEdge], config: Dict[str, Any]) -> None:
        f_factor = config.get("forward_weight_factor", 1.0)
        b_factor = config.get("backward_weight_factor", 0.5)

        for e in edges:
            if e.source_node not in self.node_ids or e.target_node not in self.node_ids:
                continue
            base_w = max(e.confidence, 0.01)

            # Forward edge (Source -> Target)
            self.adj[e.source_node].append((e.target_node, e.edge_id, base_w * f_factor))
            # Backward edge (Target -> Source)
            self.adj[e.target_node].append((e.source_node, e.edge_id, base_w * b_factor))

    def neighbours(self, node_id: str) -> List[Tuple[str, str, float]]:
        return self.adj.get(node_id, [])

    def out_weight(self, node_id: str) -> float:
        return sum(w for _, _, w in self.neighbours(node_id)) or 1.0

# ============================================================================
# PPR ENGINE
# ============================================================================

class PPRTraversal:
    """Personalized PageRank over a ``GraphState``.

    Parameters
    ----------
    config:
        Optional dictionary with override keys matching ``PPRInput`` fields:
        ``damping_factor``, ``max_iterations``, ``min_score_threshold``,
        ``max_hops``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def traverse(self, inp: PPRInput) -> PPROutput:
        """Run iterative PPR and return ranked nodes above threshold.

        The algorithm follows the standard power-iteration formulation::

            r(v) = (1 - d) * s(v) + d * Σ_{u→v} r(u) * w(u,v) / W_out(u)

        where *d* is the damping factor and *s* is the personalisation
        vector (uniform over seed nodes, zero elsewhere).
        """
        trav_cfg = {
            "forward_weight_factor": inp.forward_weight_factor,
            "backward_weight_factor": inp.backward_weight_factor,
        }
        graph = _AdjacencyGraph(inp.graph_state, config=trav_cfg)
        all_nodes = list(graph.node_ids)

        if not all_nodes:
            return PPROutput(ranked_nodes=[], traversed_edges=[], convergence_iterations=0)

        # Personalisation vector: uniform over seeds that exist in graph
        valid_seeds = [s for s in inp.seed_nodes if s in graph.node_ids]
        if not valid_seeds:
            logger.warning("No valid seed nodes found in graph; returning empty result")
            return PPROutput(ranked_nodes=[], traversed_edges=[], convergence_iterations=0)

        seed_weight = 1.0 / len(valid_seeds)
        personal: Dict[str, float] = defaultdict(float)
        for s in valid_seeds:
            personal[s] = seed_weight

        # Initialise scores to personalisation vector
        scores: Dict[str, float] = {n: personal.get(n, 0.0) for n in all_nodes}

        damping = inp.damping_factor
        teleport = 1.0 - damping
        converged_at = inp.max_iterations

        for iteration in range(1, inp.max_iterations + 1):
            # Initialise new scores with the teleport (personalisation) component
            new_scores: Dict[str, float] = {
                n: teleport * personal.get(n, 0.0) for n in all_nodes
            }

            # Push component: distribute current scores to neighbors
            for u in all_nodes:
                score_u = scores.get(u, 0.0)
                if score_u == 0:
                    continue
                
                out_w = graph.out_weight(u)
                for v, _, weight_u_v in graph.neighbours(u):
                    new_scores[v] += damping * score_u * (weight_u_v / out_w)

            # Check convergence (L1 norm)
            delta = sum(abs(new_scores[n] - scores[n]) for n in all_nodes)
            scores = new_scores

            if delta < 1e-8:
                converged_at = iteration
                logger.debug("PPR converged at iteration %d (delta=%.2e)", iteration, delta)
                break

        # Optional hop-distance filter
        reachable: Optional[Set[str]] = None
        if inp.max_hops is not None:
            reachable = self._bfs_reachable(graph, valid_seeds, inp.max_hops)

        # Rank and filter
        threshold = inp.min_score_threshold
        ranked: List[Tuple[str, float]] = []
        for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if score < threshold:
                continue
            if reachable is not None and node not in reachable:
                continue
            ranked.append((node, score))

        # Collect traversed edges: edges between any two ranked nodes
        ranked_set = {n for n, _ in ranked}
        traversed_edges: List[str] = []
        seen_edge_ids: Set[str] = set()
        for node in ranked_set:
            for neighbour, edge_id, _ in graph.neighbours(node):
                if neighbour in ranked_set and edge_id not in seen_edge_ids:
                    seen_edge_ids.add(edge_id)
                    traversed_edges.append(edge_id)

        logger.info(
            "PPR complete: %d nodes ranked (of %d), %d edges traversed, "
            "converged at iteration %d",
            len(ranked), len(all_nodes), len(traversed_edges), converged_at,
        )

        return PPROutput(
            ranked_nodes=ranked,
            traversed_edges=traversed_edges,
            convergence_iterations=converged_at,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_reachable(
        graph: _AdjacencyGraph,
        seeds: List[str],
        max_hops: int,
    ) -> Set[str]:
        """Return set of node IDs reachable from *seeds* within *max_hops*."""
        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                for neighbour, _, _ in graph.neighbours(node):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        return visited
