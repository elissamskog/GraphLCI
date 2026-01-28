"""
Scope Resolver Module
Location: librarian/resolver/scope_resolver.py
Owner: LIBRARIAN Instance

Responsibility: Determine the complete document scope needed to answer a
query.  Runs PPR traversal to identify the relevant subgraph, then uses
LLM-driven edge interpretation to classify each edge's relevance, generate
reasoning hints, and produce the final ``ScopeObject``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from schemas.contracts import (
    DependencyType,
    DocumentRole,
    DomainPair,
    DomainType,
    EdgeContext,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphState,
    NodeType,
    RouterDecision,
    ScopeMetadata,
    ScopeObject,
    ScopedDocument,
    SupportingDocument,
)
from librarian.resolver.ppr_traversal import PPRInput, PPROutput, PPRTraversal

logger = logging.getLogger(__name__)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ResolverInput:
    """Input to the scope resolver."""
    router_decision: RouterDecision
    graph_state: GraphState
    traversal_config: Dict[str, Any]
    query_text: str = ""


@dataclass
class ResolverOutput:
    """Output from the scope resolver."""
    scope_object: ScopeObject
    traversal_trace: List[str]


# ============================================================================
# LLM CLIENT PROTOCOL
# ============================================================================

class LLMClient(Protocol):
    """Protocol for the underlying LLM call."""
    def generate(self, prompt: str) -> str: ...


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

EDGE_INTERPRETATION_PROMPT = """\
You are a scope resolver determining which documents are required to verify a specific claim.

CLAIM UNDER AUDIT: {query_text}
SEARCH STRATEGY: {intent_classification}
PRIMARY TARGETS: {explicit_docs}

GRAPH TRAVERSAL RESULTS:
The following structural edges connect the target documents:

{edges_found}

TASK: Identify which documents and edges are essential for a grounded audit of this claim.

For each edge, assess:
1. Is this edge relevant to the logical or temporal verification of the claim?
2. What reasoning hint should be passed to the synthesis model?
3. What precedence rule applies to resolve conflicting information?

EDGE TYPE SEMANTICS:
- AMENDS: The information in the source document updates or corrects the target.
- GOVERNS: The source document defines the requirements or rules for the target.
- DEFINES_TERM: The source defines concepts or terminology used in the target.
- INCORPORATES: The target is considered a structural part of the source.
- TEMPORAL_SEQUENCE: Establishes a chronological order between facts or events.

Respond in JSON:
{{
  "relevant_edges": [
    {{
      "edge_id": "...",
      "relevance_score": 0.0-1.0,
      "reasoning_hint": "Chronological precedence: Doc B provides the updated founding date for Entity X.",
      "precedence_rule": "If conflict, Doc B updates Doc A" | null
    }}
  ],
  "supporting_documents": [
    {{
      "doc_id": "...",
      "dependency_type": "DEFINITION | AMENDMENT | REGULATION | REFERENCE",
      "required_sections": ["Section 1.1", "Table 4"] | null
    }}
  ],
  "domain_pair": {{
    "primary_domain": "...",
    "secondary_domain": "..." | null,
    "cross_domain_logic": "Higher-order documentation defines the baseline for synthesis" | null
  }},
  "completeness_confidence": 0.0-1.0,
  "warnings": ["..."]
}}"""


# ============================================================================
# PARSING / MAPPING HELPERS
# ============================================================================

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from an LLM response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return json.loads(text)


def _safe_domain(raw: str) -> DomainType:
    try:
        return DomainType(raw.upper())
    except (ValueError, AttributeError):
        return DomainType.LEGAL


def _safe_dependency_type(raw: str) -> DependencyType:
    try:
        return DependencyType(raw.upper())
    except (ValueError, AttributeError):
        return DependencyType.REFERENCE


def _edge_type_to_dependency(edge_type: EdgeType) -> DependencyType:
    """Map an EdgeType to the closest DependencyType."""
    mapping = {
        EdgeType.DEFINES_TERM: DependencyType.DEFINITION,
        EdgeType.AMENDS: DependencyType.AMENDMENT,
        EdgeType.GOVERNS: DependencyType.REGULATION,
        EdgeType.TEMPORAL_SEQUENCE: DependencyType.AMENDMENT,
    }
    return mapping.get(edge_type, DependencyType.REFERENCE)


# ============================================================================
# SCOPE RESOLVER
# ============================================================================

class ScopeResolver:
    """Determine document scope via PPR traversal and LLM edge interpretation.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    config:
        Dictionary with optional keys:
        - ``damping_factor`` (float, default 0.85)
        - ``max_iterations`` (int, default 100)
        - ``min_score_threshold`` (float, default 0.01)
        - ``max_hops`` (int, default 3)
        - ``min_edge_confidence`` (float, default 0.7)
        - ``max_supporting_documents`` (int, default 5)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or {}
        self._ppr = PPRTraversal(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, inp: ResolverInput) -> ResolverOutput:
        """Run PPR traversal then LLM edge interpretation.

        Flow
        ----
        1. Identify seed nodes from ``RouterDecision.identified_targets``
        2. Run PPR to rank the full graph relative to those seeds
        3. Collect edges between ranked nodes
        4. Send edges to LLM for semantic interpretation
        5. Build and return ``ScopeObject``
        """
        decision = inp.router_decision
        graph = inp.graph_state
        trav_cfg = inp.traversal_config or {}

        # ----- Step 1: Seed nodes -----
        seed_nodes = self._find_seed_nodes(
            decision.identified_targets.explicit_docs,
            decision.identified_targets.implicit_entities,
            graph,
        )

        if not seed_nodes:
            logger.warning("No seed nodes resolved; returning empty scope")
            return self._empty_result(decision)

        # ----- Step 2: PPR traversal -----
        ppr_input = PPRInput(
            seed_nodes=seed_nodes,
            graph_state=graph,
            damping_factor=trav_cfg.get("damping_factor", self._config.get("damping_factor", 0.85)),
            max_iterations=trav_cfg.get("max_iterations", self._config.get("max_iterations", 100)),
            min_score_threshold=trav_cfg.get("min_score_threshold", self._config.get("min_score_threshold", 0.01)),
            max_hops=trav_cfg.get("max_hops", self._config.get("max_hops", 3)),
        )
        ppr_output = self._ppr.traverse(ppr_input)

        # ----- Step 3: Collect candidate edges -----
        ranked_ids = {nid for nid, _ in ppr_output.ranked_nodes}

        # DEBUG: show graph edges vs ranked node IDs
        print(f"\n[DEBUG] ranked_ids from PPR: {ranked_ids}")
        print(f"[DEBUG] graph has {len(graph.edges)} total edges:")
        for e in graph.edges:
            src_in = e.source_node in ranked_ids
            tgt_in = e.target_node in ranked_ids
            print(f"  edge={e.edge_id}  src={e.source_node}(ranked={src_in})  "
                  f"tgt={e.target_node}(ranked={tgt_in})  conf={e.confidence}")

        candidate_edges = [
            e for e in graph.edges
            if e.source_node in ranked_ids and e.target_node in ranked_ids
        ]
        print(f"[DEBUG] candidate_edges (both endpoints ranked): {len(candidate_edges)}")

        min_edge_conf = self._config.get("min_edge_confidence", 0.7)
        confident_edges = [e for e in candidate_edges if e.confidence >= min_edge_conf]
        print(f"[DEBUG] confident_edges (conf >= {min_edge_conf}): {len(confident_edges)}")

        # Build lookup tables
        node_map: Dict[str, GraphNode] = {n.node_id: n for n in graph.nodes}
        score_map: Dict[str, float] = dict(ppr_output.ranked_nodes)

        # DEBUG: show all node_ids in the graph for cross-reference
        print(f"[DEBUG] graph node_ids: {set(node_map.keys())}")

        # ----- Step 4: LLM edge interpretation -----
        interpretation = self._interpret_edges(
            decision=decision,
            edges=confident_edges,
            node_map=node_map,
            query_text=inp.query_text,
        )

        # DEBUG: show interpretation result (edge IDs from LLM vs edge_map)
        interp_edge_ids = [ie["edge_id"] for ie in interpretation.get("relevant_edges", [])]
        graph_edge_ids = [e.edge_id for e in confident_edges]
        print(f"[DEBUG] interpretation returned {len(interp_edge_ids)} relevant edges: {interp_edge_ids}")
        print(f"[DEBUG] confident_edges edge_ids: {graph_edge_ids}")
        mismatched = set(interp_edge_ids) - set(graph_edge_ids)
        if mismatched:
            print(f"[DEBUG] *** MISMATCH: interpreted edge IDs not in graph: {mismatched}")

        # ----- Step 5: Build ScopeObject -----
        scope_object = self._build_scope_object(
            decision=decision,
            ppr_output=ppr_output,
            interpretation=interpretation,
            confident_edges=confident_edges,
            node_map=node_map,
            score_map=score_map,
        )

        # DEBUG [Request #1]: Check if edge_context is populated after scope resolution
        print(f"\n[DEBUG] ScopeObject.edge_context has {len(scope_object.edge_context)} entries:")
        for ec in scope_object.edge_context:
            print(f"  {ec.edge_id}: {ec.source_doc} --{ec.edge_type.value}--> {ec.target_doc}")

        traversal_trace = [
            f"seed={seed_nodes}",
            f"ppr_ranked={len(ppr_output.ranked_nodes)}",
            f"edges_candidate={len(candidate_edges)}",
            f"edges_confident={len(confident_edges)}",
            f"converged_iter={ppr_output.convergence_iterations}",
        ]

        return ResolverOutput(
            scope_object=scope_object,
            traversal_trace=traversal_trace,
        )

    # ------------------------------------------------------------------
    # Seed node resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _find_seed_nodes(
        explicit_docs: List[str],
        implicit_entities: List[str],
        graph: GraphState,
    ) -> List[str]:
        """Map explicit doc IDs and implicit entity names to graph node IDs."""
        node_ids: List[str] = []
        seen: Set[str] = set()

        # Direct match on doc_id for DOCUMENT nodes
        for node in graph.nodes:
            if node.node_type == NodeType.DOCUMENT and node.doc_id in explicit_docs:
                if node.node_id not in seen:
                    node_ids.append(node.node_id)
                    seen.add(node.node_id)

        # Fuzzy match on content_summary / defined_terms for implicit entities
        if implicit_entities:
            lower_entities = {e.lower() for e in implicit_entities}
            for node in graph.nodes:
                if node.node_id in seen:
                    continue
                summary_lower = node.content_summary.lower()
                terms_lower = {t.lower() for t in node.defined_terms}
                for entity in lower_entities:
                    if entity in summary_lower or entity in terms_lower:
                        node_ids.append(node.node_id)
                        seen.add(node.node_id)
                        break

        return node_ids

    # ------------------------------------------------------------------
    # LLM edge interpretation
    # ------------------------------------------------------------------

    def _interpret_edges(
        self,
        decision: RouterDecision,
        edges: List[GraphEdge],
        node_map: Dict[str, GraphNode],
        query_text: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to semantically interpret the candidate edges."""
        if not edges:
            return {
                "relevant_edges": [],
                "supporting_documents": [],
                "domain_pair": {
                    "primary_domain": decision.domain_expectation.value,
                    "secondary_domain": None,
                    "cross_domain_logic": None,
                },
                "completeness_confidence": 0.0,
                "warnings": ["No edges found for interpretation"],
            }

        edges_description = self._format_edges_for_prompt(edges, node_map)

        prompt = EDGE_INTERPRETATION_PROMPT.format(
            query_text=query_text,
            intent_classification=decision.search_strategy.classification.value,
            explicit_docs=", ".join(decision.identified_targets.explicit_docs),
            edges_found=edges_description,
        )

        raw_response = self._llm.generate(prompt)

        try:
            return _parse_json_response(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse edge interpretation response: %s", exc)
            return self._fallback_interpretation(decision, edges, node_map)

    @staticmethod
    def _format_edges_for_prompt(
        edges: List[GraphEdge],
        node_map: Dict[str, GraphNode],
    ) -> str:
        """Render edges as a human-readable block for the prompt."""
        lines: List[str] = []
        for e in edges:
            src = node_map.get(e.source_node)
            tgt = node_map.get(e.target_node)
            src_label = f"{src.doc_id} ({src.content_summary[:60]})" if src else e.source_node
            tgt_label = f"{tgt.doc_id} ({tgt.content_summary[:60]})" if tgt else e.target_node
            lines.append(
                f"- [{e.edge_id}] {src_label} --{e.edge_type.value}--> "
                f"{tgt_label}  (confidence={e.confidence:.2f}, "
                f"hint=\"{e.reasoning_hint}\")"
            )
        return "\n".join(lines)

    def _fallback_interpretation(
        self,
        decision: RouterDecision,
        edges: List[GraphEdge],
        node_map: Dict[str, GraphNode],
    ) -> Dict[str, Any]:
        """Deterministic fallback when the LLM call fails."""
        relevant = []
        supporting_docs: Dict[str, Dict[str, Any]] = {}

        for e in edges:
            relevant.append({
                "edge_id": e.edge_id,
                "relevance_score": e.confidence,
                "reasoning_hint": e.reasoning_hint,
                "precedence_rule": None,
            })
            # Identify supporting doc from target side of structural edges
            dep_type = _edge_type_to_dependency(e.edge_type)
            tgt_node = node_map.get(e.target_node)
            if tgt_node and tgt_node.doc_id not in supporting_docs:
                supporting_docs[tgt_node.doc_id] = {
                    "doc_id": tgt_node.doc_id,
                    "dependency_type": dep_type.value,
                    "required_sections": None,
                }

        return {
            "relevant_edges": relevant,
            "supporting_documents": list(supporting_docs.values()),
            "domain_pair": {
                "primary_domain": decision.domain_expectation.value,
                "secondary_domain": None,
                "cross_domain_logic": None,
            },
            "completeness_confidence": 0.5,
            "warnings": ["Used deterministic fallback — LLM parse failed"],
        }

    # ------------------------------------------------------------------
    # ScopeObject assembly
    # ------------------------------------------------------------------

    def _build_scope_object(
        self,
        decision: RouterDecision,
        ppr_output: PPROutput,
        interpretation: Dict[str, Any],
        confident_edges: List[GraphEdge],
        node_map: Dict[str, GraphNode],
        score_map: Dict[str, float],
    ) -> ScopeObject:
        """Assemble the final ScopeObject from PPR + interpretation."""

        # --- Primary documents ---
        explicit_set = set(decision.identified_targets.explicit_docs)
        primary_documents: List[ScopedDocument] = []
        seen_doc_ids: Set[str] = set()

        # Explicit targets first
        for node_id, score in ppr_output.ranked_nodes:
            node = node_map.get(node_id)
            if node and node.node_type == NodeType.DOCUMENT and node.doc_id in explicit_set:
                if node.doc_id not in seen_doc_ids:
                    role = DocumentRole.SOURCE if len(primary_documents) == 0 else DocumentRole.TARGET
                    primary_documents.append(
                        ScopedDocument(
                            doc_id=node.doc_id,
                            relevance_score=score,
                            role=role,
                        )
                    )
                    seen_doc_ids.add(node.doc_id)

        # High-scoring non-explicit document nodes as CONTEXT
        for node_id, score in ppr_output.ranked_nodes:
            node = node_map.get(node_id)
            if (
                node
                and node.node_type == NodeType.DOCUMENT
                and node.doc_id not in seen_doc_ids
            ):
                primary_documents.append(
                    ScopedDocument(
                        doc_id=node.doc_id,
                        relevance_score=score,
                        role=DocumentRole.CONTEXT,
                    )
                )
                seen_doc_ids.add(node.doc_id)

        # --- Supporting documents from interpretation ---
        max_supporting = self._config.get("max_supporting_documents", 5)
        supporting_documents: List[SupportingDocument] = []
        for sd in interpretation.get("supporting_documents", [])[:max_supporting]:
            supporting_documents.append(
                SupportingDocument(
                    doc_id=sd["doc_id"],
                    dependency_type=_safe_dependency_type(sd.get("dependency_type", "REFERENCE")),
                    required_sections=sd.get("required_sections"),
                )
            )

        # --- Edge context ---
        edge_map: Dict[str, GraphEdge] = {e.edge_id: e for e in confident_edges}
        interpreted_edges = {
            ie["edge_id"]: ie for ie in interpretation.get("relevant_edges", [])
        }

        edge_context: List[EdgeContext] = []
        for edge_id, ie in interpreted_edges.items():
            edge = edge_map.get(edge_id)
            if not edge:
                continue
            src_node = node_map.get(edge.source_node)
            tgt_node = node_map.get(edge.target_node)
            edge_context.append(
                EdgeContext(
                    edge_id=edge_id,
                    edge_type=edge.edge_type,
                    source_doc=src_node.doc_id if src_node else edge.source_node,
                    target_doc=tgt_node.doc_id if tgt_node else edge.target_node,
                    reasoning_hint=ie.get("reasoning_hint", edge.reasoning_hint),
                    precedence_rule=ie.get("precedence_rule"),
                )
            )

        # --- Domain pair ---
        dp = interpretation.get("domain_pair", {})
        primary_dom = _safe_domain(dp.get("primary_domain", decision.domain_expectation.value))
        secondary_raw = dp.get("secondary_domain")
        secondary_dom = _safe_domain(secondary_raw) if secondary_raw else None
        if secondary_dom and secondary_dom != primary_dom:
            effective_primary = DomainType.CROSS_DOMAIN
        else:
            effective_primary = primary_dom

        domain_pair = DomainPair(
            primary_domain=effective_primary,
            secondary_domain=secondary_dom,
            cross_domain_logic=dp.get("cross_domain_logic"),
        )

        # --- Scope metadata ---
        # Rough token estimate: 500 tokens per document in scope
        token_estimate = (len(primary_documents) + len(supporting_documents)) * 500
        scope_metadata = ScopeMetadata(
            total_token_estimate=token_estimate,
            traversal_depth=ppr_output.convergence_iterations,
            edges_traversed=len(ppr_output.traversed_edges),
            completeness_confidence=float(interpretation.get("completeness_confidence", 0.0)),
        )

        warnings = interpretation.get("warnings", [])

        return ScopeObject(
            query_id=decision.query_id,
            primary_documents=primary_documents,
            supporting_documents=supporting_documents,
            edge_context=edge_context,
            domain_pair=domain_pair,
            scope_metadata=scope_metadata,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Empty-result helper
    # ------------------------------------------------------------------

    def _empty_result(self, decision: RouterDecision) -> ResolverOutput:
        """Return a minimal ResolverOutput when no seeds are found."""
        return ResolverOutput(
            scope_object=ScopeObject(
                query_id=decision.query_id,
                primary_documents=[],
                supporting_documents=[],
                edge_context=[],
                domain_pair=DomainPair(
                    primary_domain=decision.domain_expectation,
                    secondary_domain=None,
                    cross_domain_logic=None,
                ),
                scope_metadata=ScopeMetadata(
                    total_token_estimate=0,
                    traversal_depth=0,
                    edges_traversed=0,
                    completeness_confidence=0.0,
                ),
                warnings=["No seed nodes found in graph"],
            ),
            traversal_trace=["seed=[] — aborted"],
        )
