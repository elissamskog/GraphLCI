"""
Context Builder Module
Location: librarian/assembler/context_builder.py
Owner: LIBRARIAN Instance

Responsibility: Assemble the final LCI payload for downstream reasoning.
Enforces graph-aware pruning (2-hop limit) and token budget management.
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from schemas.contracts import (
    AliasedDocument,
    AliasMap,
    DependencyType,
    DocumentRole,
    DomainPair,
    DomainType,
    EdgeContext,
    EdgeType,
    LCIPayload,
    LCIPreamble,
    OrderedDocument,
    RawDocument,
    ScopeObject,
)

logger = logging.getLogger(__name__)

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ContextBuildInput:
    """Input to the context builder."""
    scope_object: ScopeObject
    raw_documents: List[RawDocument]
    decontextualized_contents: Dict[str, str]   # doc_id -> processed text
    max_context_tokens: int


@dataclass
class ContextBuildOutput:
    """Output of the context builder."""
    lci_payload: LCIPayload
    assembly_warnings: List[str]
    alias_map: Optional[AliasMap] = None  # For post-inference metadata join


def extract_precedence_rules(edge_context: List[EdgeContext]) -> List[str]:
    """Collect all non-null precedence rules from edge context."""
    return [
        edge.precedence_rule
        for edge in edge_context
        if edge.precedence_rule
    ]

# ============================================================================
# GRAPH-AWARE PRUNING
# ============================================================================

def get_reachable_docs(scope: ScopeObject, max_hops: int = 2) -> Set[str]:
    """Identify documents within N hops of primary audit targets."""
    # Start with primary targets and sources
    reachable = {d.doc_id for d in scope.primary_documents}
    current_layer = set(reachable)

    for i in range(max_hops):
        next_layer = set()
        for edge in scope.edge_context:
            if edge.source_doc in current_layer and edge.target_doc not in reachable:
                next_layer.add(edge.target_doc)
            elif edge.target_doc in current_layer and edge.source_doc not in reachable:
                next_layer.add(edge.source_doc)
        
        if not next_layer:
            break
        reachable.update(next_layer)
        current_layer = next_layer
        
    return reachable

# ============================================================================
# DOCUMENT ORDERING
# ============================================================================

def order_documents(scope: ScopeObject, allowed_ids: Set[str]) -> List[str]:
    """Order reachable documents for optimal LCI injection."""
    ordered: List[str] = []
    seen: Set[str] = set()

    def _add(doc_id: str) -> None:
        if doc_id in allowed_ids and doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)

    # 1. Definitions/Foundational docs first
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.DEFINITION:
            _add(doc.doc_id)

    # 2. Governing/Policy docs second
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.REGULATION:
            _add(doc.doc_id)

    # 3. Primary targets
    for doc in scope.primary_documents:
        if doc.role in (DocumentRole.SOURCE, DocumentRole.TARGET):
            _add(doc.doc_id)

    # 4. Contextual evidence
    for doc in scope.primary_documents:
        if doc.role == DocumentRole.CONTEXT:
            _add(doc.doc_id)

    # 5. General References
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.REFERENCE:
            _add(doc.doc_id)

    # 6. Updates/Amendments last
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.AMENDMENT:
            _add(doc.doc_id)

    return ordered

# ============================================================================
# NEUTRAL EDGE GRAMMAR
# ============================================================================

def generate_edge_grammar(edge_context: List[EdgeContext]) -> str:
    """Explains structural relationships using neutral auditing terms."""
    if not edge_context:
        return ""

    grammar_lines: List[str] = [
        "DOCUMENT RELATIONSHIP RULES:",
        "Apply the following structural logic during synthesis:",
        "",
    ]

    for edge in edge_context:
        if edge.edge_type == EdgeType.GOVERNS:
            grammar_lines.append(
                f"- {edge.source_doc} GOVERNS {edge.target_doc}: "
                f"{edge.source_doc} establishes the baseline requirements."
            )
        elif edge.edge_type == EdgeType.AMENDS:
            grammar_lines.append(
                f"- {edge.source_doc} AMENDS {edge.target_doc}: "
                f"Where they conflict, the information in {edge.source_doc} is the most recent."
            )
        elif edge.edge_type == EdgeType.DEFINES_TERM:
            grammar_lines.append(
                f"- {edge.source_doc} DEFINES terms used in {edge.target_doc}: "
                f"Interpret terms according to their specific definitions in {edge.source_doc}."
            )
        elif edge.edge_type == EdgeType.INCORPORATES:
            grammar_lines.append(
                f"- {edge.source_doc} INCORPORATES {edge.target_doc}: "
                f"The referenced document is formally part of the primary record."
            )
        elif edge.edge_type == EdgeType.TEMPORAL_SEQUENCE:
            grammar_lines.append(
                f"- {edge.source_doc} precedes {edge.target_doc} in "
                f"TEMPORAL SEQUENCE: Use this to establish chronology."
            )

    grammar_lines.append("\nResolve contradictions based on these structural rules.")
    return "\n".join(grammar_lines)

# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._default_max_tokens = self._config.get("max_context_tokens", 200_000)
        self._max_hops = self._config.get("max_graph_hops", 2)
        self._max_top_k = self._config.get("max_top_k")  # v5

    def build(self, inp: ContextBuildInput) -> ContextBuildOutput:
        scope = inp.scope_object
        max_tokens = inp.max_context_tokens or self._default_max_tokens
        warnings: List[str] = []

        # 1. Pruning: Only keep docs within 2 hops of the query targets
        reachable_ids = get_reachable_docs(scope, max_hops=self._max_hops)
        
        # 2. Preamble (Neutral)
        edge_grammar = generate_edge_grammar(scope.edge_context)
        preamble = LCIPreamble(
            edge_grammar=edge_grammar,
            domain_logic="", # Simplified for general auditing
            precedence_rules=extract_precedence_rules(scope.edge_context),
        )
        preamble_tokens = _estimate_tokens(edge_grammar)

        # 3. Rank-based Pruning (Top-K)
        all_candidates: List[Tuple[str, float]] = []
        for d in scope.primary_documents:
            all_candidates.append((d.doc_id, d.relevance_score))
        for d in scope.supporting_documents:
            all_candidates.append((d.doc_id, d.relevance_score))

        # Filter by reachability
        filtered_candidates = [
            (doc_id, score) for doc_id, score in all_candidates
            if doc_id in reachable_ids
        ]

        # Sort by score descending
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply Top-K ceiling
        max_k = self._max_top_k
        if max_k is not None and len(filtered_candidates) > max_k:
            warnings.append(f"Top-K ceiling applied: kept {max_k} of {len(filtered_candidates)} docs")
            filtered_candidates = filtered_candidates[:max_k]

        allowed_ids = {doc_id for doc_id, _ in filtered_candidates}

        # 4. Order documents for optimal Injection
        ordered_ids = order_documents(scope, allowed_ids)

        # 5. Assembly with UID generation and alias mapping
        documents: List[OrderedDocument] = []
        alias_mappings: Dict[str, AliasedDocument] = {}
        running_tokens = preamble_tokens
        
        for doc_id in ordered_ids:
            content = inp.decontextualized_contents.get(doc_id)
            if not content: continue

            doc_tokens = _estimate_tokens(content)
            if running_tokens + doc_tokens > max_tokens:
                warnings.append(f"Token limit reached at {doc_id}")
                break

            # Generate immutable UID and alias
            doc_uid = uuid.uuid4().hex
            alias = f"[D{len(documents) + 1}]"

            # Resolve role
            role_map = {d.doc_id: d.role.value for d in scope.primary_documents}
            role = role_map.get(doc_id, "CONTEXT")

            documents.append(OrderedDocument(
                doc_id=doc_id,
                role=role,
                content=content,
                injection_order=len(documents),
                uid=doc_uid,
            ))
            
            # Build alias mapping for post-inference join
            raw_doc = next((d for d in inp.raw_documents if d.doc_id == doc_id), None)
            source_url = raw_doc.metadata.get("source_url") if raw_doc else None
            alias_mappings[alias] = AliasedDocument(
                alias=alias,
                uid=doc_uid,
                doc_id=doc_id,
                chunk_id=None,  # Document-level, not chunk
                source_url=source_url,
            )
            running_tokens += doc_tokens

        alias_map = AliasMap(mappings=alias_mappings) if alias_mappings else None

        return ContextBuildOutput(
            lci_payload=LCIPayload(preamble=preamble, documents=documents, total_tokens=running_tokens),
            assembly_warnings=warnings,
            alias_map=alias_map,
        )