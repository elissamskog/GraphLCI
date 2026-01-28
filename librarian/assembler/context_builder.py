"""
Context Builder Module
Location: librarian/assembler/context_builder.py
Owner: LIBRARIAN Instance

Responsibility: Assemble the final LCI payload for downstream reasoning.
Handles document ordering (definitions before usage, regulations before
contracts, amendments last), edge grammar generation, and token budget
management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from schemas.contracts import (
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


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


# ============================================================================
# DOCUMENT ORDERING
# ============================================================================

def order_documents(scope: ScopeObject) -> List[str]:
    """Order documents for optimal LCI injection.

    Priority
    --------
    1. Definition documents (``DEFINES_TERM`` edges) — must come first
    2. Regulatory documents (``GOVERNS`` edges) — establish floor
    3. Primary source documents
    4. Primary target documents
    5. Amendment documents (``AMENDS`` edges) — apply last
    """
    ordered: List[str] = []
    seen: Set[str] = set()

    def _add(doc_id: str) -> None:
        if doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)

    # 1. Definitions first
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.DEFINITION:
            _add(doc.doc_id)

    # 2. Regulations second
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.REGULATION:
            _add(doc.doc_id)

    # 3. Primary source and target documents
    for doc in scope.primary_documents:
        if doc.role in (DocumentRole.SOURCE, DocumentRole.TARGET):
            _add(doc.doc_id)

    # 4. Context documents
    for doc in scope.primary_documents:
        if doc.role == DocumentRole.CONTEXT:
            _add(doc.doc_id)

    # 5. Amendments last
    for doc in scope.supporting_documents:
        if doc.dependency_type == DependencyType.AMENDMENT:
            _add(doc.doc_id)

    # 6. Any remaining supporting docs (REFERENCE, etc.)
    for doc in scope.supporting_documents:
        _add(doc.doc_id)

    return ordered


# ============================================================================
# EDGE GRAMMAR GENERATION
# ============================================================================

def generate_edge_grammar(edge_context: List[EdgeContext]) -> str:
    """Generate a natural language preamble explaining edge semantics.

    The resulting text is injected ahead of the documents in the LCI
    payload so the downstream model understands the structural
    relationships before it reads the content.
    """
    # DEBUG [Request #2]: Confirm generate_edge_grammar() is being called
    print(f"\n[DEBUG] generate_edge_grammar() called")
    # DEBUG [Request #3]: Print edge_context list before grammar generation
    print(f"[DEBUG] edge_context has {len(edge_context)} entries:")
    for ec in edge_context:
        print(f"  {ec.edge_id}: {ec.source_doc} --{ec.edge_type.value}--> {ec.target_doc}"
              f"  hint={ec.reasoning_hint!r}")
    if not edge_context:
        print("[DEBUG] edge_context is EMPTY — returning empty grammar")
        return ""

    grammar_lines: List[str] = [
        "DOCUMENT RELATIONSHIP RULES:",
        "When analyzing these documents, apply the following logic:",
        "",
    ]

    for edge in edge_context:
        if edge.edge_type == EdgeType.GOVERNS:
            grammar_lines.append(
                f"- {edge.source_doc} GOVERNS {edge.target_doc}: "
                f"Regulatory requirements set a floor. {edge.target_doc} can be "
                f"stricter but not more permissive."
            )
        elif edge.edge_type == EdgeType.AMENDS:
            grammar_lines.append(
                f"- {edge.source_doc} AMENDS {edge.target_doc}: "
                f"Where they conflict, {edge.source_doc} supersedes."
            )
        elif edge.edge_type == EdgeType.DEFINES_TERM:
            grammar_lines.append(
                f"- {edge.source_doc} DEFINES terms used in {edge.target_doc}: "
                f"Interpret terms according to their defined meanings."
            )
        elif edge.edge_type == EdgeType.INCORPORATES:
            grammar_lines.append(
                f"- {edge.source_doc} INCORPORATES {edge.target_doc}: "
                f"The referenced document is legally part of the primary document."
            )
        elif edge.edge_type == EdgeType.CROSS_REFERENCES:
            grammar_lines.append(
                f"- {edge.source_doc} CROSS-REFERENCES {edge.target_doc}: "
                f"Explicit citation exists between these documents."
            )
        elif edge.edge_type == EdgeType.SAME_TRANSACTION:
            grammar_lines.append(
                f"- {edge.source_doc} and {edge.target_doc} are part of the "
                f"SAME TRANSACTION: Interpret them as a unified agreement."
            )
        elif edge.edge_type == EdgeType.TEMPORAL_SEQUENCE:
            grammar_lines.append(
                f"- {edge.source_doc} precedes {edge.target_doc} in "
                f"TEMPORAL SEQUENCE: Check for implicit supersession."
            )
        elif edge.edge_type == EdgeType.CONTAINS:
            # Structural parent→child edge — enables graph traversal only,
            # no reasoning hint needed in the LCI preamble.
            continue

        # Append precedence rule if present
        if edge.precedence_rule:
            grammar_lines.append(f"  Precedence: {edge.precedence_rule}")

    grammar_lines.append("")
    grammar_lines.append(
        "Apply these rules when identifying contradictions or conflicts."
    )

    return "\n".join(grammar_lines)


# ============================================================================
# DOMAIN LOGIC GENERATION
# ============================================================================

def generate_domain_logic(domain_pair: DomainPair) -> str:
    """Generate domain-specific reasoning instructions for the preamble."""
    lines: List[str] = ["DOMAIN CONTEXT:"]

    primary = domain_pair.primary_domain
    secondary = domain_pair.secondary_domain

    domain_descriptions = {
        DomainType.LEGAL: "These are legal documents (contracts, agreements). "
                          "Apply contractual interpretation principles.",
        DomainType.REGULATORY: "These are regulatory documents. "
                               "Compliance requirements are mandatory minimums.",
        DomainType.SCIENTIFIC: "These are scientific/technical documents. "
                               "Focus on methodological accuracy.",
        DomainType.FINANCIAL: "These are financial documents. "
                              "Numerical precision is critical.",
        DomainType.CROSS_DOMAIN: "These documents span multiple domains. "
                                  "Apply the stricter standard where domains conflict.",
    }

    lines.append(domain_descriptions.get(
        primary,
        f"Primary domain: {primary.value}.",
    ))

    if secondary and secondary != primary:
        lines.append(
            f"Secondary domain: {secondary.value}. "
            f"Cross-domain analysis required."
        )

    if domain_pair.cross_domain_logic:
        lines.append(f"Cross-domain rule: {domain_pair.cross_domain_logic}")

    return "\n".join(lines)


# ============================================================================
# PRECEDENCE RULES EXTRACTION
# ============================================================================

def extract_precedence_rules(edge_context: List[EdgeContext]) -> List[str]:
    """Collect all non-null precedence rules from edge context."""
    return [
        edge.precedence_rule
        for edge in edge_context
        if edge.precedence_rule
    ]


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """Assemble the final LCI payload from resolved scope and documents.

    Parameters
    ----------
    config:
        Optional dictionary with keys:
        - ``max_context_tokens`` (int, default 200_000)
        - ``include_edge_preamble`` (bool, default True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._default_max_tokens = self._config.get("max_context_tokens", 200_000)
        self._include_preamble = self._config.get("include_edge_preamble", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, inp: ContextBuildInput) -> ContextBuildOutput:
        """Build the LCI payload.

        Flow
        ----
        1. Generate preamble (edge grammar + domain logic)
        2. Order documents according to dependency priority
        3. Assemble ``OrderedDocument`` list within token budget
        4. Return ``LCIPayload`` and any warnings
        """
        scope = inp.scope_object
        max_tokens = inp.max_context_tokens or self._default_max_tokens
        warnings: List[str] = []

        # --- 1. Preamble ---
        if self._include_preamble:
            edge_grammar = generate_edge_grammar(scope.edge_context)
            domain_logic = generate_domain_logic(scope.domain_pair)
            precedence_rules = extract_precedence_rules(scope.edge_context)
        else:
            edge_grammar = ""
            domain_logic = ""
            precedence_rules = []

        preamble = LCIPreamble(
            edge_grammar=edge_grammar,
            domain_logic=domain_logic,
            precedence_rules=precedence_rules,
        )
        preamble_tokens = _estimate_tokens(edge_grammar + domain_logic)

        # --- 2. Order documents ---
        ordered_ids = order_documents(scope)

        # Build lookup: doc_id -> content (prefer decontextualized)
        raw_lookup: Dict[str, str] = {d.doc_id: d.content for d in inp.raw_documents}
        content_lookup: Dict[str, str] = {}
        for doc_id in ordered_ids:
            content_lookup[doc_id] = inp.decontextualized_contents.get(
                doc_id, raw_lookup.get(doc_id, "")
            )

        # Build role lookup from scope
        role_lookup: Dict[str, str] = {}
        for doc in scope.primary_documents:
            role_lookup[doc.doc_id] = doc.role.value
        for doc in scope.supporting_documents:
            if doc.doc_id not in role_lookup:
                role_lookup[doc.doc_id] = doc.dependency_type.value

        # --- 3. Assemble within token budget ---
        documents: List[OrderedDocument] = []
        running_tokens = preamble_tokens
        injection_order = 0

        for doc_id in ordered_ids:
            content = content_lookup.get(doc_id)
            if content is None:
                warnings.append(f"Document {doc_id} not found in raw documents; skipped")
                continue

            doc_tokens = _estimate_tokens(content)

            if running_tokens + doc_tokens > max_tokens:
                warnings.append(
                    f"Document {doc_id} excluded: would exceed token budget "
                    f"({running_tokens + doc_tokens} > {max_tokens})"
                )
                continue

            documents.append(
                OrderedDocument(
                    doc_id=doc_id,
                    role=role_lookup.get(doc_id, "CONTEXT"),
                    content=content,
                    injection_order=injection_order,
                )
            )
            running_tokens += doc_tokens
            injection_order += 1

        lci_payload = LCIPayload(
            preamble=preamble,
            documents=documents,
            total_tokens=running_tokens,
        )

        logger.info(
            "Context built: %d documents, %d tokens (budget %d), %d warnings",
            len(documents), running_tokens, max_tokens, len(warnings),
        )

        return ContextBuildOutput(
            lci_payload=lci_payload,
            assembly_warnings=warnings,
        )
