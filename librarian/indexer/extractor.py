"""
Entity and Relationship Extractor
Location: librarian/indexer/extractor.py
Owner: LIBRARIAN Instance

Responsibility: Extract nodes and edges from documents using LLM-driven
extraction with a gleaning (self-reflection) loop to maximise recall.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from schemas.contracts import (
    DiscoveredDomain,
    DocumentSummary,
    DomainTaxonomy,
    DomainType,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RawDocument,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ExtractionInput:
    """Input for a single document extraction pass."""
    document: RawDocument
    extraction_config: Dict[str, Any]
    corpus_context: Optional[List[Dict[str, str]]] = None
    taxonomy: Optional[DomainTaxonomy] = None
    document_summary: Optional[DocumentSummary] = None


@dataclass
class GleaningResult:
    """Output of a single gleaning (self-reflection) pass."""
    additional_nodes: List[GraphNode]
    additional_edges: List[GraphEdge]
    corrections: List[Dict[str, str]]
    confidence_complete: float


@dataclass
class ExtractionOutput:
    """Full extraction result for one document."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    gleaning_iterations: int
    missed_entities: List[str]


# ============================================================================
# LLM CLIENT PROTOCOL
# ============================================================================

class LLMClient(Protocol):
    """Protocol for the underlying LLM call.

    Any object that exposes a ``generate`` method returning the raw text
    response can be injected into the ``Extractor``.
    """

    def generate(self, prompt: str) -> str: ...


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

INITIAL_EXTRACTION_PROMPT = """\
You are an analytical auditing engine extracting structured evidence from documents.

DOCUMENT ID: {doc_id}
DOCUMENT CONTENT:
{content}
{corpus_context_section}
{document_analysis_section}

TASK: Extract entities and relationships following this taxonomy.

NODE TYPES:
- DOCUMENT: The source material being audited.
- ENTITY: Subjects, organizations, persons, or specific objects (e.g., "Arthur's Magazine", "FIFA").
- DEFINITION: Specific meanings assigned to terms or concepts within the text.
- CLAIM: Specific assertions, factual statements, or findings (e.g., "Founded in 1844").

EDGE TYPES:
- AMENDS: This document updates, corrects, or changes information in another.
- GOVERNS: A policy, rule, or higher-order document applies to this one.
- DEFINES_TERM: A definition for a term or concept used elsewhere in the corpus.
- INCORPORATES: This document includes content from another by reference.
- CROSS_REFERENCES: An explicit citation or link to another document/section.
- SAME_TRANSACTION: Related documents belonging to the same event or logical group.
- TEMPORAL_SEQUENCE: Establishes a chronological order (e.g., versioning or event history).

{domain_classification_section}

Respond in JSON:
{{
  "document_node": {{
    "node_id": "...",
    "domain": "{domain_values}",
    "content_summary": "...",
    "defined_terms": ["term1", "term2"]
  }},
  "entities": [
    {{"node_id": "...", "node_type": "ENTITY", "content_summary": "..."}}
  ],
  "definitions": [
    {{"node_id": "...", "term": "...", "definition_text": "..."}}
  ],
  "claims": [
    {{"node_id": "...", "claim_type": "...", "content_summary": "..."}}
  ],
  "edges": [
    {{
      "edge_id": "...",
      "source_node": "...",
      "target_node": "...",
      "edge_type": "AMENDS | GOVERNS | DEFINES_TERM | INCORPORATES | CROSS_REFERENCES | SAME_TRANSACTION | TEMPORAL_SEQUENCE",
      "confidence": 0.0-1.0,
      "reasoning_hint": "Chronological link: Doc B provides the founding date for Entity A mentioned in Doc A."
    }}
  ]
}}"""

GLEANING_PROMPT = """\
You previously extracted entities and claims from this document.

DOCUMENT CONTENT:
{content}

YOUR PREVIOUS EXTRACTION:
{previous_extraction}

TASK: Review your extraction for missed entities, claims, or relationships.

Common misses:
1. Implicit subjects (pronouns or titles referring to previously mentioned entities).
2. Indirect references to external sources or documents.
3. Defined concepts used in the text but not captured as definitions.
4. Temporal links or versioning dependencies between facts.
5. Structural relationships (e.g., one statement providing the basis for another).

Respond in JSON:
{{
  "additional_entities": [...],
  "additional_claims": [...],
  "additional_edges": [...],
  "corrections": [
    {{"original": "...", "corrected": "...", "reason": "..."}}
  ],
  "confidence_complete": 0.0-1.0
}}"""


def _format_corpus_context(corpus_context: Optional[List[Dict[str, str]]]) -> str:
    """Format corpus context for injection into the extraction prompt."""
    if not corpus_context:
        return ""
    lines = [
        "\nCORPUS CONTEXT (other documents in this corpus — "
        "use their doc_id values as source_node/target_node for "
        "cross-document edges like AMENDS, GOVERNS, SAME_TRANSACTION):"
    ]
    for entry in corpus_context:
        lines.append(f"- {entry['doc_id']}: {entry.get('summary', '(no summary)')}")
        doc_type = entry.get("document_type")
        if doc_type:
            lines.append(f"  Type: {doc_type}")
        key_themes = entry.get("key_themes")
        if key_themes:
            lines.append(f"  Key themes: {', '.join(key_themes)}")
    lines.append("")
    return "\n".join(lines)


def _format_domain_classification(
    taxonomy: Optional[DomainTaxonomy],
) -> tuple[str, str]:
    """Build the domain classification section and value list for the prompt.

    Returns ``(domain_section, domain_values)`` where *domain_section* is
    the full block of text and *domain_values* is the pipe-separated list
    used inside the JSON example.
    """
    if taxonomy is not None and taxonomy.domains:
        section_lines = ["DOMAIN CLASSIFICATION:"]
        names: List[str] = []
        for d in taxonomy.domains:
            section_lines.append(f"- {d.name}: {d.description}")
            names.append(d.name)
        return "\n".join(section_lines), " | ".join(names)

    # Fallback: hardcoded defaults (matches original prompt)
    section = (
        "DOMAIN CLASSIFICATION:\n"
        "- LEGAL: Contracts, agreements, terms\n"
        "- REGULATORY: Laws, regulations, compliance requirements\n"
        "- SCIENTIFIC: Protocols, studies, technical specs\n"
        "- FINANCIAL: Reports, statements, valuations"
    )
    return section, "LEGAL | REGULATORY | SCIENTIFIC | FINANCIAL"


def _format_document_analysis(
    document_summary: Optional[DocumentSummary],
) -> str:
    """Build a DOCUMENT ANALYSIS preamble from an EXPLORER summary.

    Returns an empty string when no summary is available so the prompt
    stays unchanged for the legacy (no-exploration) path.
    """
    if document_summary is None:
        return ""
    lines = ["DOCUMENT ANALYSIS (from corpus exploration):"]
    if document_summary.document_type:
        lines.append(f"- Document type: {document_summary.document_type}")
    if document_summary.apparent_domain:
        lines.append(f"- Apparent domain: {document_summary.apparent_domain}")
    if document_summary.key_themes:
        lines.append(f"- Key themes: {', '.join(document_summary.key_themes)}")
    if document_summary.key_terms:
        lines.append(f"- Key terms: {', '.join(document_summary.key_terms)}")
    if document_summary.summary:
        lines.append(f"- Summary: {document_summary.summary}")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# PARSING HELPERS
# ============================================================================

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from an LLM response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` wrapper
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return json.loads(text)


def _safe_node_type(raw: str) -> NodeType:
    """Map a raw string to NodeType, defaulting to ENTITY."""
    try:
        return NodeType(raw.upper())
    except (ValueError, AttributeError):
        return NodeType.ENTITY


def _safe_edge_type(raw: str) -> EdgeType:
    """Map a raw string to EdgeType, defaulting to CROSS_REFERENCES."""
    try:
        return EdgeType(raw.upper())
    except (ValueError, AttributeError):
        return EdgeType.CROSS_REFERENCES


def _safe_domain(raw: str) -> DomainType:
    """Map a raw string to DomainType, defaulting to LEGAL."""
    try:
        return DomainType(raw.upper())
    except (ValueError, AttributeError):
        return DomainType.LEGAL


def _make_id(prefix: str) -> str:
    """Generate a short unique id with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ============================================================================
# EXTRACTION RESULT BUILDERS
# ============================================================================

def _build_nodes_from_initial(
    doc_id: str,
    data: Dict[str, Any],
) -> List[GraphNode]:
    """Convert the raw initial-extraction JSON into ``GraphNode`` objects."""
    nodes: List[GraphNode] = []

    # Document node
    doc_node = data.get("document_node", {})
    nodes.append(
        GraphNode(
            node_id=doc_node.get("node_id") or _make_id("doc"),
            doc_id=doc_id,
            node_type=NodeType.DOCUMENT,
            domain=_safe_domain(doc_node.get("domain", "LEGAL")),
            content_summary=doc_node.get("content_summary", ""),
            defined_terms=doc_node.get("defined_terms", []),
        )
    )

    # Entity nodes
    for ent in data.get("entities", []):
        nodes.append(
            GraphNode(
                node_id=ent.get("node_id") or _make_id("ent"),
                doc_id=doc_id,
                node_type=NodeType.ENTITY,
                domain=_safe_domain(doc_node.get("domain", "LEGAL")),
                content_summary=ent.get("content_summary", ""),
                defined_terms=[],
            )
        )

    # Definition nodes
    for defn in data.get("definitions", []):
        nodes.append(
            GraphNode(
                node_id=defn.get("node_id") or _make_id("def"),
                doc_id=doc_id,
                node_type=NodeType.DEFINITION,
                domain=_safe_domain(doc_node.get("domain", "LEGAL")),
                content_summary=defn.get("definition_text", ""),
                defined_terms=[defn.get("term", "")],
            )
        )

    # Clause nodes
    for clause in data.get("clauses", []):
        nodes.append(
            GraphNode(
                node_id=clause.get("node_id") or _make_id("cls"),
                doc_id=doc_id,
                node_type=NodeType.CLAUSE,
                domain=_safe_domain(doc_node.get("domain", "LEGAL")),
                content_summary=clause.get("content_summary", ""),
                defined_terms=[],
            )
        )

    return nodes


def _build_edges_from_raw(raw_edges: List[Dict[str, Any]]) -> List[GraphEdge]:
    """Convert a list of raw edge dicts into ``GraphEdge`` objects."""
    edges: List[GraphEdge] = []
    for e in raw_edges:
        source = e.get("source_node", "")
        target = e.get("target_node", "")
        if not source or not target:
            logger.warning(
                "Dropping edge %s: missing endpoint (source=%r, target=%r)",
                e.get("edge_id", "<no-id>"), source, target,
            )
            continue
        edges.append(
            GraphEdge(
                edge_id=e.get("edge_id") or _make_id("edge"),
                source_node=source,
                target_node=target,
                edge_type=_safe_edge_type(e.get("edge_type", "")),
                confidence=float(e.get("confidence", 0.5)),
                reasoning_hint=e.get("reasoning_hint", ""),
            )
        )
    return edges


def _generate_structural_edges(
    doc_node: GraphNode, child_nodes: List[GraphNode],
) -> List[GraphEdge]:
    """Create CONTAINS edges from a DOCUMENT node to its child nodes."""
    edges: List[GraphEdge] = []
    for child in child_nodes:
        if child.node_type == NodeType.DOCUMENT:
            continue
        edges.append(
            GraphEdge(
                edge_id=_make_id("contains"),
                source_node=doc_node.node_id,
                target_node=child.node_id,
                edge_type=EdgeType.CONTAINS,
                confidence=1.0,
                reasoning_hint=f"{doc_node.doc_id} contains {child.node_id}",
            )
        )
    return edges


def _build_nodes_from_gleaning(
    doc_id: str,
    entities: List[Dict[str, Any]],
    domain_fallback: str = "LEGAL",
) -> List[GraphNode]:
    """Convert additional entities discovered during gleaning."""
    nodes: List[GraphNode] = []
    for ent in entities:
        node_type_str = ent.get("node_type", "ENTITY")
        nodes.append(
            GraphNode(
                node_id=ent.get("node_id") or _make_id("gln"),
                doc_id=doc_id,
                node_type=_safe_node_type(node_type_str),
                domain=_safe_domain(ent.get("domain", domain_fallback)),
                content_summary=ent.get("content_summary", ""),
                defined_terms=ent.get("defined_terms", []),
            )
        )
    return nodes


# ============================================================================
# EXTRACTOR
# ============================================================================

class Extractor:
    """Entity and relationship extractor with gleaning loop.

    The extractor performs an initial LLM-driven extraction pass over a
    document, then runs up to ``max_gleaning_iterations`` self-reflection
    passes.  Each gleaning pass reviews the previous extraction and
    surfaces missed entities, edges, and corrections.  The loop terminates
    early when the model reports ``confidence_complete`` above
    ``min_confidence_threshold``.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol (must have a
        ``generate(prompt: str) -> str`` method).
    config:
        Dictionary with optional keys:
        - ``max_gleaning_iterations`` (int, default 3)
        - ``min_confidence_threshold`` (float, default 0.95)
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        self._llm = llm_client
        self._config = config or {}
        self._max_gleaning = self._config.get("max_gleaning_iterations", 3)
        self._min_confidence = self._config.get("min_confidence_threshold", 0.95)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, inp: ExtractionInput) -> ExtractionOutput:
        """Run full extraction pipeline: initial pass + gleaning loop.

        Parameters
        ----------
        inp:
            ``ExtractionInput`` containing the raw document and config
            overrides.

        Returns
        -------
        ExtractionOutput
            Consolidated nodes, edges, iteration count, and any entities
            flagged as potentially missed.
        """
        doc = inp.document
        max_iter = inp.extraction_config.get(
            "max_gleaning_iterations", self._max_gleaning
        )
        confidence_threshold = inp.extraction_config.get(
            "min_confidence_threshold", self._min_confidence
        )

        # --- Initial extraction ---
        nodes, edges, initial_data = self._initial_extract(
            doc,
            inp.corpus_context,
            taxonomy=inp.taxonomy,
            document_summary=inp.document_summary,
        )

        # --- Gleaning loop ---
        missed_entities: List[str] = []
        iteration = 0
        previous_json = json.dumps(initial_data, indent=2)

        for iteration in range(1, max_iter + 1):
            gleaning = self._glean(doc, previous_json)

            # Merge additional discoveries
            nodes.extend(gleaning.additional_nodes)
            edges.extend(gleaning.additional_edges)

            # Apply corrections (record only; heavy mutation deferred to
            # graph_builder which has full corpus context)
            for correction in gleaning.corrections:
                missed_entities.append(
                    correction.get("reason", "correction applied")
                )

            logger.info(
                "Gleaning iteration %d: confidence_complete=%.2f, "
                "+%d nodes, +%d edges",
                iteration,
                gleaning.confidence_complete,
                len(gleaning.additional_nodes),
                len(gleaning.additional_edges),
            )

            if gleaning.confidence_complete >= confidence_threshold:
                break

            # Build updated JSON for next gleaning pass
            previous_json = self._serialize_current_state(nodes, edges)

        return ExtractionOutput(
            nodes=nodes,
            edges=edges,
            gleaning_iterations=iteration,
            missed_entities=missed_entities,
        )

    # ------------------------------------------------------------------
    # Internal: initial extraction
    # ------------------------------------------------------------------

    def _initial_extract(
        self,
        doc: RawDocument,
        corpus_context: Optional[List[Dict[str, str]]] = None,
        taxonomy: Optional[DomainTaxonomy] = None,
        document_summary: Optional[DocumentSummary] = None,
    ) -> tuple[List[GraphNode], List[GraphEdge], Dict[str, Any]]:
        """Perform the first-pass entity/edge extraction."""
        domain_section, domain_values = _format_domain_classification(taxonomy)
        prompt = INITIAL_EXTRACTION_PROMPT.format(
            doc_id=doc.doc_id,
            content=doc.content,
            corpus_context_section=_format_corpus_context(corpus_context),
            document_analysis_section=_format_document_analysis(document_summary),
            domain_classification_section=domain_section,
            domain_values=domain_values,
        )

        raw_response = self._llm.generate(prompt)
        data = _parse_json_response(raw_response)

        nodes = _build_nodes_from_initial(doc.doc_id, data)
        edges = _build_edges_from_raw(data.get("edges", []))

        # Auto-generate structural containment edges
        if nodes:
            doc_node = nodes[0]  # DOCUMENT node is always first
            edges.extend(_generate_structural_edges(doc_node, nodes[1:]))

        return nodes, edges, data

    # ------------------------------------------------------------------
    # Internal: gleaning pass
    # ------------------------------------------------------------------

    def _glean(self, doc: RawDocument, previous_json: str) -> GleaningResult:
        """Run a single gleaning (self-reflection) pass."""
        prompt = GLEANING_PROMPT.format(
            content=doc.content,
            previous_extraction=previous_json,
        )

        raw_response = self._llm.generate(prompt)

        try:
            data = _parse_json_response(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse gleaning response: %s", exc)
            return GleaningResult(
                additional_nodes=[],
                additional_edges=[],
                corrections=[],
                confidence_complete=1.0,  # bail out gracefully
            )

        additional_nodes = _build_nodes_from_gleaning(
            doc.doc_id,
            data.get("additional_entities", []),
        )
        additional_edges = _build_edges_from_raw(
            data.get("additional_edges", [])
        )
        corrections = data.get("corrections", [])
        confidence = float(data.get("confidence_complete", 0.0))

        return GleaningResult(
            additional_nodes=additional_nodes,
            additional_edges=additional_edges,
            corrections=corrections,
            confidence_complete=confidence,
        )

    # ------------------------------------------------------------------
    # Internal: serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_current_state(
        nodes: List[GraphNode], edges: List[GraphEdge]
    ) -> str:
        """Serialise the current extraction state for the next gleaning prompt."""
        state = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "doc_id": n.doc_id,
                    "node_type": n.node_type.value,
                    "domain": n.domain.value,
                    "content_summary": n.content_summary,
                    "defined_terms": n.defined_terms,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_node": e.source_node,
                    "target_node": e.target_node,
                    "edge_type": e.edge_type.value,
                    "confidence": e.confidence,
                    "reasoning_hint": e.reasoning_hint,
                }
                for e in edges
            ],
        }
        return json.dumps(state, indent=2)
