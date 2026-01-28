"""
KnowledgeAudit Shared Contracts
Location: schemas/contracts.py
Owner: ROOT Instance

All shared type definitions for inter-instance communication between
ROOT, CHUNKER, EXTRACTOR, EXPLORER, LIBRARIAN, PLANNER, and REASONER.

v4 types are kept for backward compatibility.
v5 types are added alongside, marked with # v5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable
from enum import Enum
from datetime import datetime


# ============================================================================
# LLM CLIENT PROTOCOL
# ============================================================================

@runtime_checkable
class LLMClient(Protocol):
    """Canonical LLM client interface shared by all instances."""
    def generate(self, prompt: str) -> str: ...


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SearchStrategyType(Enum):
    """Query search strategy (formerly IntentType)."""
    COMPARISON = "COMPARISON"
    # SENSEMAKING = "SENSEMAKING"  <-- RETIRED
    LOOKUP = "LOOKUP"
    CLAIM_VERIFICATION = "CLAIM_VERIFICATION"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"
    EVIDENCE_SYNTHESIS = "EVIDENCE_SYNTHESIS"
    METHODOLOGY_COMPARISON = "METHODOLOGY_COMPARISON"


class LaneType(Enum):
    """Processing lane selection."""
    LANE_A_GRAPH_LCI = "LANE_A_GRAPH_LCI"
    # LANE_B_GRAPH_ONLY = "LANE_B_GRAPH_ONLY" <-- RETIRED
    LANE_C_VECTOR_ONLY = "LANE_C_VECTOR_ONLY"


class DomainType(Enum):
    """Document domain classification."""
    LEGAL = "LEGAL"
    REGULATORY = "REGULATORY"
    SCIENTIFIC = "SCIENTIFIC"
    FINANCIAL = "FINANCIAL"
    CROSS_DOMAIN = "CROSS_DOMAIN"


class EdgeType(Enum):
    """Knowledge graph edge types."""
    AMENDS = "AMENDS"
    GOVERNS = "GOVERNS"
    DEFINES_TERM = "DEFINES_TERM"
    INCORPORATES = "INCORPORATES"
    CROSS_REFERENCES = "CROSS_REFERENCES"
    SAME_TRANSACTION = "SAME_TRANSACTION"
    TEMPORAL_SEQUENCE = "TEMPORAL_SEQUENCE"
    CONTAINS = "CONTAINS"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    EXTENDS = "EXTENDS"
    REPLICATES = "REPLICATES"
    FAILS_TO_REPLICATE = "FAILS_TO_REPLICATE"
    SUPERSEDES = "SUPERSEDES"
    CITES = "CITES"


class NodeType(Enum):
    """Knowledge graph node types."""
    DOCUMENT = "DOCUMENT"
    ENTITY = "ENTITY"
    DEFINITION = "DEFINITION"
    CLAUSE = "CLAUSE"


class DocumentRole(Enum):
    """Role of a document in a resolved scope."""
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    CONTEXT = "CONTEXT"


class DependencyType(Enum):
    """Type of dependency between documents."""
    DEFINITION = "DEFINITION"
    AMENDMENT = "AMENDMENT"
    REGULATION = "REGULATION"
    REFERENCE = "REFERENCE"


class SubstitutionFailureReason(Enum):
    """Reason a term substitution failed during decontextualization."""
    NOT_FOUND = "NOT_FOUND"
    CIRCULAR = "CIRCULAR"
    OVERFLOW = "OVERFLOW"


class StructuralEdgeType(Enum):  # v5 — index-time edges
    """Edge types created during LIBRARIAN indexing (v5)."""
    SYNONYMY = "SYNONYMY"
    CITES = "CITES"
    REFERENCES = "REFERENCES"
    SAME_SOURCE = "SAME_SOURCE"
    SAME_CHUNK = "SAME_CHUNK"
    DEFINED_BY = "DEFINED_BY"


class EpistemicRelationship(Enum):  # v5 — reasoning-time
    """Epistemic relationships inferred by REASONER (v5)."""
    SUPPORTS = "SUPPORTS"
    STRONGLY_SUPPORTS = "STRONGLY_SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    STRONGLY_CONTRADICTS = "STRONGLY_CONTRADICTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    PARTIALLY_CONTRADICTS = "PARTIALLY_CONTRADICTS"
    REFINES = "REFINES"
    EXTENDS = "EXTENDS"
    SUPERSEDES = "SUPERSEDES"
    PRECEDES = "PRECEDES"
    UNRELATED = "UNRELATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class VerdictType(Enum):  # v5
    """Final verdict types from REASONER (v5)."""
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    PARTIALLY_REFUTED = "PARTIALLY_REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_FOUND = "NOT_FOUND"


class TraceLevel(Enum):  # v5 evaluation
    """Diagnostic trace verbosity for evaluation output."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class TokenUsage:  # v5 evaluation
    """Token consumption from a single LLM or embedding stage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ============================================================================
# INPUT CONTRACTS
# ============================================================================

@dataclass
class RawDocument:
    """Single document for ingestion."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class RawCorpus:
    """Input to INDEXER."""
    documents: List[RawDocument]
    corpus_id: str
    ingestion_timestamp: datetime


@dataclass
class QueryRequest:
    """Input to PLANNER and ROOT orchestrator."""
    query_id: str
    query_text: str
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class ScopeRequest:
    """Input to RESOLVER (via LIBRARIAN)."""
    query_id: str
    query_text: str
    router_decision: RouterDecision
    traversal_config: Optional[Dict[str, Any]] = None


@dataclass
class AssemblyRequest:
    """Input to ASSEMBLER (via LIBRARIAN)."""
    query_id: str
    scope_object: ScopeObject
    raw_documents: List[RawDocument]
    assembly_config: Optional[Dict[str, Any]] = None


# ============================================================================
# OUTPUT CONTRACTS
# ============================================================================

@dataclass
@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    node_id: str
    doc_id: str
    node_type: NodeType
    domain: DomainType
    content_summary: str
    defined_terms: List[str]


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    confidence: float
    reasoning_hint: str


@dataclass
class GraphState:
    """Graph representation of corpus."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


@dataclass
class VectorChunk:
    """Chunk in vector store."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: List[float]
    token_count: int


@dataclass
class VectorState:
    """Vector store representation."""
    chunks: List[VectorChunk]


@dataclass
class ExtractionMetadata:
    """Metadata from indexing process."""
    gleaning_iterations: int
    extraction_errors: List[str]
    coverage_score: float


@dataclass
class IndexedCorpus:
    """Output from INDEXER."""
    corpus_id: str
    graph_state: GraphState
    vector_state: VectorState
    extraction_metadata: ExtractionMetadata


@dataclass
class SearchStrategyClassification:
    """Search strategy analysis result (formerly IntentClassification)."""
    classification: SearchStrategyType
    confidence: float
    reasoning: str


@dataclass
class IdentifiedTargets:
    """Extracted targets from query."""
    explicit_docs: List[str]
    implicit_entities: List[str]


@dataclass
class RouterDecision:
    """Output from PLANNER."""
    query_id: str
    search_strategy: SearchStrategyClassification  # formerly intent
    selected_lane: LaneType
    requires_graph_scope: bool
    identified_targets: IdentifiedTargets
    domain_expectation: DomainType


@dataclass
class ScopedDocument:
    """Document in resolved scope."""
    doc_id: str
    relevance_score: float
    role: DocumentRole


@dataclass
class SupportingDocument:
    """Supporting document with dependency info."""
    doc_id: str
    dependency_type: DependencyType
    required_sections: Optional[List[str]]


@dataclass
class EdgeContext:
    """Edge information for reasoning."""
    edge_id: str
    edge_type: EdgeType
    source_doc: str
    target_doc: str
    reasoning_hint: str
    precedence_rule: Optional[str]


@dataclass
class DomainPair:
    """Domain classification for document pair."""
    primary_domain: DomainType
    secondary_domain: Optional[DomainType]
    cross_domain_logic: Optional[str]


@dataclass
class ScopeMetadata:
    """Metadata about scope resolution."""
    total_token_estimate: int
    traversal_depth: int
    edges_traversed: int
    completeness_confidence: float


@dataclass
class ScopeObject:
    """Output from RESOLVER."""
    query_id: str
    primary_documents: List[ScopedDocument]
    supporting_documents: List[SupportingDocument]
    edge_context: List[EdgeContext]
    domain_pair: DomainPair
    scope_metadata: ScopeMetadata
    warnings: List[str]


@dataclass
class LCIPreamble:
    """Preamble for LCI context."""
    edge_grammar: str
    domain_logic: str
    precedence_rules: List[str]


@dataclass
class OrderedDocument:
    """Document ordered for injection."""
    doc_id: str
    role: str
    content: str
    injection_order: int


@dataclass
class LCIPayload:
    """Payload for Long-Context Injection."""
    preamble: LCIPreamble
    documents: List[OrderedDocument]
    total_tokens: int


@dataclass
class SubstitutionFailure:
    """Record of failed term substitution."""
    term: str
    failure_reason: SubstitutionFailureReason


@dataclass
class DecontextualizationReport:
    """Report on definition resolution."""
    terms_resolved: int
    terms_failed: List[SubstitutionFailure]
    substitution_success_rate: float


@dataclass
class AssembledContext:
    """Output from ASSEMBLER."""
    query_id: str
    lci_payload: LCIPayload
    decontextualization_report: DecontextualizationReport
    assembly_warnings: List[str]


# ============================================================================
# INTER-INSTANCE MESSAGES
# ============================================================================

@dataclass
class LibrarianRequest:
    """Request from ROOT to LIBRARIAN.

    For INDEX requests the payload is an ``IndexRequest`` (which wraps
    ``RawCorpus`` plus optional ``ExplorationResult``).  Legacy callers
    may still pass a bare ``RawCorpus``; LIBRARIAN handles both.
    For RESOLVE_AND_ASSEMBLE the payload is a ``ScopeRequest``.
    """
    request_type: str  # "INDEX" | "RESOLVE_AND_ASSEMBLE"
    payload: Any       # IndexRequest | RawCorpus | ScopeRequest


@dataclass
class LibrarianResponse:
    """Response from LIBRARIAN to ROOT."""
    success: bool
    payload: Any       # IndexedCorpus | AssembledContext
    errors: List[str]


@dataclass
class PlannerRequest:
    """Request from ROOT to PLANNER."""
    query_request: QueryRequest


@dataclass
class PlannerResponse:
    """Response from PLANNER to ROOT."""
    success: bool
    router_decision: Optional[RouterDecision]
    errors: List[str]


# ============================================================================
# EXPLORER CONTRACTS
# ============================================================================

@dataclass
class DocumentSummary:
    """Summary produced by EXPLORER for a single document."""
    doc_id: str
    summary: str
    key_themes: List[str]
    apparent_domain: str
    key_terms: List[str]
    document_type: str


@dataclass
class DiscoveredDomain:
    """A domain discovered by EXPLORER from corpus analysis."""
    domain_id: str
    name: str
    description: str
    indicators: List[str]


@dataclass
class DomainRelationship:
    """Relationship between two discovered domains."""
    source_domain: str
    target_domain: str
    relationship_type: str
    reasoning: str


@dataclass
class DomainTaxonomy:
    """Taxonomy of domains discovered across the corpus."""
    domains: List[DiscoveredDomain]
    cross_domain_relationships: List[DomainRelationship]


@dataclass
class ExplorationResult:
    """Full output from an EXPLORER run."""
    corpus_id: str
    summaries: List[DocumentSummary]
    taxonomy: DomainTaxonomy
    exploration_metadata: Dict[str, Any]


@dataclass
class ExplorationRequest:
    """Request payload for EXPLORER."""
    corpus: RawCorpus
    config: Optional[Dict[str, Any]] = None


@dataclass
class ExplorationResponse:
    """Response from EXPLORER."""
    success: bool
    result: Optional[ExplorationResult]
    errors: List[str]


@dataclass
class IndexRequest:
    """Payload for INDEX requests (replaces bare RawCorpus)."""
    corpus: RawCorpus
    exploration: Optional[ExplorationResult] = None


# ============================================================================
# METRIC SCORE (kept here — used by both module and evaluation code)
# ============================================================================

@dataclass
class MetricScore:
    """Single metric measurement with pass/fail threshold."""
    name: str
    value: float
    threshold: float
    passed: bool


# ============================================================================
# v5 EMBEDDING CLIENT PROTOCOL
# ============================================================================

@runtime_checkable
class EmbeddingClient(Protocol):  # v5
    """Embedding client interface for vector operations."""
    def embed(self, texts: List[str]) -> List[List[float]]: ...


# ============================================================================
# v5 EXPLORER CONTRACTS
# ============================================================================

@dataclass
class TextChunk:  # v5
    """A chunk of text from a source document."""
    chunk_id: str
    source_doc_id: str
    text: str
    token_count: int
    position: int
    page_number: Optional[int] = None
    section_header: Optional[str] = None


@dataclass
class ExtractedEntity:  # v5
    """An entity extracted from a text chunk."""
    surface_form: str
    entity_type: str
    chunk_id: str
    char_offset: Optional[int] = None
    description: Optional[str] = None       # GraphRAG entity description


@dataclass
class ExtractedClaim:  # v5
    """A claim extracted from a text chunk."""
    claim_id: str
    claim_text: str
    subject_entity: str
    object_entity: str
    predicate: str
    description: str
    chunk_id: str
    confidence: float


@dataclass
class CoreferenceGroup:  # v5
    """A group of co-referent entity mentions."""
    canonical_form: str
    variants: List[str]


@dataclass
class EmbeddedNode:  # v5
    """A node with an embedding vector."""
    node_id: str
    node_type: str
    canonical_text: str
    embedding: List[float]
    source_doc_id: str
    source_chunks: List[str]
    metadata: Dict[str, Any]


@dataclass
class NodeSet:  # v5
    """Collection of embedded nodes produced by EXPLORER."""
    nodes: List[EmbeddedNode]
    source_doc_ids: List[str]
    chunk_count: int
    entity_count: int
    claim_count: int
    extraction_config: Optional[Dict[str, Any]] = None
    coreference_groups: Optional[List[CoreferenceGroup]] = None
    avg_claims_per_chunk: Optional[float] = None


@dataclass
class ExplorerInput:  # v5
    """Input to the v5 EXPLORER pipeline."""
    documents: List[RawDocument]
    config: Optional[Dict[str, Any]] = None


@dataclass
class ExplorerOutput:  # v5
    """Output from the v5 EXPLORER pipeline."""
    node_set: NodeSet
    taxonomy: Optional[DomainTaxonomy] = None
    chunk_node_map: Optional[Dict[str, List[str]]] = None
    trace: Optional[ExtractorTrace] = None


@dataclass
class ExplorerTrace:  # v5
    """Trace/telemetry from an EXPLORER run."""
    trace_id: str
    input_doc_ids: List[str]
    total_chunks: int
    entities_extracted: int
    claims_extracted: int
    gleaning_iterations: int
    llm_calls: int
    extraction_time_seconds: float


# ============================================================================
# v5 CHUNKER / EXTRACTOR CONTRACTS
# ============================================================================

@dataclass
class ChunkerInput:  # v5
    """Input to the CHUNKER module."""
    documents: List[RawDocument]
    config: Optional[Dict[str, Any]] = None


@dataclass
class ChunkerOutput:  # v5
    """Output from CHUNKER — handoff to EXTRACTOR."""
    chunks: List[TextChunk]
    source_doc_ids: List[str]
    chunking_config: Optional[Dict[str, Any]] = None
    trace: Optional[ChunkerTrace] = None


@dataclass
class ExtractorInput:  # v5
    """Input to the EXTRACTOR module."""
    chunks: List[TextChunk]
    source_doc_ids: List[str]
    config: Optional[Dict[str, Any]] = None


@dataclass
class ChunkSummary:  # v5
    """Summary of a single produced chunk — lightweight diagnostic record."""
    chunk_id: str
    source_doc_id: str
    token_count: int
    position: int
    section_header: Optional[str]
    text_preview: str               # first 300 chars of chunk text


@dataclass
class ChunkerTrace:  # v5
    """Trace/telemetry from a CHUNKER run."""
    trace_id: str
    input_doc_ids: List[str]
    total_chunks: int
    chunks: Optional[List[ChunkSummary]] = None


@dataclass
class ChunkExtractionDetail:  # v5
    """Per-chunk extraction diagnostics — what the LLM produced for one chunk."""
    chunk_id: str
    source_doc_id: str
    token_count: int
    text_preview: str                        # first 300 chars of chunk text
    entities: List[ExtractedEntity]          # entities extracted (including gleaned)
    claims: List[ExtractedClaim]             # claims extracted (including gleaned)
    gleaning_iterations: int
    gleaning_new_entities: int               # entities found by gleaning only
    gleaning_new_claims: int                 # claims found by gleaning only


@dataclass
class ExtractorTrace:  # v5
    """Trace/telemetry from an EXTRACTOR run."""
    trace_id: str
    input_chunk_count: int
    entities_extracted: int
    claims_extracted: int
    gleaning_iterations: int
    llm_calls: int
    extraction_time_seconds: float
    chunk_details: Optional[List[ChunkExtractionDetail]] = None


# ============================================================================
# v5 LIBRARIAN CONTRACTS
# ============================================================================

@dataclass
class StructuralEdge:  # v5
    """An edge in the v5 knowledge graph."""
    edge_id: str
    source: str
    target: str
    edge_type: StructuralEdgeType
    weight: float
    metadata: Dict[str, Any]
    similarity: Optional[float] = None
    is_cross_document: bool = False


@dataclass
class KnowledgeGraph:  # v5
    """Full knowledge graph produced by LIBRARIAN."""
    nodes: Dict[str, EmbeddedNode]
    edges: Dict[str, StructuralEdge]
    adjacency: Dict[str, List[str]]
    node_by_doc: Dict[str, List[str]]
    node_by_type: Optional[Dict[str, List[str]]] = None
    bridge_edges: Optional[List[str]] = None
    node_count: int = 0
    edge_count: int = 0
    document_count: int = 0
    bridge_count: int = 0


@dataclass
class Community:  # v5
    """A community detected in the knowledge graph."""
    community_id: str
    level: int
    node_ids: List[str]
    parent_community_id: Optional[str] = None
    child_community_ids: Optional[List[str]] = None
    summary: Optional[str] = None
    key_themes: Optional[List[str]] = None


@dataclass
class CommunityStructure:  # v5
    """Hierarchical community structure over the graph."""
    communities: Dict[str, Community]
    levels: Dict[int, List[str]]
    node_to_community: Dict[str, str]


@dataclass
class LibrarianInput:  # v5
    """Input to the v5 LIBRARIAN pipeline."""
    node_set: NodeSet
    chunk_node_map: Optional[Dict[str, List[str]]] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    taxonomy: Optional[DomainTaxonomy] = None


@dataclass
class LibrarianOutput:  # v5
    """Output from the v5 LIBRARIAN pipeline."""
    graph: KnowledgeGraph
    community_structure: Optional[CommunityStructure] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    similarity_index_path: Optional[str] = None


@dataclass
class LibrarianStats:  # v5
    """Statistics from a LIBRARIAN indexing run."""
    node_count: int
    edge_count: int
    synonymy_edge_count: int
    explicit_edge_count: int
    bridge_edge_count: int
    document_count: int
    community_count: Optional[int] = None


@dataclass
class LibrarianTrace:  # v5
    """Trace/telemetry from a LIBRARIAN run."""
    trace_id: str
    input_node_count: int
    explicit_edges_created: int
    synonymy_edges_created: int
    bridge_edges_identified: int
    similarity_threshold_used: float
    communities_detected: Optional[int] = None
    total_time_seconds: float = 0.0
    llm_calls: int = 0


# ============================================================================
# v5 PLANNER CONTRACTS
# ============================================================================

@dataclass
class SeedNode:  # v5
    """A seed node found by embedding similarity."""
    node_id: str
    similarity: float
    node_type: str
    canonical_text: str
    source_doc_id: str


@dataclass
class ScopedNode:  # v5
    """A node in the PPR-expanded scope."""
    node: EmbeddedNode
    ppr_score: float
    rank: int


@dataclass
class Scope:  # v5
    """The subgraph scope selected by PPR expansion."""
    nodes: Dict[str, ScopedNode]
    edges: Dict[str, StructuralEdge]
    document_ids: List[str]
    seed_node_ids: List[str]
    total_ppr_mass: float


@dataclass
class FocusedContext:  # v5
    """Context for Lane A (Graph-LCI) queries."""
    documents: Dict[str, RawDocument]
    edge_grammar: str
    node_context: str
    scope: Scope


@dataclass
class GlobalContext:  # v5
    """
    [DEPRECATED] Context for Lane B (Graph-only) queries.
    Retained for backward compatibility.
    """
    community_summaries: Dict[str, str]
    scope: Scope


@dataclass
class RankedChunk:  # v5
    """A chunk ranked by relevance for vector-only queries."""
    chunk: TextChunk
    relevance_score: float


@dataclass
class TargetedContext:  # v5
    """Context for Lane C (Vector-only) queries."""
    chunks: List[RankedChunk]
    scope: Scope


@dataclass
class RouterDecisionV5:  # v5
    """Router decision with v5 fields."""
    query_id: str
    search_strategy: SearchStrategyClassification  # formerly intent
    selected_lane: LaneType
    seed_nodes: List[SeedNode]
    scope: Optional[Scope] = None
    confidence: float = 0.0


@dataclass
class PlannerInput:  # v5
    """Input to the v5 PLANNER pipeline."""
    query: QueryRequest
    graph: KnowledgeGraph
    similarity_index: Any  # SimilarityIndex instance
    target_documents: Optional[List[str]] = None
    community_structure: Optional[CommunityStructure] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    chunks: Optional[List[TextChunk]] = None  # original chunker output


@dataclass
class PlannerOutput:  # v5
    """Output from the v5 PLANNER pipeline."""
    router_decision: RouterDecisionV5
    focused_context: Optional[FocusedContext] = None
    global_context: Optional[GlobalContext] = None
    targeted_context: Optional[TargetedContext] = None


@dataclass
class PlannerTrace:  # v5
    """Trace/telemetry from a PLANNER run."""
    trace_id: str
    query_id: str
    search_strategy: str
    seeds_found: int
    ppr_iterations: int
    scope_nodes: int
    scope_documents: int
    selected_lane: str
    total_time_ms: float


# ============================================================================
# v5 REASONER CONTRACTS
# ============================================================================

@dataclass
class InferredRelationship:  # v5
    """A relationship inferred by the REASONER."""
    source_id: str
    target_id: str
    relationship: EpistemicRelationship
    confidence: float
    reasoning: str
    evidence_quote: Optional[str] = None


@dataclass
class GroundedEvidence:  # v5
    """Evidence grounded to a specific source location."""
    claim: str
    source_document: str
    source_chunk: Optional[str] = None
    exact_quote: str = ""
    page_number: Optional[int] = None
    char_offset: Optional[int] = None


@dataclass
class Verdict:  # v5
    """Final verdict from the REASONER."""
    verdict_type: VerdictType
    answer: str
    confidence: float
    summary: str
    primary_relationship: Optional[EpistemicRelationship] = None


@dataclass
class ScopeSummary:  # v5
    """Summary of the scope used for a finding."""
    document_ids: List[str]
    document_count: int
    node_count: int
    edge_count: int
    bridge_edges_used: int


@dataclass
class ConfidenceFactors:  # v5
    """Breakdown of confidence components."""
    overall: float
    evidence_quality: float
    source_agreement: float
    coverage: float

@dataclass
class Finding:  # v5
    """Complete finding produced by the REASONER."""
    finding_id: str
    query_id: str
    timestamp: datetime
    verdict: Verdict
    relationships: List[InferredRelationship]
    evidence: List[GroundedEvidence]
    scope: ScopeSummary
    lane_used: str
    search_strategy: str
    confidence_factors: ConfidenceFactors
    reasoning_trace: str

@dataclass
class ReasonerInput:  # v5
    """Input to the v5 REASONER pipeline."""
    query: QueryRequest
    query_id: str
    router_decision: RouterDecisionV5
    focused_context: Optional[FocusedContext] = None
    global_context: Optional[GlobalContext] = None
    targeted_context: Optional[TargetedContext] = None


@dataclass
class ReasonerOutput:  # v5
    """Output from the v5 REASONER pipeline."""
    finding: Finding
    trace: ReasonerTrace
    raw_response: str

@dataclass
class ReasonerTrace:  # v5
    """Trace/telemetry from a REASONER run."""
    trace_id: str
    query_id: str
    lane_used: str
    search_strategy: str
    documents_in_scope: int
    relationships_inferred: int
    evidence_grounded: int
    verdict_type: str
    verdict_confidence: float
    llm_time_ms: float
    total_time_ms: float
    final_prompt: Optional[str] = None
    prompt_token_count: Optional[int] = None


# ============================================================================
# EVALUATION CONTRACTS — backward-compat re-exports
# ============================================================================
# These types now live in schemas.evaluation_contracts but are re-exported
# here so that existing ``from schemas.contracts import GroundTruthCase``
# statements continue to work.  This block MUST appear after all v5 types
# are defined because evaluation_contracts imports some of them.

from schemas.evaluation_contracts import (  # noqa: E402, F401
    EvalSeverity,
    IntrinsicMetrics,
    GroundTruthCase,
    GroundTruthResult,
    JudgeVerdict,
    JudgeReport,
    AblationVariant,
    AblationResult,
    AblationReport,
    SciFactVerdict,
    SciFactTestCase,
    SciFactPrediction,
    SciFactAblationReport,
    WikiClaimLabel,
    WikiFactClaimsRow,
    WikiClaimResult,
    WikiRowResult,
    WikiRunSummary,
    # Unified evaluation contracts
    EvalClaim,
    EvalGroup,
    ClaimTraceData,
    UnifiedClaimResult,
    GroupResult,
    UnifiedRunSummary,
)
