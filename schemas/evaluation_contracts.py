"""
KnowledgeAudit Evaluation Contracts
Location: schemas/evaluation_contracts.py
Owner: ROOT Instance

All evaluation-specific type definitions. Moved from contracts.py to
separate module I/O types from evaluation-only types.

v4 evaluation types are preserved for backward compatibility.
v5 evaluation types are added alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from schemas.contracts import (
    GraphEdge,
    SearchStrategyType,
    LaneType,
    MetricScore,
    QueryRequest,
    RawDocument,
    # v5 types needed for ground truth
    EpistemicRelationship,
    VerdictType,
    ExplorerOutput,
    LibrarianOutput,
    PlannerOutput,
    ReasonerOutput,
)


# ============================================================================
# v4 EVALUATION CONTRACTS (moved from contracts.py)
# ============================================================================

class EvalSeverity(Enum):
    """Evaluation result severity."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class IntrinsicMetrics:
    """Aggregated intrinsic validation results (Stage 1)."""
    output_completeness: MetricScore
    edge_validity: MetricScore
    document_ordering: MetricScore
    substitution_rate: MetricScore
    coverage: MetricScore
    overall_pass: bool


@dataclass
class GroundTruthCase:
    """Labeled test case for Stage 2 evaluation."""
    case_id: str
    description: str
    documents: List[RawDocument]
    query: QueryRequest
    expected_domains: List[str]
    expected_relationship_type: Optional[str]
    expected_relationship_direction: Optional[str]   # "DORA -> MSA"
    expected_contradiction: Optional[str]
    expected_strategy: SearchStrategyType
    expected_lane: LaneType
    is_synthetic: bool


@dataclass
class GroundTruthResult:
    """Result of comparing pipeline output against ground truth."""
    case_id: str
    domain_match: bool
    relationship_type_match: Optional[bool]
    relationship_direction_match: Optional[bool]
    strategy_match: bool
    lane_match: bool
    details: Dict[str, Any]


@dataclass
class JudgeVerdict:
    """LLM-as-judge verdict for a single case (Stage 3)."""
    case_id: str
    with_grammar_detected: bool
    without_grammar_detected: bool
    grammar_helped: bool
    with_grammar_reasoning: str
    without_grammar_reasoning: str


@dataclass
class JudgeReport:
    """Aggregated judge results across all cases."""
    verdicts: List[JudgeVerdict]
    grammar_lift_rate: float
    with_grammar_detection_rate: float
    without_grammar_detection_rate: float


class AblationVariant(Enum):
    """Ablation study variants (Stage 4)."""
    FULL_SYSTEM = "FULL_SYSTEM"
    LCI_NAIVE = "LCI_NAIVE"
    LCI_ORACLE = "LCI_ORACLE"


@dataclass
class AblationResult:
    """Ablation result for a single case."""
    case_id: str
    full_system_detected: bool
    lci_naive_detected: bool
    lci_oracle_detected: bool
    full_system_reasoning: str
    lci_naive_reasoning: str
    lci_oracle_reasoning: str


@dataclass
class AblationReport:
    """Aggregated ablation study results."""
    results: List[AblationResult]
    full_system_f1: float
    lci_naive_f1: float
    lci_oracle_f1: float
    graph_contribution_delta: float   # full - naive
    grammar_contribution_delta: float  # full - oracle


# ============================================================================
# SCIFACT EVALUATION CONTRACTS (moved from contracts.py)
# ============================================================================

class SciFactVerdict(Enum):
    """Verdict for a scientific claim."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass
class SciFactTestCase:
    """Labeled test case for SciFact evaluation."""
    claim_id: int
    claim_text: str
    expected_verdict: SciFactVerdict
    expected_evidence: Dict[str, List[int]]  # doc_id -> sentence indices


@dataclass
class SciFactPrediction:
    """Pipeline prediction for a SciFact claim."""
    claim_id: int
    predicted_verdict: SciFactVerdict
    predicted_evidence: Dict[str, List[int]]
    predicted_edges: List[GraphEdge]


@dataclass
class SciFactAblationReport:
    """Results from SciFact graph ablation study."""
    with_graph_accuracy: float
    without_graph_accuracy: float
    delta: float
    target_met: bool  # delta >= 0.10
    per_claim_results: List[Dict[str, Any]]


# ============================================================================
# v5 EVALUATION CONTRACTS
# ============================================================================

@dataclass
class V5MetricScore:
    """Single v5 metric measurement with pass/fail threshold."""
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str = ""


@dataclass
class ChunkerMetrics:
    """Intrinsic metrics for CHUNKER output."""
    chunk_count: V5MetricScore
    doc_coverage: V5MetricScore

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in [self.chunk_count, self.doc_coverage])

    @property
    def scores(self) -> List[V5MetricScore]:
        return [self.chunk_count, self.doc_coverage]


@dataclass
class ExplorerMetrics:
    """Intrinsic metrics for EXPLORER output."""
    node_coverage: V5MetricScore
    entity_yield: V5MetricScore
    claim_yield: V5MetricScore
    embedding_validity: V5MetricScore

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in [
            self.node_coverage,
            self.entity_yield,
            self.claim_yield,
            self.embedding_validity,
        ])

    @property
    def scores(self) -> List[V5MetricScore]:
        return [
            self.node_coverage,
            self.entity_yield,
            self.claim_yield,
            self.embedding_validity,
        ]


@dataclass
class LibrarianMetrics:
    """Intrinsic metrics for LIBRARIAN output."""
    graph_connectivity: V5MetricScore
    bridge_edge_density: V5MetricScore
    document_coverage: V5MetricScore

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in [
            self.graph_connectivity,
            self.bridge_edge_density,
            self.document_coverage,
        ])

    @property
    def scores(self) -> List[V5MetricScore]:
        return [
            self.graph_connectivity,
            self.bridge_edge_density,
            self.document_coverage,
        ]


@dataclass
class PlannerMetrics:
    """Intrinsic metrics for PLANNER output."""
    seed_discovery: V5MetricScore
    scope_coverage: V5MetricScore
    lane_consistency: V5MetricScore

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in [
            self.seed_discovery,
            self.scope_coverage,
            self.lane_consistency,
        ])

    @property
    def scores(self) -> List[V5MetricScore]:
        return [
            self.seed_discovery,
            self.scope_coverage,
            self.lane_consistency,
        ]


@dataclass
class ReasonerMetrics:
    """Intrinsic metrics for REASONER output."""
    evidence_grounding_rate: V5MetricScore
    relationship_completeness: V5MetricScore
    confidence_validity: V5MetricScore

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in [
            self.evidence_grounding_rate,
            self.relationship_completeness,
            self.confidence_validity,
        ])

    @property
    def scores(self) -> List[V5MetricScore]:
        return [
            self.evidence_grounding_rate,
            self.relationship_completeness,
            self.confidence_validity,
        ]


@dataclass
class V5PipelineMetrics:
    """Aggregated v5 pipeline intrinsic metrics."""
    explorer: ExplorerMetrics
    librarian: LibrarianMetrics
    planner: PlannerMetrics
    reasoner: ReasonerMetrics
    overall_pass: bool
    chunker: Optional[ChunkerMetrics] = None

    @property
    def all_scores(self) -> List[V5MetricScore]:
        scores = []
        if self.chunker is not None:
            scores.extend(self.chunker.scores)
        scores.extend(
            self.explorer.scores
            + self.librarian.scores
            + self.planner.scores
            + self.reasoner.scores
        )
        return scores


@dataclass
class V5GroundTruthCase:
    """Labeled test case for v5 pipeline evaluation."""
    case_id: str
    description: str
    documents: List[RawDocument]
    query: QueryRequest
    expected_verdict: VerdictType
    expected_relationship: EpistemicRelationship
    expected_lane: LaneType
    expected_strategy: SearchStrategyType
    is_synthetic: bool
    gold_doc_ids: Optional[List[str]] = None


@dataclass
class V5GroundTruthResult:
    """Result of comparing v5 pipeline output against ground truth."""
    case_id: str
    verdict_match: bool
    relationship_match: bool
    lane_match: bool
    relationship_match: bool
    lane_match: bool
    strategy_match: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_matched(self) -> bool:
        return (
            self.verdict_match
            and self.relationship_match
            and self.lane_match
            and self.strategy_match
        )


# ============================================================================
# TWO-STAGE FAILURE ISOLATION CONTRACTS
# ============================================================================

class FailureStage(Enum):
    """Top-level failure classification for two-stage evaluation."""
    SUCCESS = "SUCCESS"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    REASONING_FAILURE = "REASONING_FAILURE"


class RetrievalFailureAttribution(Enum):
    """Sub-classification for retrieval failures."""
    LIBRARIAN_FAILURE = "LIBRARIAN_FAILURE"
    PLANNER_FAILURE = "PLANNER_FAILURE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class GraphRecallResult:
    """Per-gold-document retrieval check."""
    gold_doc_id: str
    in_graph: bool
    in_scope: bool
    node_count_in_graph: int
    node_count_in_scope: int
    attribution: RetrievalFailureAttribution


@dataclass
class StageAResult:
    """Stage A: Retrieval audit — did the gold documents land in scope?"""
    gold_doc_ids: List[str]
    scope_doc_ids: List[str]
    per_doc_results: List[GraphRecallResult]
    graph_recall_at_k: float
    all_gold_in_scope: bool
    failure_attribution: RetrievalFailureAttribution
    has_scope: bool


@dataclass
class StageBResult:
    """Stage B: Reasoning audit — given correct retrieval, is the verdict right?"""
    expected_verdict: str
    actual_verdict: str
    verdict_correct: bool
    confidence: float
    stage_a_passed: bool


@dataclass
class TwoStageResult:
    """Combined two-stage evaluation for a single case."""
    failure_stage: FailureStage
    stage_a: StageAResult
    stage_b: Optional[StageBResult]
    retrieval_attribution: RetrievalFailureAttribution


@dataclass
class TwoStageReport:
    """Aggregate two-stage evaluation across all cases."""
    total_cases: int
    success_count: int
    retrieval_failure_count: int
    reasoning_failure_count: int
    librarian_failure_count: int
    planner_failure_count: int
    success_rate: float
    retrieval_failure_rate: float
    reasoning_failure_rate: float
    graph_recall_at_k_mean: float
    verdict_accuracy_when_retrieved: float


# ============================================================================
# WIKIFACTCLAIMS EVALUATION CONTRACTS
# ============================================================================

@dataclass
class WikiClaimLabel:
    """A single claim with its ground-truth verdict label."""
    text: str
    label: SciFactVerdict


@dataclass
class WikiFactClaimsRow:
    """One row from the WikiFactClaims dataset (HotpotQA-derived).

    Each row contains a multi-hop question, its answer, 10 context passages,
    and supporting-fact annotations.  Claims are derived externally and carry
    SUPPORTS / REFUTES / NOT_ENOUGH_INFO labels.
    """
    row_id: str
    question: str
    answer: str
    question_type: str          # comparison | bridge
    level: str                  # medium | hard
    claims: List[WikiClaimLabel]
    supporting_facts_titles: List[str]
    supporting_facts_sent_ids: List[int]
    context_titles: List[str]
    context_sentences: List[List[str]]


@dataclass
class WikiClaimResult:
    """Evaluation result for a single claim within a WikiFactClaims row."""
    row_id: str
    claim_index: int
    claim_text: str
    expected_verdict: SciFactVerdict
    predicted_verdict_type: Optional[VerdictType] = None
    mapped_verdict: Optional[SciFactVerdict] = None
    correct: bool = False
    confidence: float = 0.0
    planner_scope_titles: List[str] = field(default_factory=list)
    supporting_facts_in_scope: int = 0
    supporting_facts_total: int = 0
    scope_recall: float = 0.0
    error: Optional[str] = None
    # --- Planner trace ---
    trace_seeds_found: int = 0
    trace_scope_nodes: int = 0
    trace_scope_documents: int = 0
    trace_selected_lane: str = ""
    trace_search_strategy: str = ""
    trace_router_confidence: float = 0.0
    trace_ppr_total_mass: float = 0.0
    # --- Reasoner trace ---
    trace_docs_in_scope: int = 0
    trace_relationships_inferred: int = 0
    trace_evidence_grounded: int = 0
    trace_llm_time_ms: float = 0.0
    trace_total_time_ms: float = 0.0
    trace_verdict_answer: str = ""
    trace_verdict_summary: str = ""
    trace_reasoning_trace: str = ""
    trace_raw_response: str = ""
    # --- Confidence factor breakdown ---
    trace_confidence_overall: float = 0.0
    trace_confidence_evidence_quality: float = 0.0
    trace_confidence_source_agreement: float = 0.0
    trace_confidence_coverage: float = 0.0
    # --- Detail lists ---
    trace_relationships_detail: List[Dict[str, Any]] = field(default_factory=list)
    trace_evidence_detail: List[Dict[str, Any]] = field(default_factory=list)
    trace_seed_details: List[Dict[str, Any]] = field(default_factory=list)
    trace_ranked_chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WikiRowResult:
    """Aggregated evaluation result for all claims in a single row."""
    row_id: str
    question: str
    question_type: str
    level: str
    index_time_seconds: float
    graph_nodes: int
    graph_edges: int
    graph_bridge_count: int = 0
    graph_community_count: int = 0
    extractor_entities: int = 0
    extractor_claims: int = 0
    extractor_llm_calls: int = 0
    extractor_time_seconds: float = 0.0
    claim_results: List[WikiClaimResult] = field(default_factory=list)


@dataclass
class WikiRunSummary:
    """Aggregate summary of the WikiFactClaims evaluation run."""
    total_rows: int
    total_claims: int
    completed_claims: int
    errored_claims: int
    verdict_accuracy: float
    mean_scope_recall: float
    mean_confidence: float
    per_verdict_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_type_accuracy: Dict[str, float] = field(default_factory=dict)
    per_level_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_index_time: float = 0.0
    llm_model: str = ""
    embed_model: str = ""
    row_results: List[WikiRowResult] = field(default_factory=list)


# ============================================================================
# UNIFIED EVALUATION CONTRACTS
# ============================================================================

@dataclass
class EvalClaim:
    """Dataset-agnostic evaluation claim."""
    claim_id: str
    claim_text: str
    expected_verdict: SciFactVerdict
    gold_doc_ids: List[str] = field(default_factory=list)
    expected_evidence: Dict[str, List[int]] = field(default_factory=dict)  # SciFact sentence-level
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalGroup:
    """Group of claims sharing a single index.
    SciFact: 1 group (full corpus). WikiFactClaims: N groups (1 per row)."""
    group_id: str
    claims: List[EvalClaim]
    documents: List[RawDocument]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimTraceData:
    """Nested per-claim trace (superset of both runners)."""
    # Planner
    seeds_found: int = 0
    scope_nodes: int = 0
    scope_documents: int = 0
    selected_lane: str = ""
    search_strategy: str = ""
    router_confidence: float = 0.0
    ppr_total_mass: float = 0.0
    seed_details: List[Dict[str, Any]] = field(default_factory=list)
    scope_node_details: List[Dict[str, Any]] = field(default_factory=list)
    ranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    planner_scope_doc_ids: List[str] = field(default_factory=list)
    planner_scope_recall: float = 0.0
    # Reasoner
    docs_in_scope: int = 0
    relationships_inferred: int = 0
    evidence_grounded: int = 0
    llm_time_ms: float = 0.0
    total_time_ms: float = 0.0
    verdict_answer: str = ""
    verdict_summary: str = ""
    reasoning_trace: str = ""
    raw_response: str = ""
    # Confidence
    confidence_overall: float = 0.0
    confidence_evidence_quality: float = 0.0
    confidence_source_agreement: float = 0.0
    confidence_coverage: float = 0.0
    # Scope from finding
    scope_document_ids: List[str] = field(default_factory=list)
    scope_bridge_edges_used: int = 0
    # Detail
    relationships_detail: List[Dict[str, Any]] = field(default_factory=list)
    evidence_detail: List[Dict[str, Any]] = field(default_factory=list)
    # Two-stage (SciFact only, zeroed for Wiki)
    two_stage_failure: str = ""
    retrieval_attribution: str = ""
    graph_recall_at_k: float = 0.0
    gold_docs_in_scope: int = 0
    gold_docs_total: int = 0
    gold_docs_in_graph: int = 0
    evidence_failure_attribution: List[Dict[str, Any]] = field(default_factory=list)

    # Per-stage performance (v5 evaluation)
    stage_timing_planner_ms: float = 0.0
    stage_timing_reasoner_ms: float = 0.0
    stage_timing_total_ms: float = 0.0
    reasoner_prompt_tokens: int = 0
    reasoner_completion_tokens: int = 0
    reasoner_total_tokens: int = 0
    planner_prompt_tokens: int = 0
    planner_total_tokens: int = 0

    # Reasoner prompt (FULL level, needs REASONER module change)
    final_prompt: Optional[str] = None
    prompt_token_count: Optional[int] = None


@dataclass
class UnifiedClaimResult:
    """Result for a single claim, dataset-agnostic."""
    claim_id: str
    group_id: str
    claim_text: str
    expected_verdict: SciFactVerdict
    predicted_verdict_type: Optional[VerdictType] = None
    mapped_verdict: Optional[SciFactVerdict] = None
    correct: bool = False
    confidence: float = 0.0
    evidence_doc_ids: List[str] = field(default_factory=list)
    expected_doc_ids: List[str] = field(default_factory=list)
    evidence_doc_recall: float = 0.0
    scope_recall: float = 0.0
    lane_used: str = ""
    search_strategy: str = ""
    relationships: List[str] = field(default_factory=list)
    error: Optional[str] = None
    trace: ClaimTraceData = field(default_factory=ClaimTraceData)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupResult:
    """Result for one group (1 index + N claims)."""
    group_id: str
    index_time_seconds: float
    graph_nodes: int
    graph_edges: int
    graph_bridge_count: int = 0
    graph_community_count: int = 0
    corpus_size: int = 0
    extractor_entities: int = 0
    extractor_claims: int = 0
    extractor_llm_calls: int = 0
    extractor_time_seconds: float = 0.0
    claim_results: List[UnifiedClaimResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    index_diagnostics: Optional[Any] = None  # IndexDiagnostics (SciFact)


@dataclass
class UnifiedRunSummary:
    """Aggregate summary for any dataset."""
    dataset_name: str
    total_groups: int
    total_claims: int
    completed: int
    errored: int
    verdict_accuracy: float = 0.0
    mean_evidence_recall: float = 0.0
    mean_scope_recall: float = 0.0
    mean_confidence: float = 0.0
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_verdict_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_index_time: float = 0.0
    llm_model: str = ""
    embed_model: str = ""
    group_results: List[GroupResult] = field(default_factory=list)
    # SciFact-specific
    corpus_size: int = 0
    corpus_mode: str = ""
    graph_nodes: int = 0
    graph_edges: int = 0
    two_stage_claims_with_evidence: int = 0
    two_stage_success_rate: float = 0.0
    two_stage_retrieval_failure_rate: float = 0.0
    two_stage_reasoning_failure_rate: float = 0.0
    two_stage_librarian_failure_count: int = 0
    two_stage_planner_failure_count: int = 0
    two_stage_graph_recall_mean: float = 0.0
    two_stage_verdict_acc_when_retrieved: float = 0.0
    index_diagnostics: Optional[Any] = None
    # Wiki-specific
    per_type_accuracy: Dict[str, float] = field(default_factory=dict)
    per_level_accuracy: Dict[str, float] = field(default_factory=dict)
