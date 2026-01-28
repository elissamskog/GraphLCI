"""
SciFact Full-Corpus v5 Evaluation
Location: evaluation/run_scifact_v5.py
Owner: ROOT Instance

Indexes the full SciFact corpus (~5K abstracts) once with the v5 pipeline,
then runs every labeled claim as a query. Writes concise metrics and
predictions to output.md, full trace data to output.trace.json.

Usage:
    python -m evaluation.run_scifact_v5 [--limit N] [--output path]
    python -m evaluation.run_scifact_v5 --limit 3 --mock
    python -m evaluation.run_scifact_v5 --model qwen2.5:14b --embed-model nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# -- sys.path setup ----------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_PLANNER_DIR = os.path.join(_PROJECT_ROOT, "planner")
if _PLANNER_DIR not in sys.path:
    sys.path.insert(0, _PLANNER_DIR)

from orchestrator import OrchestratorV5
from llm_client import get_client, get_embedding_client
from evaluation.data_loaders.scifact_loader import SciFactLoader
from schemas.contracts import (
    VerdictType,
    QueryRequest,
    ReasonerOutput,
    TraceLevel,
)
from schemas.evaluation_contracts import SciFactVerdict
from evaluation.two_stage_metrics import evaluate_two_stage

_logger = logging.getLogger(__name__)


# ============================================================================
# UNBUFFERED PRINT
# ============================================================================

def _print(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True so output appears immediately in pipes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================================
# VERDICT MAPPING
# ============================================================================

_VERDICT_MAP: Dict[VerdictType, SciFactVerdict] = {
    VerdictType.CONFIRMED: SciFactVerdict.SUPPORTS,
    VerdictType.PARTIALLY_CONFIRMED: SciFactVerdict.SUPPORTS,
    VerdictType.REFUTED: SciFactVerdict.REFUTES,
    VerdictType.PARTIALLY_REFUTED: SciFactVerdict.REFUTES,
    VerdictType.INCONCLUSIVE: SciFactVerdict.NOT_ENOUGH_INFO,
    VerdictType.NOT_FOUND: SciFactVerdict.NOT_ENOUGH_INFO,
}


def map_verdict(verdict_type: VerdictType) -> SciFactVerdict:
    """Map a v5 VerdictType to a SciFactVerdict."""
    return _VERDICT_MAP.get(verdict_type, SciFactVerdict.NOT_ENOUGH_INFO)


# ============================================================================
# SMALL-CORPUS BUILDER
# ============================================================================

def build_small_corpus(
    loader: Any,
    test_cases: List[Any],
    target_size: int,
) -> list:
    """Build a focused corpus containing evidence docs + random distractors.

    Guarantees every evidence document from the selected test cases is included.
    Pads with random non-evidence documents to reach target_size.
    If evidence docs alone exceed target_size, returns all evidence docs (no cap).
    """
    import random

    all_docs = loader.get_all_abstracts()
    docs_by_id = {d.doc_id: d for d in all_docs}

    # Collect every evidence doc_id across all selected claims
    evidence_ids: set = set()
    for case in test_cases:
        evidence_ids.update(case.expected_evidence.keys())

    # Evidence documents (guaranteed included)
    evidence_docs = [docs_by_id[did] for did in evidence_ids if did in docs_by_id]

    # Pad with distractors
    needed = max(0, target_size - len(evidence_docs))
    if needed > 0:
        non_evidence = [d for d in all_docs if d.doc_id not in evidence_ids]
        distractors = random.sample(non_evidence, min(needed, len(non_evidence)))
    else:
        distractors = []

    documents = evidence_docs + distractors
    random.shuffle(documents)  # Don't leak evidence-first ordering
    return documents


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ClaimResult:
    """Result for a single claim evaluation."""
    claim_id: int
    claim_text: str
    expected_verdict: SciFactVerdict
    predicted_verdict_type: Optional[VerdictType] = None
    mapped_verdict: Optional[SciFactVerdict] = None
    correct: bool = False
    confidence: float = 0.0
    evidence_doc_ids: List[str] = field(default_factory=list)
    expected_doc_ids: List[str] = field(default_factory=list)
    evidence_doc_recall: float = 0.0
    lane_used: str = ""
    intent: str = ""
    relationships: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Planner trace diagnostics
    trace_seeds_found: int = 0
    trace_scope_nodes: int = 0
    trace_scope_documents: int = 0
    trace_selected_lane: str = ""
    trace_intent: str = ""
    # Reasoner trace diagnostics
    trace_docs_in_scope: int = 0
    trace_relationships_inferred: int = 0
    trace_evidence_grounded: int = 0
    trace_llm_time_ms: float = 0.0
    trace_total_time_ms: float = 0.0
    # Reasoner detail (standard+ trace level)
    trace_verdict_answer: str = ""
    trace_verdict_summary: str = ""
    trace_reasoning_trace: str = ""
    trace_raw_response: str = ""               # FULL level only
    trace_confidence_overall: float = 0.0
    trace_confidence_evidence_quality: float = 0.0
    trace_confidence_source_agreement: float = 0.0
    trace_confidence_coverage: float = 0.0
    trace_scope_document_ids: List[str] = field(default_factory=list)  # from finding.scope (reasoner), not planner
    trace_scope_bridge_edges_used: int = 0
    trace_relationships_detail: List[Dict[str, Any]] = field(default_factory=list)
    trace_evidence_detail: List[Dict[str, Any]] = field(default_factory=list)
    # Planner detail (standard+ trace level)
    trace_seed_details: List[Dict[str, Any]] = field(default_factory=list)
    trace_router_confidence: float = 0.0
    trace_ppr_total_mass: float = 0.0
    trace_scope_node_details: List[Dict[str, Any]] = field(default_factory=list)
    trace_ranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    # Two-stage failure isolation
    two_stage_failure: str = ""
    retrieval_attribution: str = ""
    graph_recall_at_k: float = 0.0
    gold_docs_in_scope: int = 0
    gold_docs_total: int = 0
    gold_docs_in_graph: int = 0
    # Planner scope (from planner_output.router_decision.scope, not finding.scope)
    planner_scope_doc_ids: List[str] = field(default_factory=list)
    planner_scope_recall: float = 0.0
    # Per-evidence-doc failure attribution
    evidence_failure_attribution: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IndexDiagnostics:
    """Diagnostic data captured from the indexing pipeline."""
    # Chunker output
    chunker_total_chunks: int = 0
    chunker_doc_count: int = 0
    chunker_token_stats: Dict[str, float] = field(default_factory=dict)
    chunker_section_headers_detected: int = 0
    chunker_chunks_per_doc: Dict[str, int] = field(default_factory=dict)
    # Extractor output
    extractor_entities_extracted: int = 0
    extractor_claims_extracted: int = 0
    extractor_gleaning_iterations: int = 0
    extractor_llm_calls: int = 0
    extractor_time_seconds: float = 0.0
    extractor_error: Optional[str] = None
    extractor_chunks_with_zero_entities: int = 0
    extractor_chunks_with_zero_claims: int = 0
    extractor_mean_entities_per_chunk: float = 0.0
    extractor_mean_claims_per_chunk: float = 0.0
    # Explorer output
    explorer_chunk_count: int = 0
    explorer_entity_count: int = 0
    explorer_claim_count: int = 0
    explorer_node_count: int = 0
    explorer_source_doc_count: int = 0
    explorer_coref_groups: int = 0
    explorer_avg_claims_per_chunk: float = 0.0
    # Librarian output
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_document_count: int = 0
    graph_bridge_count: int = 0
    community_count: int = 0
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    docs_with_nodes: int = 0
    docs_without_nodes: int = 0
    explorer_error: Optional[str] = None
    # Coreference detail
    coref_group_details: List[Dict[str, Any]] = field(default_factory=list)
    coref_multi_doc_groups: int = 0
    # Cross-doc entity overlap
    cross_doc_entity_overlap: List[Dict[str, Any]] = field(default_factory=list)
    cross_doc_overlap_entity_count: int = 0
    unique_entities_total: int = 0


@dataclass
class RunSummary:
    """Aggregate summary of the evaluation run."""
    total_claims: int = 0
    completed: int = 0
    errored: int = 0
    verdict_accuracy: float = 0.0
    mean_evidence_recall: float = 0.0
    mean_confidence: float = 0.0
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_verdict_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    claim_results: List[ClaimResult] = field(default_factory=list)
    index_time_seconds: float = 0.0
    corpus_size: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    llm_model: str = ""
    embed_model: str = ""
    corpus_mode: str = "full"
    index_diagnostics: Optional[IndexDiagnostics] = None
    # Two-stage aggregate metrics
    two_stage_claims_with_evidence: int = 0
    two_stage_success_rate: float = 0.0
    two_stage_retrieval_failure_rate: float = 0.0
    two_stage_reasoning_failure_rate: float = 0.0
    two_stage_librarian_failure_count: int = 0
    two_stage_planner_failure_count: int = 0
    two_stage_graph_recall_mean: float = 0.0
    two_stage_verdict_acc_when_retrieved: float = 0.0


# ============================================================================
# EVIDENCE RECALL
# ============================================================================

def compute_evidence_doc_recall(
    finding: Any,
    expected_evidence: Dict[str, List[int]],
) -> float:
    """Compute document-level recall of evidence.

    Returns len(found_doc_ids & expected_doc_ids) / len(expected_doc_ids).
    """
    if not expected_evidence:
        return 1.0  # No expected evidence = trivially recalled

    expected_doc_ids = set(expected_evidence.keys())
    found_doc_ids = set()
    for ev in finding.evidence:
        if ev.source_document:
            found_doc_ids.add(str(ev.source_document))

    intersection = found_doc_ids & expected_doc_ids
    return len(intersection) / len(expected_doc_ids)


# ============================================================================
# SINGLE CLAIM RUNNER
# ============================================================================

def run_single_claim(
    orchestrator: OrchestratorV5,
    index: Any,
    case: Any,
    n: int,
    total: int,
    trace_level: TraceLevel = TraceLevel.STANDARD,
) -> ClaimResult:
    """Process a single claim through the v5 pipeline.

    Parameters
    ----------
    orchestrator : OrchestratorV5
    index : LibrarianOutput
    case : SciFactTestCase
    n : int — 1-based claim index
    total : int — total claim count
    trace_level : TraceLevel — diagnostic verbosity
    """
    result = ClaimResult(
        claim_id=case.claim_id,
        claim_text=case.claim_text,
        expected_verdict=case.expected_verdict,
        expected_doc_ids=list(case.expected_evidence.keys()),
    )

    try:
        query = QueryRequest(
            query_id=f"scifact-v5-{case.claim_id}",
            query_text=case.claim_text,
        )

        reasoner_output: ReasonerOutput = orchestrator.process_query_v5(query, index)
        finding = reasoner_output.finding

        # Extract fields
        result.predicted_verdict_type = finding.verdict.verdict_type
        result.mapped_verdict = map_verdict(finding.verdict.verdict_type)
        result.correct = result.mapped_verdict == case.expected_verdict
        result.confidence = finding.verdict.confidence
        result.lane_used = finding.lane_used
        result.intent = finding.intent_classified
        result.evidence_doc_ids = [
            str(ev.source_document) for ev in finding.evidence if ev.source_document
        ]
        result.relationships = [
            rel.relationship.value for rel in finding.relationships
        ]
        result.evidence_doc_recall = compute_evidence_doc_recall(
            finding, case.expected_evidence,
        )

        # Capture Planner trace diagnostics
        planner_out = orchestrator.last_planner_output
        if planner_out:
            rd = planner_out.router_decision
            result.trace_seeds_found = len(rd.seed_nodes)
            result.trace_selected_lane = rd.selected_lane.value
            result.trace_intent = rd.intent.classification.value
            if rd.scope:
                result.trace_scope_nodes = len(rd.scope.nodes)
                result.trace_scope_documents = len(rd.scope.document_ids)
                # Planner scope doc IDs (from router_decision.scope, not finding.scope)
                result.planner_scope_doc_ids = list(rd.scope.document_ids)

        # Two-stage failure isolation
        gold_doc_ids = list(case.expected_evidence.keys())
        try:
            two_stage = evaluate_two_stage(
                case_id=str(case.claim_id),
                gold_doc_ids=gold_doc_ids,
                graph=index.graph,
                planner_output=orchestrator.last_planner_output,
                reasoner_output=reasoner_output,
                expected_verdict=case.expected_verdict.value,
                verdict_mapper=lambda vt: map_verdict(vt).value,
            )
            result.two_stage_failure = two_stage.failure_stage.value
            result.retrieval_attribution = two_stage.retrieval_attribution.value
            result.graph_recall_at_k = two_stage.stage_a.graph_recall_at_k
            result.gold_docs_in_scope = sum(
                1 for dr in two_stage.stage_a.per_doc_results if dr.in_scope
            )
            result.gold_docs_total = len(gold_doc_ids)
            result.gold_docs_in_graph = sum(
                1 for dr in two_stage.stage_a.per_doc_results if dr.in_graph
            )
        except Exception:
            _logger.debug(
                "Two-stage evaluation failed for claim %s",
                case.claim_id,
                exc_info=True,
            )

        # Planner scope recall (independent of two_stage eval)
        if gold_doc_ids:
            scope_set = set(result.planner_scope_doc_ids)
            expected_set = set(gold_doc_ids)
            result.planner_scope_recall = (
                len(scope_set & expected_set) / len(expected_set)
            )
        else:
            result.planner_scope_recall = 1.0  # vacuous

        # Per-evidence-doc failure attribution (independent of two_stage eval)
        grounded_doc_ids = set(result.evidence_doc_ids)
        scope_set = set(result.planner_scope_doc_ids)
        for doc_id in gold_doc_ids:
            graph_node_ids = index.graph.node_by_doc.get(doc_id, [])
            in_graph = len(graph_node_ids) > 0
            in_scope = doc_id in scope_set
            in_grounded = doc_id in grounded_doc_ids

            if in_grounded:
                status = "GROUNDED"
            elif in_scope:
                status = "IN_SCOPE_NOT_GROUNDED"
            elif in_graph:
                status = "IN_GRAPH_NOT_IN_SCOPE"
            else:
                status = "ABSENT_FROM_GRAPH"

            result.evidence_failure_attribution.append({
                "doc_id": doc_id,
                "in_graph": in_graph,
                "in_scope": in_scope,
                "in_grounded": in_grounded,
                "status": status,
            })

        # Capture Reasoner trace diagnostics
        trace = reasoner_output.trace
        result.trace_docs_in_scope = trace.documents_in_scope
        result.trace_relationships_inferred = trace.relationships_inferred
        result.trace_evidence_grounded = trace.evidence_grounded
        result.trace_llm_time_ms = trace.llm_time_ms
        result.trace_total_time_ms = trace.total_time_ms

        # ---- Standard+ trace level: capture rich detail ----
        if trace_level in (TraceLevel.STANDARD, TraceLevel.FULL):
            is_full = trace_level == TraceLevel.FULL

            # Reasoner detail from finding
            result.trace_verdict_answer = finding.verdict.answer
            result.trace_verdict_summary = finding.verdict.summary
            result.trace_reasoning_trace = finding.reasoning_trace

            # Confidence factor breakdown
            cf = finding.confidence_factors
            result.trace_confidence_overall = cf.overall
            result.trace_confidence_evidence_quality = cf.evidence_quality
            result.trace_confidence_source_agreement = cf.source_agreement
            result.trace_confidence_coverage = cf.coverage

            # Scope summary
            result.trace_scope_document_ids = list(finding.scope.document_ids)
            result.trace_scope_bridge_edges_used = finding.scope.bridge_edges_used

            # Relationships detail
            rel_limit = len(finding.relationships) if is_full else 20
            for rel in finding.relationships[:rel_limit]:
                result.trace_relationships_detail.append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relationship": rel.relationship.value,
                    "confidence": rel.confidence,
                    "reasoning": rel.reasoning,
                    "evidence_quote": rel.evidence_quote or "",
                })

            # Evidence detail
            for ev in finding.evidence:
                result.trace_evidence_detail.append({
                    "claim": ev.claim,
                    "source_document": ev.source_document,
                    "source_chunk": ev.source_chunk,
                    "exact_quote": ev.exact_quote,
                })

            # Planner detail from router_decision
            if planner_out:
                rd = planner_out.router_decision
                result.trace_router_confidence = rd.confidence

                # Seed node details
                seed_limit = len(rd.seed_nodes) if is_full else 10
                for sn in rd.seed_nodes[:seed_limit]:
                    result.trace_seed_details.append({
                        "node_id": sn.node_id,
                        "similarity": sn.similarity,
                        "node_type": sn.node_type,
                        "canonical_text": sn.canonical_text,
                        "source_doc_id": sn.source_doc_id,
                    })

                # Scope node details (top by PPR score)
                if rd.scope:
                    result.trace_ppr_total_mass = rd.scope.total_ppr_mass
                    sorted_scope = sorted(
                        rd.scope.nodes.values(),
                        key=lambda sn: sn.ppr_score,
                        reverse=True,
                    )
                    scope_limit = len(sorted_scope) if is_full else 15
                    for sn in sorted_scope[:scope_limit]:
                        result.trace_scope_node_details.append({
                            "node_id": sn.node.node_id,
                            "ppr_score": sn.ppr_score,
                            "rank": sn.rank,
                            "node_type": sn.node.node_type,
                            "canonical_text": sn.node.canonical_text,
                            "source_doc_id": sn.node.source_doc_id,
                        })

                # Lane C ranked chunks
                if planner_out.targeted_context and planner_out.targeted_context.chunks:
                    chunk_limit = len(planner_out.targeted_context.chunks) if is_full else 5
                    for rc in planner_out.targeted_context.chunks[:chunk_limit]:
                        result.trace_ranked_chunks.append({
                            "chunk_id": rc.chunk.chunk_id,
                            "source_doc_id": rc.chunk.source_doc_id,
                            "relevance_score": rc.relevance_score,
                            "token_count": rc.chunk.token_count,
                            "text_preview": rc.chunk.text[:200],
                        })

            # Full level: raw LLM response
            if is_full:
                result.trace_raw_response = reasoner_output.raw_response

        # Progress
        check = "\u2713" if result.correct else "\u2717"
        _print(
            f"  [{n}/{total}] claim={case.claim_id} "
            f"{finding.verdict.verdict_type.value}\u2192{result.mapped_verdict.value} "
            f"(exp: {case.expected_verdict.value}) {check} "
            f"conf={result.confidence:.2f} "
            f"lane={result.lane_used} scope={trace.documents_in_scope}docs "
            f"{trace.total_time_ms:.0f}ms"
        )

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        _print(f"  [{n}/{total}] claim={case.claim_id} ERROR: {result.error}")

    return result


# ============================================================================
# SUMMARY BUILDER
# ============================================================================

def build_summary(
    results: List[ClaimResult],
    index_time: float,
    corpus_size: int,
    graph_nodes: int,
    graph_edges: int,
    llm_model: str,
    embed_model: str,
) -> RunSummary:
    """Aggregate claim results into a RunSummary."""
    summary = RunSummary(
        total_claims=len(results),
        claim_results=results,
        index_time_seconds=index_time,
        corpus_size=corpus_size,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        llm_model=llm_model,
        embed_model=embed_model,
    )

    completed = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]
    summary.completed = len(completed)
    summary.errored = len(errored)

    if completed:
        summary.verdict_accuracy = sum(1 for r in completed if r.correct) / len(completed)
        summary.mean_evidence_recall = sum(r.evidence_doc_recall for r in completed) / len(completed)
        summary.mean_confidence = sum(r.confidence for r in completed) / len(completed)

    # Confusion matrix
    verdicts = [v.value for v in SciFactVerdict]
    cm: Dict[str, Dict[str, int]] = {v: {v2: 0 for v2 in verdicts} for v in verdicts}
    for r in completed:
        if r.mapped_verdict:
            cm[r.expected_verdict.value][r.mapped_verdict.value] += 1
    summary.confusion_matrix = cm

    # Per-verdict breakdown
    for v in verdicts:
        subset = [r for r in completed if r.expected_verdict.value == v]
        if subset:
            correct = sum(1 for r in subset if r.correct)
            summary.per_verdict_accuracy[v] = {
                "count": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
            }

    # Two-stage aggregate metrics (filter to claims with actual gold evidence)
    two_stage_completed = [
        r for r in completed if r.two_stage_failure and r.gold_docs_total > 0
    ]
    summary.two_stage_claims_with_evidence = len(two_stage_completed)
    if two_stage_completed:
        n = len(two_stage_completed)
        success = sum(1 for r in two_stage_completed if r.two_stage_failure == "SUCCESS")
        retr_fail = sum(1 for r in two_stage_completed if r.two_stage_failure == "RETRIEVAL_FAILURE")
        reas_fail = sum(1 for r in two_stage_completed if r.two_stage_failure == "REASONING_FAILURE")
        summary.two_stage_success_rate = success / n
        summary.two_stage_retrieval_failure_rate = retr_fail / n
        summary.two_stage_reasoning_failure_rate = reas_fail / n
        summary.two_stage_librarian_failure_count = sum(
            1 for r in two_stage_completed if r.retrieval_attribution == "LIBRARIAN_FAILURE"
        )
        summary.two_stage_planner_failure_count = sum(
            1 for r in two_stage_completed if r.retrieval_attribution == "PLANNER_FAILURE"
        )
        summary.two_stage_graph_recall_mean = sum(
            r.graph_recall_at_k for r in two_stage_completed
        ) / n
        # Verdict accuracy among claims where retrieval succeeded
        retrieved = [r for r in two_stage_completed if r.two_stage_failure != "RETRIEVAL_FAILURE"]
        if retrieved:
            summary.two_stage_verdict_acc_when_retrieved = sum(
                1 for r in retrieved if r.correct
            ) / len(retrieved)

    return summary


# ============================================================================
# OUTPUT FORMATTER
# ============================================================================

def format_output_md(summary: RunSummary) -> str:
    """Render concise metrics and predictions to output.md."""
    lines: List[str] = []

    lines.append("# SciFact v5 Evaluation Results\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append(f"- **Pipeline:** v5")
    lines.append(f"- **LLM:** {summary.llm_model}")
    lines.append(f"- **Embedding:** {summary.embed_model}")
    lines.append(f"- **Corpus:** {summary.corpus_size} documents indexed in {summary.index_time_seconds:.1f} seconds")
    lines.append(f"- **Corpus mode:** {'small-corpus (5 docs/claim)' if summary.corpus_mode == 'small-corpus' else 'full'}")
    lines.append(f"- **Graph:** {summary.graph_nodes} nodes, {summary.graph_edges} edges")
    lines.append(f"- **Claims evaluated:** {summary.total_claims} (completed: {summary.completed}, errors: {summary.errored})")
    lines.append(f"- **Timestamp:** {datetime.now().isoformat()}")
    lines.append("")

    # Summary Metrics
    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Verdict Accuracy | {summary.verdict_accuracy:.3f} |")
    lines.append(f"| Evidence Doc Recall | {summary.mean_evidence_recall:.3f} |")
    lines.append(f"| Mean Confidence | {summary.mean_confidence:.3f} |")
    lines.append("")

    # Two-Stage Failure Analysis
    if summary.two_stage_success_rate > 0 or summary.two_stage_retrieval_failure_rate > 0 or summary.two_stage_reasoning_failure_rate > 0:
        lines.append("## Two-Stage Failure Analysis\n")
        lines.append("| Category | Count | Rate |")
        lines.append("|----------|-------|------|")
        n = summary.completed
        success_n = int(summary.two_stage_success_rate * n + 0.5)
        retr_n = int(summary.two_stage_retrieval_failure_rate * n + 0.5)
        reas_n = int(summary.two_stage_reasoning_failure_rate * n + 0.5)
        lines.append(f"| SUCCESS | {success_n} | {summary.two_stage_success_rate:.3f} |")
        lines.append(f"| RETRIEVAL_FAILURE | {retr_n} | {summary.two_stage_retrieval_failure_rate:.3f} |")
        lines.append(f"| — LIBRARIAN_FAILURE | {summary.two_stage_librarian_failure_count} | - |")
        lines.append(f"| — PLANNER_FAILURE | {summary.two_stage_planner_failure_count} | - |")
        lines.append(f"| REASONING_FAILURE | {reas_n} | {summary.two_stage_reasoning_failure_rate:.3f} |")
        lines.append(f"| Graph Recall@K (mean) | - | {summary.two_stage_graph_recall_mean:.3f} |")
        lines.append(f"| Verdict Acc (when retrieved) | - | {summary.two_stage_verdict_acc_when_retrieved:.3f} |")
        lines.append(f"\n*Based on {summary.two_stage_claims_with_evidence} claims with expected evidence documents.*\n")
        lines.append("")

    # Per-Verdict Breakdown
    lines.append("## Per-Verdict Breakdown\n")
    lines.append("| Expected | Count | Correct | Accuracy |")
    lines.append("|----------|-------|---------|----------|")
    for v in SciFactVerdict:
        info = summary.per_verdict_accuracy.get(v.value, {})
        count = info.get("count", 0)
        correct = info.get("correct", 0)
        acc = info.get("accuracy", 0.0)
        lines.append(f"| {v.value} | {count} | {correct} | {acc:.3f} |")
    lines.append("")

    # Confusion Matrix
    verdicts = [v.value for v in SciFactVerdict]
    lines.append("## Confusion Matrix\n")
    lines.append("Rows = Expected, Columns = Predicted\n")
    header = "| Expected \\ Predicted | " + " | ".join(verdicts) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(verdicts) + 1))
    for expected in verdicts:
        row_data = summary.confusion_matrix.get(expected, {})
        cells = [str(row_data.get(pred, 0)) for pred in verdicts]
        lines.append(f"| {expected} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-Claim Results
    lines.append("## Per-Claim Results\n")
    lines.append("| ID | Claim (60 chars) | Expected | Predicted | OK | Conf | Ev.Recall | Lane |")
    lines.append("|-----|------------------|----------|-----------|-----|------|-----------|------|")
    for r in summary.claim_results:
        if r.error is not None:
            claim_short = r.claim_text[:57] + "..." if len(r.claim_text) > 60 else r.claim_text
            lines.append(
                f"| {r.claim_id} | {claim_short} | {r.expected_verdict.value} "
                f"| ERROR | - | - | - | - |"
            )
        else:
            claim_short = r.claim_text[:57] + "..." if len(r.claim_text) > 60 else r.claim_text
            check = "\u2713" if r.correct else "\u2717"
            pred = r.mapped_verdict.value if r.mapped_verdict else "?"
            lines.append(
                f"| {r.claim_id} | {claim_short} | {r.expected_verdict.value} "
                f"| {pred} | {check} | {r.confidence:.2f} "
                f"| {r.evidence_doc_recall:.2f} | {r.lane_used} |"
            )
    lines.append("")

    # Error Log
    errors = [r for r in summary.claim_results if r.error is not None]
    if errors:
        lines.append("## Error Log\n")
        for r in errors:
            lines.append(f"- **Claim {r.claim_id}:** {r.error}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# INDEX DIAGNOSTICS
# ============================================================================

def collect_index_diagnostics(
    orchestrator: OrchestratorV5,
    index: Any,
    corpus_size: int,
) -> IndexDiagnostics:
    """Collect diagnostic data from Chunker, Extractor, Explorer, and Librarian outputs."""
    diag = IndexDiagnostics()

    # --- Chunker trace ---
    chunker_out = orchestrator.last_chunker_output
    if chunker_out and chunker_out.trace:
        ct = chunker_out.trace
        diag.chunker_total_chunks = ct.total_chunks
        diag.chunker_doc_count = len(ct.input_doc_ids)
        if ct.chunks:  # List[ChunkSummary]
            tokens = [c.token_count for c in ct.chunks]
            if tokens:
                diag.chunker_token_stats = {
                    "min": min(tokens),
                    "max": max(tokens),
                    "mean": sum(tokens) / len(tokens),
                    "median": sorted(tokens)[len(tokens) // 2],
                }
            diag.chunker_section_headers_detected = sum(
                1 for c in ct.chunks if c.section_header
            )
            # Chunks per doc — top 10 by count
            cpc = Counter(c.source_doc_id for c in ct.chunks)
            diag.chunker_chunks_per_doc = dict(cpc.most_common(10))

    # --- Extractor trace ---
    explorer_out = orchestrator.last_explorer_output
    if explorer_out and explorer_out.trace:
        et = explorer_out.trace  # ExtractorTrace
        diag.extractor_entities_extracted = et.entities_extracted
        diag.extractor_claims_extracted = et.claims_extracted
        diag.extractor_gleaning_iterations = et.gleaning_iterations
        diag.extractor_llm_calls = et.llm_calls
        diag.extractor_time_seconds = et.extraction_time_seconds
        if et.chunk_details:
            diag.extractor_chunks_with_zero_entities = sum(
                1 for d in et.chunk_details if len(d.entities) == 0
            )
            diag.extractor_chunks_with_zero_claims = sum(
                1 for d in et.chunk_details if len(d.claims) == 0
            )
            diag.extractor_mean_entities_per_chunk = (
                sum(len(d.entities) for d in et.chunk_details) / len(et.chunk_details)
            )
            diag.extractor_mean_claims_per_chunk = (
                sum(len(d.claims) for d in et.chunk_details) / len(et.chunk_details)
            )

    # Check for hidden error in extraction_config
    if explorer_out and explorer_out.node_set:
        ns = explorer_out.node_set
        if ns.extraction_config and isinstance(ns.extraction_config, dict):
            diag.extractor_error = ns.extraction_config.get("error")

    # --- Explorer diagnostics (node_set level) ---
    if explorer_out and explorer_out.node_set:
        ns = explorer_out.node_set
        diag.explorer_chunk_count = ns.chunk_count
        diag.explorer_entity_count = ns.entity_count
        diag.explorer_claim_count = ns.claim_count
        diag.explorer_node_count = len(ns.nodes)
        diag.explorer_source_doc_count = len(ns.source_doc_ids)
        diag.explorer_avg_claims_per_chunk = ns.avg_claims_per_chunk or 0.0
        if ns.coreference_groups:
            diag.explorer_coref_groups = len(ns.coreference_groups)
        # Check for hidden error in extraction_config (Explorer catch-all)
        if ns.extraction_config and "error" in ns.extraction_config:
            diag.explorer_error = ns.extraction_config["error"]

    # --- Librarian / graph diagnostics ---
    graph = index.graph
    diag.graph_node_count = len(graph.nodes)
    diag.graph_edge_count = len(graph.edges)
    diag.graph_document_count = graph.document_count
    diag.graph_bridge_count = graph.bridge_count

    # Edge type distribution
    edge_types: Dict[str, int] = Counter()
    for edge in graph.edges.values():
        edge_types[edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type)] += 1
    diag.edge_type_distribution = dict(edge_types)

    # Node type distribution
    node_types: Dict[str, int] = Counter()
    for node in graph.nodes.values():
        ntype = node.node_type if isinstance(node.node_type, str) else str(node.node_type)
        node_types[ntype] += 1
    diag.node_type_distribution = dict(node_types)

    # Coverage: docs with at least one node
    if graph.node_by_doc:
        diag.docs_with_nodes = len(graph.node_by_doc)
        diag.docs_without_nodes = corpus_size - diag.docs_with_nodes

    # Community structure
    if index.community_structure:
        diag.community_count = len(index.community_structure.communities)

    # --- Coreference analysis ---
    explorer_out_coref = orchestrator.last_explorer_output
    if explorer_out_coref and explorer_out_coref.node_set:
        ns = explorer_out_coref.node_set
        if ns.coreference_groups:
            for cg in ns.coreference_groups:
                diag.coref_group_details.append({
                    "canonical_form": cg.canonical_form,
                    "variant_count": len(cg.variants),
                    "variants": cg.variants,
                })

            # Multi-doc coref groups: canonical form maps to nodes from 2+ docs
            # Build canonical_form → set of source_doc_ids
            canonical_to_docs: Dict[str, set] = defaultdict(set)
            for node in ns.nodes:
                canonical_to_docs[node.canonical_text].add(node.source_doc_id)
            coref_canonicals = {cg.canonical_form for cg in ns.coreference_groups}
            diag.coref_multi_doc_groups = sum(
                1 for cf in coref_canonicals
                if len(canonical_to_docs.get(cf, set())) > 1
            )

        # --- Cross-document entity overlap ---
        # Group nodes by canonical_text, count distinct source_doc_ids
        entity_docs: Dict[str, set] = defaultdict(set)
        for node in ns.nodes:
            entity_docs[node.canonical_text].add(node.source_doc_id)
        diag.unique_entities_total = len(entity_docs)
        for entity, doc_ids in entity_docs.items():
            if len(doc_ids) >= 2:
                diag.cross_doc_entity_overlap.append({
                    "entity": entity,
                    "doc_count": len(doc_ids),
                    "doc_ids": sorted(doc_ids),
                })
        diag.cross_doc_overlap_entity_count = len(diag.cross_doc_entity_overlap)

    return diag


def log_index_diagnostics(diag: IndexDiagnostics) -> None:
    """Print index diagnostics to stdout."""
    _print(f"\n  --- Chunker Diagnostics ---")
    _print(f"    Total chunks:          {diag.chunker_total_chunks}")
    _print(f"    Documents chunked:     {diag.chunker_doc_count}")
    if diag.chunker_token_stats:
        ts = diag.chunker_token_stats
        _print(f"    Token stats:           min={ts.get('min', 0):.0f} max={ts.get('max', 0):.0f} "
               f"mean={ts.get('mean', 0):.0f} median={ts.get('median', 0):.0f}")
    _print(f"    Section headers:       {diag.chunker_section_headers_detected} chunks")

    _print(f"\n  --- Extractor Diagnostics ---")
    if diag.extractor_error:
        _print(f"    *** EXTRACTOR ERROR: {diag.extractor_error} ***")
    _print(f"    Entities extracted:    {diag.extractor_entities_extracted}")
    _print(f"    Claims extracted:      {diag.extractor_claims_extracted}")
    _print(f"    Gleaning iterations:   {diag.extractor_gleaning_iterations}")
    _print(f"    LLM calls:            {diag.extractor_llm_calls}")
    _print(f"    Extraction time:       {diag.extractor_time_seconds:.1f}s")
    _print(f"    Chunks with 0 entities: {diag.extractor_chunks_with_zero_entities}")
    _print(f"    Chunks with 0 claims:   {diag.extractor_chunks_with_zero_claims}")
    _print(f"    Mean entities/chunk:    {diag.extractor_mean_entities_per_chunk:.2f}")
    _print(f"    Mean claims/chunk:      {diag.extractor_mean_claims_per_chunk:.2f}")

    _print(f"\n  --- Explorer Diagnostics ---")
    if diag.explorer_error:
        _print(f"    *** EXPLORER ERROR: {diag.explorer_error} ***")
    _print(f"    Chunks produced:       {diag.explorer_chunk_count}")
    _print(f"    Entities extracted:    {diag.explorer_entity_count}")
    _print(f"    Claims extracted:      {diag.explorer_claim_count}")
    _print(f"    Nodes built:           {diag.explorer_node_count}")
    _print(f"    Source docs covered:   {diag.explorer_source_doc_count}")
    _print(f"    Coref groups:          {diag.explorer_coref_groups}")
    _print(f"    Avg claims/chunk:      {diag.explorer_avg_claims_per_chunk:.2f}")

    _print(f"\n  --- Librarian Diagnostics ---")
    _print(f"    Graph nodes:           {diag.graph_node_count}")
    _print(f"    Graph edges:           {diag.graph_edge_count}")
    _print(f"    Documents in graph:    {diag.graph_document_count}")
    _print(f"    Bridge edges:          {diag.graph_bridge_count}")
    _print(f"    Communities:           {diag.community_count}")
    _print(f"    Docs with nodes:       {diag.docs_with_nodes}")
    _print(f"    Docs without nodes:    {diag.docs_without_nodes}")

    if diag.edge_type_distribution:
        _print(f"    Edge types:")
        for etype, count in sorted(diag.edge_type_distribution.items(), key=lambda x: -x[1]):
            _print(f"      {etype}: {count}")

    if diag.node_type_distribution:
        _print(f"    Node types:")
        for ntype, count in sorted(diag.node_type_distribution.items(), key=lambda x: -x[1]):
            _print(f"      {ntype}: {count}")

    _print(f"\n  --- Coreference Analysis ---")
    _print(f"    Total coref groups:    {len(diag.coref_group_details)}")
    _print(f"    Multi-doc groups:      {diag.coref_multi_doc_groups}")
    if diag.unique_entities_total > 0:
        pct = diag.cross_doc_overlap_entity_count / diag.unique_entities_total * 100
        _print(f"    Cross-doc entities:    {diag.cross_doc_overlap_entity_count} / "
               f"{diag.unique_entities_total} ({pct:.1f}%)")
    else:
        _print(f"    Cross-doc entities:    0 / 0 (0.0%)")


# ============================================================================
# DIAGNOSTICS JSON WRITER
# ============================================================================

def _serialize_chunk_details(chunk_details: list, trace_chunks: int) -> List[Dict[str, Any]]:
    """Serialize chunk extraction details with configurable limit.

    Parameters
    ----------
    chunk_details : list of ChunkExtractionDetail
    trace_chunks : int
        Number of chunk details to include. 0 = skip, -1 = all.
    """
    if trace_chunks == 0 or not chunk_details:
        return []
    items = chunk_details if trace_chunks < 0 else chunk_details[:trace_chunks]
    result = []
    for d in items:
        result.append({
            "chunk_id": d.chunk_id,
            "source_doc_id": d.source_doc_id,
            "token_count": d.token_count,
            "text_preview": d.text_preview,
            "entity_count": len(d.entities),
            "claim_count": len(d.claims),
            "gleaning_iterations": d.gleaning_iterations,
            "gleaning_new_entities": d.gleaning_new_entities,
            "gleaning_new_claims": d.gleaning_new_claims,
        })
    return result


def write_trace_json(
    summary: RunSummary,
    output_path: str,
    orchestrator: Optional[Any] = None,
    trace_chunks: int = 50,
    trace_level: TraceLevel = TraceLevel.STANDARD,
) -> None:
    """Write full trace data to a JSON file alongside output.md."""
    diag_path = output_path.rsplit(".", 1)[0] + ".trace.json"

    data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": summary.llm_model,
            "embed_model": summary.embed_model,
            "corpus_size": summary.corpus_size,
            "total_claims": summary.total_claims,
            "corpus_mode": summary.corpus_mode,
            "trace_level": trace_level.value,
        },
        "index": {
            "time_seconds": summary.index_time_seconds,
            "graph_nodes": summary.graph_nodes,
            "graph_edges": summary.graph_edges,
        },
        "metrics": {
            "verdict_accuracy": summary.verdict_accuracy,
            "mean_evidence_recall": summary.mean_evidence_recall,
            "mean_confidence": summary.mean_confidence,
            "completed": summary.completed,
            "errored": summary.errored,
        },
        "confusion_matrix": summary.confusion_matrix,
        "per_verdict_accuracy": summary.per_verdict_accuracy,
    }

    # Index diagnostics
    if summary.index_diagnostics:
        d = summary.index_diagnostics
        data["index_diagnostics"] = {
            "chunker": {
                "total_chunks": d.chunker_total_chunks,
                "doc_count": d.chunker_doc_count,
                "token_stats": d.chunker_token_stats,
                "section_headers_detected": d.chunker_section_headers_detected,
                "chunks_per_doc_top10": d.chunker_chunks_per_doc,
            },
            "extractor": {
                "entities_extracted": d.extractor_entities_extracted,
                "claims_extracted": d.extractor_claims_extracted,
                "gleaning_iterations": d.extractor_gleaning_iterations,
                "llm_calls": d.extractor_llm_calls,
                "time_seconds": d.extractor_time_seconds,
                "error": d.extractor_error,
                "chunks_with_zero_entities": d.extractor_chunks_with_zero_entities,
                "chunks_with_zero_claims": d.extractor_chunks_with_zero_claims,
                "mean_entities_per_chunk": d.extractor_mean_entities_per_chunk,
                "mean_claims_per_chunk": d.extractor_mean_claims_per_chunk,
            },
            "explorer": {
                "chunk_count": d.explorer_chunk_count,
                "entity_count": d.explorer_entity_count,
                "claim_count": d.explorer_claim_count,
                "node_count": d.explorer_node_count,
                "source_doc_count": d.explorer_source_doc_count,
                "coref_groups": d.explorer_coref_groups,
                "avg_claims_per_chunk": d.explorer_avg_claims_per_chunk,
                "error": d.explorer_error,
            },
            "librarian": {
                "graph_node_count": d.graph_node_count,
                "graph_edge_count": d.graph_edge_count,
                "graph_document_count": d.graph_document_count,
                "bridge_count": d.graph_bridge_count,
                "community_count": d.community_count,
                "docs_with_nodes": d.docs_with_nodes,
                "docs_without_nodes": d.docs_without_nodes,
                "edge_type_distribution": d.edge_type_distribution,
                "node_type_distribution": d.node_type_distribution,
            },
            "coreference_analysis": {
                "total_groups": len(d.coref_group_details),
                "multi_doc_groups": d.coref_multi_doc_groups,
                "group_details": d.coref_group_details,
            },
            "cross_doc_entity_overlap": {
                "unique_entities_total": d.unique_entities_total,
                "overlap_entity_count": d.cross_doc_overlap_entity_count,
                "overlap_ratio": (
                    d.cross_doc_overlap_entity_count / d.unique_entities_total
                    if d.unique_entities_total > 0 else 0.0
                ),
                "entities": d.cross_doc_entity_overlap,
            },
        }

    # Full chunker trace (per-chunk summaries)
    if orchestrator:
        chunker_out = orchestrator.last_chunker_output
        if chunker_out and chunker_out.trace:
            ct = chunker_out.trace
            chunk_summaries: List[Dict[str, Any]] = []
            if ct.chunks:
                limit = len(ct.chunks) if trace_chunks < 0 else min(trace_chunks, len(ct.chunks))
                for c in ct.chunks[:limit]:
                    chunk_summaries.append({
                        "chunk_id": c.chunk_id,
                        "source_doc_id": c.source_doc_id,
                        "token_count": c.token_count,
                        "position": c.position,
                        "section_header": c.section_header,
                        "text_preview": c.text_preview,
                    })
            data["chunker_trace"] = {
                "trace_id": ct.trace_id,
                "input_doc_ids_count": len(ct.input_doc_ids),
                "total_chunks": ct.total_chunks,
                "chunk_summaries": chunk_summaries,
                "chunk_summaries_included": len(chunk_summaries),
                "chunk_summaries_total": len(ct.chunks) if ct.chunks else 0,
            }

        # Full extractor trace (per-chunk extraction details)
        explorer_out = orchestrator.last_explorer_output
        if explorer_out and explorer_out.trace:
            et = explorer_out.trace
            data["extractor_trace"] = {
                "trace_id": et.trace_id,
                "input_chunk_count": et.input_chunk_count,
                "entities_extracted": et.entities_extracted,
                "claims_extracted": et.claims_extracted,
                "gleaning_iterations": et.gleaning_iterations,
                "llm_calls": et.llm_calls,
                "extraction_time_seconds": et.extraction_time_seconds,
                "chunk_details": _serialize_chunk_details(
                    et.chunk_details or [], trace_chunks
                ),
                "chunk_details_included": min(
                    trace_chunks if trace_chunks >= 0 else len(et.chunk_details or []),
                    len(et.chunk_details or []),
                ),
                "chunk_details_total": len(et.chunk_details) if et.chunk_details else 0,
            }

    # Per-claim traces
    claim_traces = []
    is_standard_plus = trace_level in (TraceLevel.STANDARD, TraceLevel.FULL)
    for r in summary.claim_results:
        planner_trace: Dict[str, Any] = {
            "seeds_found": r.trace_seeds_found,
            "scope_nodes": r.trace_scope_nodes,
            "scope_documents": r.trace_scope_documents,
            "selected_lane": r.trace_selected_lane,
            "intent": r.trace_intent,
        }
        reasoner_trace: Dict[str, Any] = {
            "docs_in_scope": r.trace_docs_in_scope,
            "relationships_inferred": r.trace_relationships_inferred,
            "evidence_grounded": r.trace_evidence_grounded,
            "llm_time_ms": r.trace_llm_time_ms,
            "total_time_ms": r.trace_total_time_ms,
        }

        if is_standard_plus:
            # Extend planner trace
            planner_trace["router_confidence"] = r.trace_router_confidence
            planner_trace["ppr_total_mass"] = r.trace_ppr_total_mass
            planner_trace["seed_details"] = r.trace_seed_details
            planner_trace["scope_node_details"] = r.trace_scope_node_details
            planner_trace["ranked_chunks"] = r.trace_ranked_chunks

            # Extend reasoner trace
            reasoner_trace["verdict_answer"] = r.trace_verdict_answer
            reasoner_trace["verdict_summary"] = r.trace_verdict_summary
            reasoner_trace["reasoning_trace"] = r.trace_reasoning_trace
            reasoner_trace["confidence_overall"] = r.trace_confidence_overall
            reasoner_trace["confidence_evidence_quality"] = r.trace_confidence_evidence_quality
            reasoner_trace["confidence_source_agreement"] = r.trace_confidence_source_agreement
            reasoner_trace["confidence_coverage"] = r.trace_confidence_coverage
            reasoner_trace["scope_document_ids"] = r.trace_scope_document_ids
            reasoner_trace["scope_bridge_edges_used"] = r.trace_scope_bridge_edges_used
            reasoner_trace["relationships_detail"] = r.trace_relationships_detail
            reasoner_trace["evidence_detail"] = r.trace_evidence_detail

            if trace_level == TraceLevel.FULL:
                reasoner_trace["raw_response"] = r.trace_raw_response

        entry: Dict[str, Any] = {
            "claim_id": r.claim_id,
            "claim_text": r.claim_text[:120],
            "expected": r.expected_verdict.value,
            "predicted_raw": r.predicted_verdict_type.value if r.predicted_verdict_type else None,
            "predicted_mapped": r.mapped_verdict.value if r.mapped_verdict else None,
            "correct": r.correct,
            "confidence": r.confidence,
            "lane": r.lane_used,
            "intent": r.intent,
            "evidence_doc_ids": r.evidence_doc_ids,
            "expected_doc_ids": r.expected_doc_ids,
            "evidence_doc_recall": r.evidence_doc_recall,
            "relationships": r.relationships,
            "planner_trace": planner_trace,
            "reasoner_trace": reasoner_trace,
            "error": r.error,
            "two_stage_failure": r.two_stage_failure,
            "retrieval_attribution": r.retrieval_attribution,
            "graph_recall_at_k": r.graph_recall_at_k,
            "gold_docs_in_scope": r.gold_docs_in_scope,
            "gold_docs_in_graph": r.gold_docs_in_graph,
            "gold_docs_total": r.gold_docs_total,
            "planner_scope_doc_ids": r.planner_scope_doc_ids,
            "planner_scope_recall": r.planner_scope_recall,
            "evidence_failure_attribution": r.evidence_failure_attribution,
        }
        claim_traces.append(entry)
    data["claim_traces"] = claim_traces

    with open(diag_path, "w") as f:
        json.dump(data, f, indent=2)
    _print(f"  Trace written to {diag_path}")


# ============================================================================
# CHECKPOINT WRITER
# ============================================================================

def write_checkpoint(results: List[ClaimResult], output_path: str) -> None:
    """Write partial results to a checkpoint JSON file."""
    checkpoint_path = output_path + ".checkpoint.json"
    data = []
    for r in results:
        data.append({
            "claim_id": r.claim_id,
            "expected": r.expected_verdict.value,
            "predicted": r.mapped_verdict.value if r.mapped_verdict else None,
            "predicted_raw": r.predicted_verdict_type.value if r.predicted_verdict_type else None,
            "correct": r.correct,
            "confidence": r.confidence,
            "lane": r.lane_used,
            "intent": r.intent,
            "evidence_doc_recall": r.evidence_doc_recall,
            "trace_seeds_found": r.trace_seeds_found,
            "trace_scope_nodes": r.trace_scope_nodes,
            "trace_scope_documents": r.trace_scope_documents,
            "trace_selected_lane": r.trace_selected_lane,
            "trace_intent": r.trace_intent,
            "trace_docs_in_scope": r.trace_docs_in_scope,
            "trace_total_time_ms": r.trace_total_time_ms,
            "trace_verdict_summary": r.trace_verdict_summary,
            "trace_confidence_overall": r.trace_confidence_overall,
            "trace_confidence_evidence_quality": r.trace_confidence_evidence_quality,
            "trace_confidence_source_agreement": r.trace_confidence_source_agreement,
            "trace_confidence_coverage": r.trace_confidence_coverage,
            "trace_router_confidence": r.trace_router_confidence,
            "trace_scope_bridge_edges_used": r.trace_scope_bridge_edges_used,
            "error": r.error,
            "two_stage_failure": r.two_stage_failure,
            "retrieval_attribution": r.retrieval_attribution,
            "graph_recall_at_k": r.graph_recall_at_k,
            "gold_docs_in_scope": r.gold_docs_in_scope,
            "gold_docs_total": r.gold_docs_total,
            "planner_scope_recall": r.planner_scope_recall,
        })
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main(
    limit: Optional[int] = None,
    output_path: str = "output.md",
    model: Optional[str] = None,
    embed_model: Optional[str] = None,
    use_mock: bool = False,
    trace_chunks: int = 50,
    small_corpus: bool = False,
    trace_level: str = "standard",
) -> None:
    """Run the full SciFact v5 evaluation."""
    from config.logging_config import setup_logging, log_file_from_output_path
    setup_logging(log_file=log_file_from_output_path(output_path))

    tl = TraceLevel(trace_level)

    _print("=" * 70)
    _print("SciFact v5 Full-Corpus Evaluation")
    _print("=" * 70)
    _print(f"Trace level: {tl.value}")

    # 1. Create clients
    if use_mock:
        from orchestrator import MockLLMClientV5, MockEmbeddingClientV5
        llm_client = MockLLMClientV5()
        embed_client = MockEmbeddingClientV5()
        llm_model_name = "MockLLMClientV5"
        embed_model_name = "MockEmbeddingClientV5"
        _print("Using MOCK clients")
    else:
        if model:
            os.environ["OLLAMA_MODEL"] = model
        if embed_model:
            os.environ["OLLAMA_EMBED_MODEL"] = embed_model
        llm_client = get_client("ollama")
        embed_client = get_embedding_client("ollama")
        llm_model_name = getattr(llm_client, "model", "ollama")
        embed_model_name = getattr(embed_client, "model", "ollama")
        _print(f"LLM: {llm_model_name}")
        _print(f"Embedding: {embed_model_name}")

    # 2. Create orchestrator
    config: Dict[str, Any] = {
        "explorer": {},
        "librarian": {},
        "planner": {},
        "reasoner": {},
    }
    orchestrator = OrchestratorV5(llm_client, embed_client, config)

    # 3. Load SciFact data
    _print("\nLoading SciFact dataset...")
    loader = SciFactLoader()
    test_cases = loader.get_test_cases(limit=limit)
    _print(f"  Claims: {len(test_cases)} test cases" +
           (f" (limited to {limit})" if limit else ""))

    if small_corpus:
        if limit is None:
            _print("WARNING: --small-corpus requires --limit N. Falling back to full corpus.")
            all_documents = loader.get_all_abstracts()
        else:
            target = limit * 5
            all_documents = build_small_corpus(loader, test_cases, target)
            _print(f"  Small corpus: {len(all_documents)} documents "
                   f"(target: {target}, evidence docs included)")
    else:
        all_documents = loader.get_all_abstracts()

    _print(f"  Corpus: {len(all_documents)} documents")

    # Log evidence coverage
    evidence_ids: set = set()
    for case in test_cases:
        evidence_ids.update(case.expected_evidence.keys())
    if small_corpus and limit is not None:
        included_ids = {d.doc_id for d in all_documents}
        missing = evidence_ids - included_ids
        if missing:
            _print(f"  WARNING: {len(missing)} evidence docs missing from corpus: {missing}")
        else:
            _print(f"  All {len(evidence_ids)} evidence docs present in corpus")

    # 4. Index corpus (once)
    _print(f"\nIndexing {len(all_documents)} documents...")
    t0 = time.time()
    index = orchestrator.index_corpus_v5(all_documents)
    index_time = time.time() - t0

    graph_nodes = len(index.graph.nodes)
    graph_edges = len(index.graph.edges)
    comm_count = len(index.community_structure.communities) if index.community_structure else 0
    _print(f"  Index complete in {index_time:.1f}s")
    _print(f"  Graph: {graph_nodes} nodes, {graph_edges} edges, {comm_count} communities")

    # 4b. Collect and log index diagnostics
    index_diag = collect_index_diagnostics(orchestrator, index, len(all_documents))
    log_index_diagnostics(index_diag)

    # 5. Process each claim
    _print(f"\nProcessing {len(test_cases)} claims...\n")
    results: List[ClaimResult] = []

    try:
        for i, case in enumerate(test_cases, 1):
            result = run_single_claim(orchestrator, index, case, i, len(test_cases), tl)
            results.append(result)

            # Checkpoint every 10 claims
            if i % 10 == 0:
                write_checkpoint(results, output_path)
                _print(f"  (checkpoint written: {i}/{len(test_cases)} claims)")
    except KeyboardInterrupt:
        _print(f"\n\nInterrupted after {len(results)} claims. Writing partial results...")
    finally:
        # 6. Build summary and write output
        summary = build_summary(
            results=results,
            index_time=index_time,
            corpus_size=len(all_documents),
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            llm_model=llm_model_name,
            embed_model=embed_model_name,
        )
        summary.corpus_mode = "small-corpus" if small_corpus else "full"
        summary.index_diagnostics = index_diag

        md = format_output_md(summary)
        with open(output_path, "w") as f:
            f.write(md)
        _print(f"\nResults written to {output_path}")

        # Write machine-readable trace JSON
        write_trace_json(summary, output_path, orchestrator, trace_chunks, tl)

        # Also clean up checkpoint if all claims completed
        checkpoint_path = output_path + ".checkpoint.json"
        if len(results) == len(test_cases) and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Print summary to stdout
        _print(f"\n{'=' * 70}")
        _print(f"SUMMARY")
        _print(f"{'=' * 70}")
        _print(f"  Claims:           {summary.total_claims} ({summary.completed} completed, {summary.errored} errors)")
        _print(f"  Verdict Accuracy: {summary.verdict_accuracy:.3f}")
        _print(f"  Evidence Recall:  {summary.mean_evidence_recall:.3f}")
        _print(f"  Mean Confidence:  {summary.mean_confidence:.3f}")
        _print(f"  Index Time:       {summary.index_time_seconds:.1f}s")
        _print(f"{'=' * 70}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SciFact v5 Full-Corpus Evaluation",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of claims to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="output.md",
        help="Output markdown file path (default: output.md)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama LLM model name (default: from OLLAMA_MODEL env or qwen2.5:14b)",
    )
    parser.add_argument(
        "--embed-model", type=str, default=None,
        help="Ollama embedding model name (default: from OLLAMA_EMBED_MODEL env or nomic-embed-text)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM/embedding clients for smoke testing",
    )
    parser.add_argument(
        "--trace-chunks", type=int, default=50,
        help="Number of chunk extraction details in diagnostics JSON (0=skip, -1=all, default: 50)",
    )
    parser.add_argument(
        "--small-corpus", action="store_true",
        help="Index only limit*5 documents (evidence docs + distractors) instead of full corpus",
    )
    parser.add_argument(
        "--trace-level", type=str, default="standard",
        choices=["minimal", "standard", "full"],
        help="Diagnostic verbosity: minimal (legacy), standard (default), full (raw LLM responses)",
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        output_path=args.output,
        model=args.model,
        embed_model=args.embed_model,
        use_mock=args.mock,
        trace_chunks=args.trace_chunks,
        small_corpus=args.small_corpus,
        trace_level=args.trace_level,
    )
