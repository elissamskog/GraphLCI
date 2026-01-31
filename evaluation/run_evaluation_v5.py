"""
Unified v5 Evaluation Runner
Location: evaluation/run_evaluation_v5.py
Owner: EVALUATOR Instance

Single entry-point for running SciFact or WikiFactClaims evaluation with
the v5 pipeline.  Supersedes run_scifact_v5.py and run_wikifactclaims_v5.py
(those files remain as deprecated but functional backward-compat runners).

Usage:
    python -m evaluation.run_evaluation_v5 --dataset scifact --limit 3 --mock
    python -m evaluation.run_evaluation_v5 --dataset wikifactclaims --limit 5 --mock
    python -m evaluation.run_evaluation_v5 --dataset scifact --model qwen2.5:14b \
        --embed-model nomic-embed-text --small-corpus --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings

# --- Global Noise Suppression (Must trigger before other imports) ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

# Filter MLX/PyTorch specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mx.metal")
warnings.filterwarnings("ignore", message=".*mx.metal.device_info is deprecated.*")

try:
    import tqdm
    # Env var TQDM_DISABLE=1 (set above) is usually sufficient
    pass
except ImportError:
    pass

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# -- sys.path setup ----------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_PLANNER_DIR = os.path.join(_PROJECT_ROOT, "planner")
if _PLANNER_DIR not in sys.path:
    sys.path.insert(0, _PLANNER_DIR)

from orchestrator import OrchestratorV5
from llm_client import get_client, get_embedding_client
from evaluation.adapters import DatasetAdapter, get_adapter, build_small_corpus
from evaluation.diagnostics import IndexDiagnostics, collect_index_diagnostics, log_index_diagnostics
from evaluation.output_formatters import format_output_md, write_trace_json, write_checkpoint
from schemas.contracts import (
    VerdictType,
    QueryRequest,
    ReasonerOutput,
    TraceLevel,
    StructuralEdgeType,
)
from schemas.evaluation_contracts import (
    SciFactVerdict,
    EvalClaim,
    EvalGroup,
    ClaimTraceData,
    UnifiedClaimResult,
    GroupResult,
    UnifiedRunSummary,
)
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
# DATASET ADAPTERS
# ============================================================================




# ============================================================================
# EVIDENCE DOC RECALL (SciFact)
# ============================================================================

def compute_evidence_doc_recall(
    finding: Any,
    expected_evidence: Dict[str, List[int]],
) -> float:
    """Compute document-level recall of evidence.

    Returns len(found_doc_ids & expected_doc_ids) / len(expected_doc_ids).
    """
    if not expected_evidence:
        return 1.0

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
    claim: EvalClaim,
    group_id: str,
    n: int,
    total: int,
    trace_level: TraceLevel,
    adapter: DatasetAdapter,
) -> UnifiedClaimResult:
    """Process a single claim through the v5 pipeline."""
    result = UnifiedClaimResult(
        claim_id=claim.claim_id,
        group_id=group_id,
        claim_text=claim.claim_text,
        expected_verdict=claim.expected_verdict,
        expected_doc_ids=claim.gold_doc_ids,
        metadata=dict(claim.metadata),
    )
    trace = result.trace

    try:
        query = QueryRequest(
            query_id=f"{adapter.name}-v5-{claim.claim_id}",
            query_text=claim.claim_text,
        )

        reasoner_output: ReasonerOutput = orchestrator.process_query_v5(query, index)
        finding = reasoner_output.finding

        # Core fields
        result.predicted_verdict_type = finding.verdict.verdict_type
        result.mapped_verdict = map_verdict(finding.verdict.verdict_type)
        result.correct = result.mapped_verdict == claim.expected_verdict
        result.confidence = finding.verdict.confidence
        result.lane_used = finding.lane_used
        result.search_strategy = finding.search_strategy
        result.evidence_doc_ids = [
            str(ev.source_document) for ev in finding.evidence if ev.source_document
        ]
        result.relationships = [
            rel.relationship.value for rel in finding.relationships
        ]

        # Evidence doc recall (SciFact / MuSiQue)
        if adapter.has_evidence_doc_recall:
            # Fallback to gold_doc_ids if expected_evidence (sentence-level) is empty
            target_evidence = claim.expected_evidence
            if not target_evidence and claim.gold_doc_ids:
                target_evidence = {did: [] for did in claim.gold_doc_ids}

            if target_evidence:
                result.evidence_doc_recall = compute_evidence_doc_recall(
                    finding, target_evidence,
                )

        # Planner trace
        planner_out = orchestrator.last_planner_output
        if planner_out:
            rd = planner_out.router_decision
            trace.seeds_found = len(rd.seed_nodes)
            trace.selected_lane = rd.selected_lane.value if rd.selected_lane else ""
            trace.search_strategy = (
                rd.search_strategy.classification.value
                if rd.search_strategy and rd.search_strategy.classification else ""
            )
            trace.router_confidence = rd.confidence
            if rd.scope:
                trace.scope_nodes = len(rd.scope.nodes)
                trace.scope_documents = len(rd.scope.document_ids)
                trace.planner_scope_doc_ids = list(rd.scope.document_ids)
                trace.ppr_total_mass = rd.scope.total_ppr_mass

        # Scope recall (always — gold_doc_ids set by both adapters)
        gold_doc_ids = claim.gold_doc_ids
        if gold_doc_ids:
            scope_set = set(trace.planner_scope_doc_ids)
            expected_set = set(gold_doc_ids)
            result.scope_recall = len(scope_set & expected_set) / len(expected_set)
            trace.planner_scope_recall = result.scope_recall
        else:
            result.scope_recall = 1.0
            trace.planner_scope_recall = 1.0

        # Two-stage failure isolation (SciFact)
        if adapter.has_two_stage and gold_doc_ids:
            try:
                two_stage = evaluate_two_stage(
                    case_id=claim.claim_id,
                    gold_doc_ids=gold_doc_ids,
                    graph=index.graph,
                    planner_output=orchestrator.last_planner_output,
                    reasoner_output=reasoner_output,
                    expected_verdict=claim.expected_verdict.value,
                    verdict_mapper=lambda vt: map_verdict(vt).value,
                )
                trace.two_stage_failure = two_stage.failure_stage.value
                trace.retrieval_attribution = two_stage.retrieval_attribution.value
                trace.graph_recall_at_k = two_stage.stage_a.graph_recall_at_k
                trace.gold_docs_in_scope = sum(
                    1 for dr in two_stage.stage_a.per_doc_results if dr.in_scope
                )
                trace.gold_docs_total = len(gold_doc_ids)
                trace.gold_docs_in_graph = sum(
                    1 for dr in two_stage.stage_a.per_doc_results if dr.in_graph
                )
            except Exception:
                _logger.debug(
                    "Two-stage evaluation failed for claim %s",
                    claim.claim_id,
                    exc_info=True,
                )

            # Per-evidence-doc failure attribution
            grounded_doc_ids = set(result.evidence_doc_ids)
            scope_set = set(trace.planner_scope_doc_ids)
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

                trace.evidence_failure_attribution.append({
                    "doc_id": doc_id,
                    "in_graph": in_graph,
                    "in_scope": in_scope,
                    "in_grounded": in_grounded,
                    "status": status,
                })

        # Configure logging
        log_level = logging.DEBUG if trace_level == "verbose" else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        
        # Suppress noisy third-party loggers
        for lib in ["httpcore", "httpx", "huggingface_hub", "urllib3", "sentence_transformers", "transformers", "pdfminer"]:
            logging.getLogger(lib).setLevel(logging.WARNING)
        # Reasoner trace
        rt = reasoner_output.trace
        trace.docs_in_scope = rt.documents_in_scope
        trace.relationships_inferred = rt.relationships_inferred
        trace.evidence_grounded = rt.evidence_grounded
        trace.llm_time_ms = rt.llm_time_ms
        trace.total_time_ms = rt.total_time_ms

        # Per-stage timing
        trace.stage_timing_planner_ms = orchestrator.stage_timings.get("planner", 0.0)
        trace.stage_timing_reasoner_ms = orchestrator.stage_timings.get("reasoner", 0.0)
        trace.stage_timing_total_ms = orchestrator.stage_timings.get("query_total", 0.0)

        # Token usage
        reasoner_usage = orchestrator.stage_token_usage.get("reasoner", {})
        trace.reasoner_prompt_tokens = reasoner_usage.get("prompt_tokens", 0)
        trace.reasoner_completion_tokens = reasoner_usage.get("completion_tokens", 0)
        trace.reasoner_total_tokens = reasoner_usage.get("total_tokens", 0)
        planner_usage = orchestrator.stage_token_usage.get("planner_embed", {})
        trace.planner_prompt_tokens = planner_usage.get("prompt_tokens", 0)
        trace.planner_total_tokens = planner_usage.get("total_tokens", 0)

        # Final prompt (if REASONER populated it)
        if trace_level == TraceLevel.FULL:
            if rt.final_prompt is not None:
                trace.final_prompt = rt.final_prompt
                trace.prompt_token_count = rt.prompt_token_count

        # Standard+ trace level: capture rich detail
        if trace_level in (TraceLevel.STANDARD, TraceLevel.FULL):
            is_full = trace_level == TraceLevel.FULL

            # Reasoner detail from finding
            trace.verdict_answer = finding.verdict.answer or ""
            trace.verdict_summary = finding.verdict.summary or ""
            trace.reasoning_trace = finding.reasoning_trace or ""

            # Confidence factors
            cf = finding.confidence_factors
            trace.confidence_overall = cf.overall
            trace.confidence_evidence_quality = cf.evidence_quality
            trace.confidence_source_agreement = cf.source_agreement
            trace.confidence_coverage = cf.coverage

            # Scope from finding
            trace.scope_document_ids = list(finding.scope.document_ids)
            trace.scope_bridge_edges_used = finding.scope.bridge_edges_used

            # Relationships detail
            rel_limit = len(finding.relationships) if is_full else 20
            for rel in finding.relationships[:rel_limit]:
                trace.relationships_detail.append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relationship": rel.relationship.value,
                    "confidence": rel.confidence,
                    "reasoning": rel.reasoning[:120] if rel.reasoning else "",
                    "evidence_quote": (rel.evidence_quote or "")[:120],
                })

            # Evidence detail
            for ev in finding.evidence:
                trace.evidence_detail.append({
                    "claim": (ev.claim or "")[:120],
                    "source_document": ev.source_document,
                    "source_chunk": ev.source_chunk,
                    "exact_quote": (ev.exact_quote or "")[:120],
                })

            # Planner detail
            if planner_out:
                rd = planner_out.router_decision

                seed_limit = len(rd.seed_nodes) if is_full else 10
                for sn in rd.seed_nodes[:seed_limit]:
                    trace.seed_details.append({
                        "node_id": sn.node_id,
                        "similarity": sn.similarity,
                        "node_type": sn.node_type,
                        "canonical_text": sn.canonical_text[:80],
                        "source_doc_id": sn.source_doc_id,
                    })

                if rd.scope:
                    sorted_scope = sorted(
                        rd.scope.nodes.values(),
                        key=lambda sn: sn.ppr_score,
                        reverse=True,
                    )
                    scope_limit = len(sorted_scope) if is_full else 15
                    for sn in sorted_scope[:scope_limit]:
                        trace.scope_node_details.append({
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
                        trace.ranked_chunks.append({
                            "chunk_id": getattr(rc.chunk, "chunk_id", ""),
                            "source_doc_id": getattr(rc.chunk, "source_doc_id", ""),
                            "relevance_score": rc.relevance_score,
                            "token_count": getattr(rc.chunk, "token_count", 0),
                            "text_preview": getattr(rc.chunk, "text", "")[:200],
                        })

            # Full level: raw LLM response
            if is_full:
                trace.raw_response = reasoner_output.raw_response or ""

        # Per-claim diagnostic output
        check = "\u2713" if result.correct else "\u2717"
        _print(
            f"  [{n}/{total}] claim={claim.claim_id} "
            f"{finding.verdict.verdict_type.value}\u2192{result.mapped_verdict.value} "
            f"(exp: {claim.expected_verdict.value}) {check}"
        )
        _print(
            f"        conf={result.confidence:.2f} lane={result.lane_used} "
            f"scope={trace.scope_nodes}nodes/{trace.scope_documents}docs "
            f"seeds={trace.seeds_found} ppr={trace.ppr_total_mass:.2f}"
        )
        _print(
            f"        reasoner: docs={trace.docs_in_scope} "
            f"rels={trace.relationships_inferred} "
            f"ev={trace.evidence_grounded} llm={trace.llm_time_ms:.0f}ms"
        )
        _print(
            f"        confidence: overall={trace.confidence_overall:.2f} "
            f"ev_quality={trace.confidence_evidence_quality:.2f} "
            f"agreement={trace.confidence_source_agreement:.2f} "
            f"coverage={trace.confidence_coverage:.2f}"
        )
        if planner_out and planner_out.router_decision.claim_decomposition:
            decomp = planner_out.router_decision.claim_decomposition
            if decomp.should_decompose:
                sub_txt = ", ".join([f"'{s.objective}'" for s in decomp.subclaims])
                _print(f"        planner: decomp=TRUE subclaims=[{sub_txt}]")
        
        if planner_out and planner_out.router_decision.seed_nodes:
            seeds_txt = ", ".join([f"'{s.canonical_text}'" for s in planner_out.router_decision.seed_nodes[:5]])
            if len(planner_out.router_decision.seed_nodes) > 5:
                seeds_txt += " ..."
            _print(f"        planner: seeds=[{seeds_txt}]")

        if trace.reasoner_total_tokens > 0 or trace.planner_total_tokens > 0:
            _print(
                f"        tokens: reasoner={trace.reasoner_total_tokens} "
                f"(p={trace.reasoner_prompt_tokens} c={trace.reasoner_completion_tokens}) "
                f"planner_embed={trace.planner_total_tokens} "
                f"timing: planner={trace.stage_timing_planner_ms:.0f}ms "
                f"reasoner={trace.stage_timing_reasoner_ms:.0f}ms "
                f"total={trace.stage_timing_total_ms:.0f}ms"
            )

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        _print(f"  [{n}/{total}] claim={claim.claim_id} ERROR: {result.error}")

    return result


# ============================================================================
# GROUP RUNNER
# ============================================================================

def run_group(
    orchestrator: OrchestratorV5,
    group: EvalGroup,
    group_num: int,
    total_groups: int,
    trace_level: TraceLevel,
    adapter: DatasetAdapter,
) -> GroupResult:
    """Index one group's documents, then run all claims against that index."""
    meta = group.metadata
    _print(f"\n--- Group {group_num}/{total_groups}  id={group.group_id}"
           + (f"  type={meta.get('question_type', '')}" if 'question_type' in meta else "")
           + (f"  level={meta.get('level', '')}" if 'level' in meta else "")
           + " ---")
    if 'question' in meta:
        _print(f"  Q: {meta['question'][:100]}")
    _print(f"  Documents: {len(group.documents)}")

    # Index
    t0 = time.time()
    index = orchestrator.index_corpus_v5(group.documents)
    index_time = time.time() - t0

    graph_nodes = len(index.graph.nodes)
    graph_edges = len(index.graph.edges)
    graph_bridge_count = len(index.graph.bridge_edges) if index.graph.bridge_edges else 0
    graph_community_count = (
        len(index.community_structure.communities)
        if index.community_structure else 0
    )

    # Extractor stats
    explorer_out = orchestrator.last_explorer_output
    ext_entities = 0
    ext_claims = 0
    ext_llm_calls = 0
    ext_time = 0.0
    if explorer_out and explorer_out.trace:
        ext_entities = explorer_out.trace.entities_extracted
        ext_claims = explorer_out.trace.claims_extracted
        ext_llm_calls = explorer_out.trace.llm_calls
        ext_time = explorer_out.trace.extraction_time_seconds

    _print(f"  Index: {graph_nodes} nodes, {graph_edges} edges, "
           f"{graph_bridge_count} bridges, {graph_community_count} communities "
           f"({index_time:.1f}s)")
    _print(f"  Extractor: {ext_entities} entities, {ext_claims} claims, "
           f"{ext_llm_calls} LLM calls ({ext_time:.1f}s)")

    gr = GroupResult(
        group_id=group.group_id,
        index_time_seconds=index_time,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        graph_bridge_count=graph_bridge_count,
        graph_community_count=graph_community_count,
        corpus_size=len(group.documents),
        extractor_entities=ext_entities,
        extractor_claims=ext_claims,
        extractor_llm_calls=ext_llm_calls,
        extractor_time_seconds=ext_time,
        metadata=dict(group.metadata),
    )

    # Collect index diagnostics (SciFact)
    if adapter.has_index_diagnostics:
        index_diag = collect_index_diagnostics(orchestrator, index, len(group.documents))
        gr.index_diagnostics = index_diag
        log_index_diagnostics(index_diag)

    # Run claims
    for ci, claim in enumerate(group.claims, 1):
        result = run_single_claim(
            orchestrator, index, claim, group.group_id,
            ci, len(group.claims), trace_level, adapter,
        )
        gr.claim_results.append(result)

    return gr


# ============================================================================
# SUMMARY BUILDER
# ============================================================================

def build_summary(
    group_results: List[GroupResult],
    adapter: DatasetAdapter,
    llm_model: str,
    embed_model: str,
) -> UnifiedRunSummary:
    """Aggregate group results into an UnifiedRunSummary."""
    all_results: List[UnifiedClaimResult] = []
    for gr in group_results:
        all_results.extend(gr.claim_results)

    completed = [r for r in all_results if r.error is None]
    errored = [r for r in all_results if r.error is not None]

    verdict_accuracy = (
        sum(1 for r in completed if r.correct) / len(completed)
        if completed else 0.0
    )
    mean_evidence_recall = (
        sum(r.evidence_doc_recall for r in completed) / len(completed)
        if completed else 0.0
    )
    mean_scope_recall = (
        sum(r.scope_recall for r in completed) / len(completed)
        if completed else 0.0
    )
    
    mean_confidence = (
        sum(r.confidence for r in completed) / len(completed)
        if completed else 0.0
    )

    # Per-verdict accuracy
    verdicts = [v.value for v in SciFactVerdict]
    per_verdict: Dict[str, Dict[str, Any]] = {}
    for v in verdicts:
        subset = [r for r in completed if r.expected_verdict.value == v]
        if subset:
            correct = sum(1 for r in subset if r.correct)
            per_verdict[v] = {
                "count": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
            }

    # Confusion matrix
    cm: Dict[str, Dict[str, int]] = {v: {v2: 0 for v2 in verdicts} for v in verdicts}
    for r in completed:
        if r.mapped_verdict:
            cm[r.expected_verdict.value][r.mapped_verdict.value] += 1

    total_index_time = sum(gr.index_time_seconds for gr in group_results)

    summary = UnifiedRunSummary(
        dataset_name=adapter.name,
        total_groups=len(group_results),
        total_claims=len(all_results),
        completed=len(completed),
        errored=len(errored),
        verdict_accuracy=verdict_accuracy,
        mean_evidence_recall=mean_evidence_recall,
        mean_scope_recall=mean_scope_recall,
        mean_confidence=mean_confidence,
        per_verdict_accuracy=per_verdict,
        confusion_matrix=cm,
        total_index_time=total_index_time,
        llm_model=llm_model,
        embed_model=embed_model,
        group_results=group_results,
    )

    # SciFact-specific: two-stage aggregate + corpus-level stats
    if adapter.has_two_stage:
        two_stage_completed = [
            r for r in completed
            if r.trace.two_stage_failure and r.trace.gold_docs_total > 0
        ]
        summary.two_stage_claims_with_evidence = len(two_stage_completed)
        if two_stage_completed:
            n = len(two_stage_completed)
            success = sum(1 for r in two_stage_completed if r.trace.two_stage_failure == "SUCCESS")
            retr_fail = sum(1 for r in two_stage_completed if r.trace.two_stage_failure == "RETRIEVAL_FAILURE")
            reas_fail = sum(1 for r in two_stage_completed if r.trace.two_stage_failure == "REASONING_FAILURE")
            summary.two_stage_success_rate = success / n
            summary.two_stage_retrieval_failure_rate = retr_fail / n
            summary.two_stage_reasoning_failure_rate = reas_fail / n
            summary.two_stage_librarian_failure_count = sum(
                1 for r in two_stage_completed if r.trace.retrieval_attribution == "LIBRARIAN_FAILURE"
            )
            summary.two_stage_planner_failure_count = sum(
                1 for r in two_stage_completed if r.trace.retrieval_attribution == "PLANNER_FAILURE"
            )
            summary.two_stage_graph_recall_mean = sum(
                r.trace.graph_recall_at_k for r in two_stage_completed
            ) / n
            retrieved = [r for r in two_stage_completed if r.trace.two_stage_failure != "RETRIEVAL_FAILURE"]
            if retrieved:
                summary.two_stage_verdict_acc_when_retrieved = sum(
                    1 for r in retrieved if r.correct
                ) / len(retrieved)

    # SciFact corpus-level stats (single group)
    if adapter.name == "scifact" and group_results:
        gr0 = group_results[0]
        summary.corpus_size = gr0.corpus_size
        summary.corpus_mode = gr0.metadata.get("corpus_mode", "full")
        summary.graph_nodes = gr0.graph_nodes
        summary.graph_edges = gr0.graph_edges
        summary.index_diagnostics = gr0.index_diagnostics

    # Wiki-specific: per-type / per-level
    if adapter.has_per_type_level:
        for qtype in ("comparison", "bridge"):
            subset = [
                r for gr in group_results if gr.metadata.get("question_type") == qtype
                for r in gr.claim_results if r.error is None
            ]
            if subset:
                summary.per_type_accuracy[qtype] = sum(1 for r in subset if r.correct) / len(subset)

        for level in ("medium", "hard"):
            subset = [
                r for gr in group_results if gr.metadata.get("level") == level
                for r in gr.claim_results if r.error is None
            ]
            if subset:
                summary.per_level_accuracy[level] = sum(1 for r in subset if r.correct) / len(subset)

    return summary




# ============================================================================
# MAIN
# ============================================================================

def main(
    dataset: str = "scifact",
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    embed_model: Optional[str] = None,
    use_mock: bool = False,
    trace_chunks: int = 50,
    small_corpus: bool = False,
    trace_level: str = "standard",
    data_path: Optional[str] = None,
    merge_groups: bool = False,
) -> None:
    """Run unified v5 evaluation for the specified dataset."""
    from config.logging_config import setup_logging, log_file_from_output_path

    # Ensure output directory exists
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Resolve output path
    if output_path is None:
        output_path = os.path.join(_OUTPUT_DIR, f"{dataset}_output.md")
    elif not os.path.isabs(output_path) and os.sep not in output_path:
        # If it's just a filename, put it in the output directory
        output_path = os.path.join(_OUTPUT_DIR, output_path)

    setup_logging(log_file=log_file_from_output_path(output_path))

    tl = TraceLevel(trace_level)

    adapter = get_adapter(dataset, data_path=data_path)

    _print("=" * 70)
    _print(f"{adapter.display_name} v5 Evaluation (unified runner)")
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
        llm_client = get_client()
        embed_client = get_embedding_client()
        llm_model_name = getattr(llm_client, "model", getattr(llm_client, "model_path", "unknown"))
        embed_model_name = getattr(embed_client, "model", "unknown")
        _print(f"LLM: {llm_model_name}")
        _print(f"Embedding: {embed_model_name}")

    # 2. Create orchestrator
    planner_config = {}
    if dataset == "musique":
        planner_config["decomposition"] = {
            "enable_decomposition": True,
            "min_entities_for_decomposition": 2,
        }

    config: Dict[str, Any] = {
        "explorer": {},
        "librarian": {},
        "planner": planner_config,
        "reasoner": {},
        "save_checkpoint": True,
        "checkpoint_path": os.path.join(_OUTPUT_DIR, f"{dataset}_librarian_checkpoint.json"),
    }
    orchestrator = OrchestratorV5(llm_client, embed_client, config)

    # 3. Load dataset
    _print(f"\nLoading {adapter.display_name} dataset...")
    groups = adapter.load_groups(
        limit=limit,
        small_corpus=small_corpus,
        merge_groups=merge_groups
    )

    total_claims = sum(len(g.claims) for g in groups)
    _print(f"  Groups: {len(groups)}, Claims: {total_claims}")

    # 4. Process groups
    _print(f"\nProcessing {len(groups)} groups...\n")
    group_results: List[GroupResult] = []
    total_claims_processed = 0

    try:
        for gi, group in enumerate(groups, 1):
            gr = run_group(orchestrator, group, gi, len(groups), tl, adapter)
            group_results.append(gr)
            total_claims_processed += len(gr.claim_results)

            # Checkpoint (SciFact: every 10 claims; Wiki: every group)
            if adapter.has_checkpointing and total_claims_processed % 10 == 0:
                write_checkpoint(group_results, output_path)
                _print(f"  (checkpoint written: {total_claims_processed} claims)")
            elif not adapter.has_checkpointing:
                # For Wiki, checkpoint after each group
                write_checkpoint(group_results, output_path)
    except KeyboardInterrupt:
        _print(f"\n\nInterrupted after {len(group_results)} groups. "
               "Writing partial results...")
    finally:
        # 5. Build summary and write output
        summary = build_summary(group_results, adapter, llm_model_name, embed_model_name)

        md = format_output_md(summary, adapter)
        with open(output_path, "w") as f:
            f.write(md)
        _print(f"\nResults written to {output_path}")

        # Write machine-readable trace JSON
        write_trace_json(summary, output_path, orchestrator, trace_chunks, tl)

        # Clean up checkpoint if fully completed
        checkpoint_path = output_path.rsplit(".", 1)[0] + ".checkpoint.json"
        expected_claims = sum(len(g.claims) for g in groups)
        if total_claims_processed == expected_claims and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Print summary to stdout
        _print(f"\n{'=' * 70}")
        _print(f"SUMMARY")
        _print(f"{'=' * 70}")
        _print(f"  Dataset:          {adapter.display_name}")
        _print(f"  Groups:           {summary.total_groups}")
        _print(f"  Claims:           {summary.total_claims} "
               f"({summary.completed} completed, {summary.errored} errors)")
        _print(f"  Verdict Accuracy: {summary.verdict_accuracy:.3f}")
        if adapter.has_evidence_doc_recall:
            _print(f"  Evidence Recall:  {summary.mean_evidence_recall:.3f}")
        _print(f"  Scope Recall:     {summary.mean_scope_recall:.3f}")
        _print(f"  Mean Confidence:  {summary.mean_confidence:.3f}")
        _print(f"  Total Index Time: {summary.total_index_time:.1f}s")
        if adapter.has_per_type_level:
            if summary.per_type_accuracy:
                for qtype, acc in sorted(summary.per_type_accuracy.items()):
                    _print(f"  {qtype} Accuracy: {acc:.3f}")
            if summary.per_level_accuracy:
                for level, acc in sorted(summary.per_level_accuracy.items()):
                    _print(f"  {level} Accuracy: {acc:.3f}")
        _print(f"{'=' * 70}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Pre-flight check: Ensure running from internal drive
    cwd = os.getcwd()
    if cwd.startswith("/Volumes"):
        print(f"CRITICAL ERROR: Refusing to run from external volume: {cwd}")
        print("Please run from the internal project location to ensure performance and correctness.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Unified v5 Evaluation Runner",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["scifact", "wikifactclaims", "multihop", "musique"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum claims (scifact) or rows (wikifactclaims) to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output markdown file path (default: {dataset}_output.md)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama LLM model name (default: from OLLAMA_MODEL env)",
    )
    parser.add_argument(
        "--embed-model", type=str, default=None,
        help="Ollama embedding model name (default: from OLLAMA_EMBED_MODEL env)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM/embedding clients for smoke testing",
    )
    parser.add_argument(
        "--trace-level", type=str, default="standard",
        choices=["minimal", "standard", "full"],
        help="Diagnostic verbosity (default: standard)",
    )
    parser.add_argument(
        "--trace-chunks", type=int, default=50,
        help="Chunk extraction details in diagnostics JSON (0=skip, -1=all, default: 50)",
    )
    parser.add_argument(
        "--small-corpus", action="store_true",
        help="SciFact only: index limit*5 docs instead of full corpus",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="WikiFactClaims only: path to wikifactclaims.jsonl",
    )
    parser.add_argument(
        "--merge-groups", action="store_true",
        help="WikiFactClaims only: merge all loaded groups into a single graph context",
    )

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        limit=args.limit,
        output_path=args.output,
        model=args.model,
        embed_model=args.embed_model,
        use_mock=args.mock,
        trace_chunks=args.trace_chunks,
        small_corpus=args.small_corpus,
        trace_level=args.trace_level,
        data_path=args.data_path,
        merge_groups=args.merge_groups,
    )
