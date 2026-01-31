"""
Output Formatters for v5 Evaluation
Location: evaluation/output_formatters.py
Owner: Audit Agent

Provides markdown report generation, JSON trace writing, and checkpointing.
Extracted from run_evaluation_v5.py for modularity.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from schemas.contracts import TraceLevel
from schemas.evaluation_contracts import SciFactVerdict, UnifiedRunSummary, GroupResult

if TYPE_CHECKING:
    from orchestrator import OrchestratorV5
    from evaluation.adapters import DatasetAdapter

# Default printer (unbuffered)
def _default_print(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True so output appears immediately in pipes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================================
# MARKDOWN FORMATTER
# ============================================================================

def format_output_md(
    summary: UnifiedRunSummary,
    adapter: "DatasetAdapter",
) -> str:
    """Render evaluation results as Markdown (common + conditional sections)."""
    lines: List[str] = []

    lines.append(f"# {adapter.display_name} v5 Evaluation Results\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append(f"- **Pipeline:** v5" + (
        " (per-group indexing)" if summary.total_groups > 1 else ""
    ))
    lines.append(f"- **LLM:** {summary.llm_model}")
    lines.append(f"- **Embedding:** {summary.embed_model}")
    if adapter.name == "scifact":
        lines.append(f"- **Corpus:** {summary.corpus_size} documents "
                     f"indexed in {summary.total_index_time:.1f} seconds")
        lines.append(f"- **Corpus mode:** "
                     f"{'small-corpus (5 docs/claim)' if summary.corpus_mode == 'small-corpus' else 'full'}")
        lines.append(f"- **Graph:** {summary.graph_nodes} nodes, {summary.graph_edges} edges")
        lines.append(f"- **Claims evaluated:** {summary.total_claims} "
                     f"(completed: {summary.completed}, errors: {summary.errored})")
    else:
        lines.append(f"- **Groups evaluated:** {summary.total_groups}")
        lines.append(f"- **Total claims:** {summary.total_claims} "
                     f"(completed: {summary.completed}, errors: {summary.errored})")
        lines.append(f"- **Total index time:** {summary.total_index_time:.1f}s")
    lines.append(f"- **Timestamp:** {datetime.now().isoformat()}")
    lines.append("")

    # Summary Metrics
    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Verdict Accuracy | {summary.verdict_accuracy:.3f} |")
    if adapter.has_evidence_doc_recall:
        lines.append(f"| Evidence Doc Recall | {summary.mean_evidence_recall:.3f} |")
    lines.append(f"| Planner Scope Recall | {summary.mean_scope_recall:.3f} |")
    lines.append(f"| Mean Confidence | {summary.mean_confidence:.3f} |")
    lines.append("")

    # Two-Stage Failure Analysis (SciFact only)
    if adapter.has_two_stage and (
        summary.two_stage_success_rate > 0
        or summary.two_stage_retrieval_failure_rate > 0
        or summary.two_stage_reasoning_failure_rate > 0
    ):
        lines.append("## Two-Stage Failure Analysis\n")
        lines.append("| Category | Count | Rate |")
        lines.append("|----------|-------|------|")
        n = summary.completed
        success_n = int(summary.two_stage_success_rate * n + 0.5)
        retr_n = int(summary.two_stage_retrieval_failure_rate * n + 0.5)
        reas_n = int(summary.two_stage_reasoning_failure_rate * n + 0.5)
        lines.append(f"| SUCCESS | {success_n} | {summary.two_stage_success_rate:.3f} |")
        lines.append(f"| RETRIEVAL_FAILURE | {retr_n} | {summary.two_stage_retrieval_failure_rate:.3f} |")
        lines.append(f"| \u2014 LIBRARIAN_FAILURE | {summary.two_stage_librarian_failure_count} | - |")
        lines.append(f"| \u2014 PLANNER_FAILURE | {summary.two_stage_planner_failure_count} | - |")
        lines.append(f"| REASONING_FAILURE | {reas_n} | {summary.two_stage_reasoning_failure_rate:.3f} |")
        lines.append(f"| Graph Recall@K (mean) | - | {summary.two_stage_graph_recall_mean:.3f} |")
        lines.append(f"| Verdict Acc (when retrieved) | - | {summary.two_stage_verdict_acc_when_retrieved:.3f} |")
        lines.append(f"\n*Based on {summary.two_stage_claims_with_evidence} claims "
                     f"with expected evidence documents.*\n")
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

    # Per-Type Accuracy (Wiki only)
    if adapter.has_per_type_level and summary.per_type_accuracy:
        lines.append("## Per-Type Accuracy\n")
        lines.append("| Question Type | Accuracy |")
        lines.append("|---------------|----------|")
        for qtype, acc in sorted(summary.per_type_accuracy.items()):
            lines.append(f"| {qtype} | {acc:.3f} |")
        lines.append("")

    # Per-Level Accuracy (Wiki only)
    if adapter.has_per_type_level and summary.per_level_accuracy:
        lines.append("## Per-Level Accuracy\n")
        lines.append("| Level | Accuracy |")
        lines.append("|-------|----------|")
        for level, acc in sorted(summary.per_level_accuracy.items()):
            lines.append(f"| {level} | {acc:.3f} |")
        lines.append("")

    # Per-Row/Group Summary (Wiki only, multi-group)
    if summary.total_groups > 1:
        lines.append("## Per-Group Summary\n")
        lines.append("| Group ID | Type | Level | Nodes | Edges | Index(s) | Claims | Correct |")
        lines.append("|----------|------|-------|-------|-------|----------|--------|---------|")
        for gr in summary.group_results:
            completed = [r for r in gr.claim_results if r.error is None]
            correct = sum(1 for r in completed if r.correct)
            total = len(gr.claim_results)
            qtype = gr.metadata.get("question_type", "")
            level = gr.metadata.get("level", "")
            lines.append(
                f"| {gr.group_id} | {qtype} | {level} "
                f"| {gr.graph_nodes} | {gr.graph_edges} | {gr.index_time_seconds:.1f} "
                f"| {total} | {correct}/{len(completed)} |"
            )
        lines.append("")

    # Per-Claim Results
    lines.append("## Per-Claim Results\n")
    if adapter.has_per_type_level:
        lines.append("| Group | Idx | Claim (60 chars) | Expected | Predicted | OK | Conf | Scope Recall |")
        lines.append("|-------|-----|------------------|----------|-----------|-----|------|--------------|")
    else:
        lines.append("| ID | Claim (60 chars) | Expected | Predicted | OK | Conf | Ev.Recall | Lane |")
        lines.append("|-----|------------------|----------|-----------|-----|------|-----------|------|")

    for gr in summary.group_results:
        for r in gr.claim_results:
            claim_short = (r.claim_text[:57] + "...") if len(r.claim_text) > 60 else r.claim_text
            if r.error is not None:
                if adapter.has_per_type_level:
                    lines.append(
                        f"| {r.group_id} | {r.metadata.get('claim_index', '')} | {claim_short} "
                        f"| {r.expected_verdict.value} | ERROR | - | - | - |"
                    )
                else:
                    lines.append(
                        f"| {r.claim_id} | {claim_short} | {r.expected_verdict.value} "
                        f"| ERROR | - | - | - | - |"
                    )
            else:
                check = "\u2713" if r.correct else "\u2717"
                pred = r.mapped_verdict.value if r.mapped_verdict else "?"
                if adapter.has_per_type_level:
                    lines.append(
                        f"| {r.group_id} | {r.metadata.get('claim_index', '')} | {claim_short} "
                        f"| {r.expected_verdict.value} | {pred} | {check} "
                        f"| {r.confidence:.2f} | {r.scope_recall:.2f} |"
                    )
                else:
                    lines.append(
                        f"| {r.claim_id} | {claim_short} | {r.expected_verdict.value} "
                        f"| {pred} | {check} | {r.confidence:.2f} "
                        f"| {r.evidence_doc_recall:.2f} | {r.lane_used} |"
                    )
    lines.append("")

    # Diagnostic Trace (Wiki only — multi-group)
    if adapter.has_per_type_level:
        lines.append("## Diagnostic Trace\n")
        for gr in summary.group_results:
            qtype = gr.metadata.get("question_type", "")
            level = gr.metadata.get("level", "")
            lines.append(f"### Group {gr.group_id} ({qtype} / {level})\n")
            lines.append(f"- **Index:** {gr.graph_nodes} nodes, {gr.graph_edges} edges, "
                         f"{gr.graph_bridge_count} bridges, {gr.graph_community_count} communities")
            lines.append(f"- **Extractor:** {gr.extractor_entities} entities, "
                         f"{gr.extractor_claims} claims, "
                         f"{gr.extractor_llm_calls} LLM calls ({gr.extractor_time_seconds:.1f}s)")
            lines.append("")
            for r in gr.claim_results:
                t = r.trace
                if r.error is not None:
                    lines.append(f"**Claim {r.metadata.get('claim_index', r.claim_id)}:** ERROR \u2014 {r.error}\n")
                    continue
                check = "\u2713" if r.correct else "\u2717"
                pred = r.mapped_verdict.value if r.mapped_verdict else "?"
                lines.append(
                    f"**Claim {r.metadata.get('claim_index', r.claim_id)}:** "
                    f"{r.predicted_verdict_type.value}\u2192{pred} "
                    f"(exp: {r.expected_verdict.value}) {check}  "
                    f"conf={r.confidence:.2f}  scope_recall={r.scope_recall:.2f}"
                )
                lines.append("")
                lines.append(
                    f"- **Planner:** seeds={t.seeds_found} "
                    f"scope={t.scope_nodes}nodes/{t.scope_documents}docs "
                    f"lane={t.selected_lane} "
                    f"ppr_mass={t.ppr_total_mass:.2f} "
                    f"router_conf={t.router_confidence:.2f} "
                    f"search_strategy={t.search_strategy}"
                )
                lines.append(
                    f"- **Reasoner:** docs={t.docs_in_scope} "
                    f"rels={t.relationships_inferred} "
                    f"ev={t.evidence_grounded} "
                    f"llm={t.llm_time_ms:.0f}ms "
                    f"total={t.total_time_ms:.0f}ms"
                )
                lines.append(
                    f"- **Confidence:** overall={t.confidence_overall:.2f} "
                    f"ev_quality={t.confidence_evidence_quality:.2f} "
                    f"agreement={t.confidence_source_agreement:.2f} "
                    f"coverage={t.confidence_coverage:.2f}"
                )
                if t.verdict_answer:
                    lines.append(f"- **Verdict answer:** {t.verdict_answer[:500]}")
                if t.verdict_summary:
                    lines.append(f"- **Verdict summary:** {t.verdict_summary[:500]}")
                if t.raw_response:
                    lines.append(f"- **Raw LLM response:** `{t.raw_response[:500]}`")
                if t.seed_details:
                    lines.append("- **Top seeds:**")
                    for sd in t.seed_details:
                        lines.append(
                            f"  - {sd['node_id']} sim={sd['similarity']:.4f} "
                            f"doc={sd['source_doc_id']} \u2014 {sd['canonical_text']}"
                        )
                if t.ranked_chunks:
                    lines.append("- **Top ranked chunks:**")
                    for rc in t.ranked_chunks:
                        lines.append(
                            f"  - {rc['chunk_id']} rel={rc['relevance_score']:.4f} "
                            f"doc={rc['source_doc_id']} \u2014 {rc['text_preview'][:80]}"
                        )
                if t.relationships_detail:
                    lines.append("- **Relationships:**")
                    for rd in t.relationships_detail:
                        lines.append(
                            f"  - {rd['source_id']} \u2192 {rd['target_id']} "
                            f"[{rd['relationship']}] conf={rd['confidence']:.4f}"
                        )
                if t.evidence_detail:
                    lines.append("- **Evidence:**")
                    for ed in t.evidence_detail:
                        lines.append(
                            f"  - doc={ed['source_document']} \u2014 {ed['claim'][:100]}"
                        )
                lines.append("")
        lines.append("")

    # Error Log
    errors = [
        r for gr in summary.group_results
        for r in gr.claim_results if r.error is not None
    ]
    if errors:
        lines.append("## Error Log\n")
        for r in errors:
            lines.append(f"- **Claim {r.claim_id}:** {r.error}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# TRACE JSON WRITER
# ============================================================================

def _serialize_chunk_details(
    chunk_details: list,
    trace_chunks: int,
    trace_level: TraceLevel = TraceLevel.STANDARD,
    chunk_text_lookup: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Serialize chunk extraction details with configurable limit.

    Parameters
    ----------
    chunk_text_lookup:
        Optional mapping from chunk_id to full text (from ChunkerOutput).
        Used at FULL trace level to include full chunk text.
    """
    if trace_chunks == 0 or not chunk_details:
        return []
    items = chunk_details if trace_chunks < 0 else chunk_details[:trace_chunks]
    result = []
    for d in items:
        entry: Dict[str, Any] = {
            "chunk_id": d.chunk_id,
            "source_doc_id": d.source_doc_id,
            "token_count": d.token_count,
            "text_preview": d.text_preview,
            "entity_count": len(d.entities),
            "claim_count": len(d.claims),
            "gleaning_iterations": d.gleaning_iterations,
            "gleaning_new_entities": d.gleaning_new_entities,
            "gleaning_new_claims": d.gleaning_new_claims,
        }
        if trace_level == TraceLevel.FULL:
            if chunk_text_lookup and d.chunk_id in chunk_text_lookup:
                entry["full_text"] = chunk_text_lookup[d.chunk_id]
            else:
                entry["full_text"] = d.text_preview
        result.append(entry)
    return result


def write_trace_json(
    summary: UnifiedRunSummary,
    output_path: str,
    orchestrator: Optional["OrchestratorV5"] = None,
    trace_chunks: int = 50,
    trace_level: TraceLevel = TraceLevel.STANDARD,
    printer: Callable[..., None] = _default_print,
) -> None:
    """Write group-aware trace data to a JSON file alongside output.md."""
    diag_path = output_path.rsplit(".", 1)[0] + ".trace.json"

    data: Dict[str, Any] = {
        "dataset": summary.dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": summary.llm_model,
            "embed_model": summary.embed_model,
            "total_groups": summary.total_groups,
            "total_claims": summary.total_claims,
            "trace_level": trace_level.value,
        },
        "metrics": {
            "verdict_accuracy": summary.verdict_accuracy,
            "mean_evidence_recall": summary.mean_evidence_recall,
            "mean_scope_recall": summary.mean_scope_recall,
            "mean_confidence": summary.mean_confidence,
            "completed": summary.completed,
            "errored": summary.errored,
        },
        "confusion_matrix": summary.confusion_matrix,
        "per_verdict_accuracy": summary.per_verdict_accuracy,
    }

    # SciFact-specific top-level metrics
    if summary.corpus_size > 0:
        data["config"]["corpus_size"] = summary.corpus_size
        data["config"]["corpus_mode"] = summary.corpus_mode

    if summary.two_stage_claims_with_evidence > 0:
        data["two_stage"] = {
            "claims_with_evidence": summary.two_stage_claims_with_evidence,
            "success_rate": summary.two_stage_success_rate,
            "retrieval_failure_rate": summary.two_stage_retrieval_failure_rate,
            "reasoning_failure_rate": summary.two_stage_reasoning_failure_rate,
            "librarian_failure_count": summary.two_stage_librarian_failure_count,
            "planner_failure_count": summary.two_stage_planner_failure_count,
            "graph_recall_mean": summary.two_stage_graph_recall_mean,
            "verdict_acc_when_retrieved": summary.two_stage_verdict_acc_when_retrieved,
        }

    # Wiki-specific top-level metrics
    if summary.per_type_accuracy:
        data["per_type_accuracy"] = summary.per_type_accuracy
    if summary.per_level_accuracy:
        data["per_level_accuracy"] = summary.per_level_accuracy

    # Per-group structure
    is_standard_plus = trace_level in (TraceLevel.STANDARD, TraceLevel.FULL)
    groups_data: List[Dict[str, Any]] = []

    for gr in summary.group_results:
        group_entry: Dict[str, Any] = {
            "group_id": gr.group_id,
            "index": {
                "time_seconds": gr.index_time_seconds,
                "graph_nodes": gr.graph_nodes,
                "graph_edges": gr.graph_edges,
                "graph_bridge_count": gr.graph_bridge_count,
                "graph_community_count": gr.graph_community_count,
                "corpus_size": gr.corpus_size,
                "extractor_entities": gr.extractor_entities,
                "extractor_claims": gr.extractor_claims,
                "extractor_llm_calls": gr.extractor_llm_calls,
                "extractor_time_seconds": gr.extractor_time_seconds,
            },
            "metadata": gr.metadata,
        }

        # Index diagnostics (SciFact)
        if gr.index_diagnostics:
            d = gr.index_diagnostics
            group_entry["index_diagnostics"] = {
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
                "extracted_triples": {
                    "count": d.extracted_triples_count,
                    "sample": d.extracted_triples_sample,
                },
                "synonymy_edges": {
                    "count": d.synonymy_edge_count,
                    "mean_score": d.synonymy_edge_mean_score,
                    "min_score": d.synonymy_edge_min_score,
                    "max_score": d.synonymy_edge_max_score,
                    "sample": d.synonymy_edges_sample,
                },
                "community_summaries": d.community_summaries,
                "stage_timings": d.stage_timings,
                "token_usage": d.stage_token_usage,
            }

            # Full chunker/extractor trace for SciFact (from orchestrator)
            if orchestrator:
                chunker_out = orchestrator.last_chunker_output
                if chunker_out and chunker_out.trace:
                    ct = chunker_out.trace
                    chunk_summaries: List[Dict[str, Any]] = []
                    if ct.chunks:
                        limit_n = len(ct.chunks) if trace_chunks < 0 else min(trace_chunks, len(ct.chunks))
                        # Build chunk_id -> full text for FULL level
                        full_text_map: Dict[str, str] = {}
                        if trace_level == TraceLevel.FULL:
                            full_text_map = {
                                c.chunk_id: c.text
                                for c in chunker_out.chunks
                            }
                        for c in ct.chunks[:limit_n]:
                            cs_entry: Dict[str, Any] = {
                                "chunk_id": c.chunk_id,
                                "source_doc_id": c.source_doc_id,
                                "token_count": c.token_count,
                                "position": c.position,
                                "section_header": c.section_header,
                                "text_preview": c.text_preview,
                            }
                            if trace_level == TraceLevel.FULL:
                                cs_entry["full_text"] = full_text_map.get(
                                    c.chunk_id, c.text_preview,
                                )
                            chunk_summaries.append(cs_entry)
                    group_entry["chunker_trace"] = {
                        "trace_id": ct.trace_id,
                        "input_doc_ids_count": len(ct.input_doc_ids),
                        "total_chunks": ct.total_chunks,
                        "chunk_summaries": chunk_summaries,
                        "chunk_summaries_included": len(chunk_summaries),
                        "chunk_summaries_total": len(ct.chunks) if ct.chunks else 0,
                    }

                explorer_out = orchestrator.last_explorer_output
                if explorer_out and explorer_out.trace:
                    et = explorer_out.trace
                    # Build chunk_id -> full text lookup for FULL trace
                    chunk_text_lookup: Optional[Dict[str, str]] = None
                    if trace_level == TraceLevel.FULL and chunker_out:
                        chunk_text_lookup = {
                            c.chunk_id: c.text
                            for c in chunker_out.chunks
                        }
                    group_entry["extractor_trace"] = {
                        "trace_id": et.trace_id,
                        "input_chunk_count": et.input_chunk_count,
                        "entities_extracted": et.entities_extracted,
                        "claims_extracted": et.claims_extracted,
                        "gleaning_iterations": et.gleaning_iterations,
                        "llm_calls": et.llm_calls,
                        "extraction_time_seconds": et.extraction_time_seconds,
                        "chunk_details": _serialize_chunk_details(
                            et.chunk_details or [], trace_chunks,
                            trace_level=trace_level,
                            chunk_text_lookup=chunk_text_lookup,
                        ),
                        "chunk_details_included": min(
                            trace_chunks if trace_chunks >= 0 else len(et.chunk_details or []),
                            len(et.chunk_details or []),
                        ),
                        "chunk_details_total": len(et.chunk_details) if et.chunk_details else 0,
                    }

        # Per-claim traces
        claim_traces: List[Dict[str, Any]] = []
        for r in gr.claim_results:
            t = r.trace
            planner_trace: Dict[str, Any] = {
                "seeds_found": t.seeds_found,
                "scope_nodes": t.scope_nodes,
                "scope_documents": t.scope_documents,
                "selected_lane": t.selected_lane,
                "search_strategy": t.search_strategy,
            }
            reasoner_trace: Dict[str, Any] = {
                "docs_in_scope": t.docs_in_scope,
                "relationships_inferred": t.relationships_inferred,
                "evidence_grounded": t.evidence_grounded,
                "llm_time_ms": t.llm_time_ms,
                "total_time_ms": t.total_time_ms,
            }

            if is_standard_plus:
                planner_trace["router_confidence"] = t.router_confidence
                planner_trace["ppr_total_mass"] = t.ppr_total_mass
                planner_trace["seed_details"] = t.seed_details
                planner_trace["scope_node_details"] = t.scope_node_details
                planner_trace["ranked_chunks"] = t.ranked_chunks

                reasoner_trace["verdict_answer"] = t.verdict_answer
                reasoner_trace["verdict_summary"] = t.verdict_summary
                reasoner_trace["reasoning_trace"] = t.reasoning_trace
                reasoner_trace["confidence_overall"] = t.confidence_overall
                reasoner_trace["confidence_evidence_quality"] = t.confidence_evidence_quality
                reasoner_trace["confidence_source_agreement"] = t.confidence_source_agreement
                reasoner_trace["confidence_coverage"] = t.confidence_coverage
                reasoner_trace["scope_document_ids"] = t.scope_document_ids
                reasoner_trace["scope_bridge_edges_used"] = t.scope_bridge_edges_used
                reasoner_trace["relationships_detail"] = t.relationships_detail
                reasoner_trace["evidence_detail"] = t.evidence_detail

                if trace_level == TraceLevel.FULL:
                    reasoner_trace["raw_response"] = t.raw_response

            # Performance sub-object
            performance: Dict[str, Any] = {
                "planner_ms": t.stage_timing_planner_ms,
                "reasoner_ms": t.stage_timing_reasoner_ms,
                "total_ms": t.stage_timing_total_ms,
                "reasoner_tokens": {
                    "prompt": t.reasoner_prompt_tokens,
                    "completion": t.reasoner_completion_tokens,
                    "total": t.reasoner_total_tokens,
                },
                "planner_tokens": {
                    "prompt": t.planner_prompt_tokens,
                    "total": t.planner_total_tokens,
                },
            }

            # Final prompt at FULL level
            if trace_level == TraceLevel.FULL and t.final_prompt is not None:
                reasoner_trace["final_prompt"] = t.final_prompt
                reasoner_trace["prompt_token_count"] = t.prompt_token_count

            entry: Dict[str, Any] = {
                "claim_id": r.claim_id,
                "group_id": r.group_id,
                "claim_text": r.claim_text[:120],
                "expected": r.expected_verdict.value,
                "predicted_raw": r.predicted_verdict_type.value if r.predicted_verdict_type else None,
                "predicted_mapped": r.mapped_verdict.value if r.mapped_verdict else None,
                "correct": r.correct,
                "confidence": r.confidence,
                "lane": r.lane_used,
                "search_strategy": r.search_strategy,
                "evidence_doc_ids": r.evidence_doc_ids,
                "expected_doc_ids": r.expected_doc_ids,
                "evidence_doc_recall": r.evidence_doc_recall,
                "scope_recall": r.scope_recall,
                "relationships": r.relationships,
                "planner_trace": planner_trace,
                "reasoner_trace": reasoner_trace,
                "performance": performance,
                "error": r.error,
            }

            # Two-stage fields (SciFact)
            if t.two_stage_failure:
                entry["two_stage_failure"] = t.two_stage_failure
                entry["retrieval_attribution"] = t.retrieval_attribution
                entry["graph_recall_at_k"] = t.graph_recall_at_k
                entry["gold_docs_in_scope"] = t.gold_docs_in_scope
                entry["gold_docs_in_graph"] = t.gold_docs_in_graph
                entry["gold_docs_total"] = t.gold_docs_total
                entry["planner_scope_doc_ids"] = t.planner_scope_doc_ids
                entry["planner_scope_recall"] = t.planner_scope_recall
                entry["evidence_failure_attribution"] = t.evidence_failure_attribution

            claim_traces.append(entry)
        group_entry["claim_traces"] = claim_traces
        groups_data.append(group_entry)

    data["groups"] = groups_data

    with open(diag_path, "w") as f:
        json.dump(data, f, indent=2)
    printer(f"  Trace written to {diag_path}")


# ============================================================================
# CHECKPOINT WRITER
# ============================================================================

def write_checkpoint(
    group_results: List[GroupResult],
    output_path: str,
) -> None:
    """Write partial results to a checkpoint JSON file."""
    checkpoint_path = output_path.rsplit(".", 1)[0] + ".checkpoint.json"
    data: List[Dict[str, Any]] = []
    for gr in group_results:
        for r in gr.claim_results:
            t = r.trace
            data.append({
                "claim_id": r.claim_id,
                "group_id": r.group_id,
                "expected": r.expected_verdict.value,
                "predicted": r.mapped_verdict.value if r.mapped_verdict else None,
                "predicted_raw": r.predicted_verdict_type.value if r.predicted_verdict_type else None,
                "correct": r.correct,
                "confidence": r.confidence,
                "lane": r.lane_used,
                "search_strategy": r.search_strategy,
                "evidence_doc_recall": r.evidence_doc_recall,
                "scope_recall": r.scope_recall,
                "trace_seeds_found": t.seeds_found,
                "trace_scope_nodes": t.scope_nodes,
                "trace_scope_documents": t.scope_documents,
                "trace_selected_lane": t.selected_lane,
                "trace_search_strategy": t.search_strategy,
                "trace_docs_in_scope": t.docs_in_scope,
                "trace_total_time_ms": t.total_time_ms,
                "trace_confidence_overall": t.confidence_overall,
                "trace_confidence_evidence_quality": t.confidence_evidence_quality,
                "trace_confidence_source_agreement": t.confidence_source_agreement,
                "trace_confidence_coverage": t.confidence_coverage,
                "trace_router_confidence": t.router_confidence,
                "trace_scope_bridge_edges_used": t.scope_bridge_edges_used,
                "error": r.error,
                "two_stage_failure": t.two_stage_failure,
                "retrieval_attribution": t.retrieval_attribution,
                "graph_recall_at_k": t.graph_recall_at_k,
                "gold_docs_in_scope": t.gold_docs_in_scope,
                "gold_docs_total": t.gold_docs_total,
                "planner_scope_recall": t.planner_scope_recall,
            })
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)
