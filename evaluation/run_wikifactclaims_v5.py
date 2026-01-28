"""
WikiFactClaims v5 Evaluation
Location: evaluation/run_wikifactclaims_v5.py
Owner: ROOT Instance

Per-row indexing evaluation for WikiFactClaims. Each row has 10 context
passages indexed independently, then each claim is evaluated for planner
retrieval recall and reasoner verdict accuracy.

Usage:
    python -m evaluation.run_wikifactclaims_v5 --limit 5 --mock
    python -m evaluation.run_wikifactclaims_v5 --limit 20 --output wiki_output.md \
        --model qwen2.5:14b --embed-model nomic-embed-text
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
from evaluation.data_loaders.wikifactclaims_loader import WikiFactClaimsLoader
from schemas.contracts import (
    VerdictType,
    QueryRequest,
    ReasonerOutput,
)
from schemas.evaluation_contracts import (
    SciFactVerdict,
    WikiClaimLabel,
    WikiClaimResult,
    WikiFactClaimsRow,
    WikiRowResult,
    WikiRunSummary,
)

_logger = logging.getLogger(__name__)


# ============================================================================
# UNBUFFERED PRINT
# ============================================================================

def _print(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True so output appears immediately in pipes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================================
# VERDICT MAPPING  (identical to run_scifact_v5)
# ============================================================================

_VERDICT_MAP: Dict[VerdictType, SciFactVerdict] = {
    VerdictType.CONFIRMED: SciFactVerdict.SUPPORTS,
    VerdictType.PARTIALLY_CONFIRMED: SciFactVerdict.SUPPORTS,
    VerdictType.REFUTED: SciFactVerdict.REFUTES,
    VerdictType.PARTIALLY_REFUTED: SciFactVerdict.REFUTES,
    VerdictType.INCONCLUSIVE: SciFactVerdict.NOT_ENOUGH_INFO,
    VerdictType.NOT_FOUND: SciFactVerdict.NOT_ENOUGH_INFO,
}


def _map_verdict(verdict_type: VerdictType) -> SciFactVerdict:
    return _VERDICT_MAP.get(verdict_type, SciFactVerdict.NOT_ENOUGH_INFO)


# ============================================================================
# PROCESS A SINGLE ROW
# ============================================================================

def process_row(
    orchestrator: OrchestratorV5,
    row: WikiFactClaimsRow,
    row_num: int,
    total_rows: int,
) -> WikiRowResult:
    """Index the row's 10 context docs, then evaluate each claim."""

    _print(f"\n--- Row {row_num}/{total_rows}  id={row.row_id}  "
           f"type={row.question_type}  level={row.level} ---")
    _print(f"  Q: {row.question[:100]}")

    # Build per-row corpus
    documents = WikiFactClaimsLoader.build_corpus_for_row(row)
    _print(f"  Context: {len(documents)} documents")

    # Index
    t0 = time.time()
    index = orchestrator.index_corpus_v5(documents)
    index_time = time.time() - t0

    graph_nodes = len(index.graph.nodes)
    graph_edges = len(index.graph.edges)
    graph_bridge_count = len(index.graph.bridge_edges) if index.graph.bridge_edges else 0
    graph_community_count = (
        len(index.community_structure.communities)
        if index.community_structure else 0
    )

    # Extractor stats from stashed explorer output
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
           f"{graph_bridge_count} bridges, {graph_community_count} communities ({index_time:.1f}s)")
    _print(f"  Extractor: {ext_entities} entities, {ext_claims} claims, "
           f"{ext_llm_calls} LLM calls ({ext_time:.1f}s)")

    row_result = WikiRowResult(
        row_id=row.row_id,
        question=row.question,
        question_type=row.question_type,
        level=row.level,
        index_time_seconds=index_time,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        graph_bridge_count=graph_bridge_count,
        graph_community_count=graph_community_count,
        extractor_entities=ext_entities,
        extractor_claims=ext_claims,
        extractor_llm_calls=ext_llm_calls,
        extractor_time_seconds=ext_time,
    )

    gold_titles = set(row.supporting_facts_titles)

    for ci, claim in enumerate(row.claims):
        cr = WikiClaimResult(
            row_id=row.row_id,
            claim_index=ci,
            claim_text=claim.text,
            expected_verdict=claim.label,
            supporting_facts_total=len(gold_titles),
        )

        try:
            query = QueryRequest(
                query_id=f"wiki-{row.row_id}-c{ci}",
                query_text=claim.text,
            )
            reasoner_output: ReasonerOutput = orchestrator.process_query_v5(query, index)
            finding = reasoner_output.finding

            # Verdict
            cr.predicted_verdict_type = finding.verdict.verdict_type
            cr.mapped_verdict = _map_verdict(finding.verdict.verdict_type)
            cr.correct = cr.mapped_verdict == claim.label
            cr.confidence = finding.verdict.confidence

            # Planner scope recall
            planner_out = orchestrator.last_planner_output
            if planner_out and planner_out.router_decision.scope:
                scope_doc_ids = set(planner_out.router_decision.scope.document_ids)
                cr.planner_scope_titles = list(scope_doc_ids)
                cr.supporting_facts_in_scope = len(gold_titles & scope_doc_ids)
                cr.scope_recall = (
                    cr.supporting_facts_in_scope / len(gold_titles)
                    if gold_titles else 1.0
                )
            else:
                cr.scope_recall = 0.0

            # --- Capture planner trace ---
            if planner_out:
                rd = planner_out.router_decision
                cr.trace_seeds_found = len(rd.seed_nodes)
                cr.trace_selected_lane = rd.selected_lane.value if rd.selected_lane else ""
                cr.trace_router_confidence = rd.confidence
                cr.trace_intent = (
                    rd.intent.classification.value
                    if rd.intent and rd.intent.classification else ""
                )
                if rd.scope:
                    cr.trace_scope_nodes = len(rd.scope.nodes)
                    cr.trace_scope_documents = len(rd.scope.document_ids)
                    cr.trace_ppr_total_mass = rd.scope.total_ppr_mass
                # Seed details (top 5)
                cr.trace_seed_details = [
                    {
                        "node_id": s.node_id,
                        "similarity": round(s.similarity, 4),
                        "canonical_text": s.canonical_text[:80],
                        "source_doc_id": s.source_doc_id,
                    }
                    for s in rd.seed_nodes[:5]
                ]
                # Ranked chunks from targeted context (top 5)
                if planner_out.targeted_context and planner_out.targeted_context.chunks:
                    cr.trace_ranked_chunks = [
                        {
                            "chunk_id": getattr(rc.chunk, "chunk_id", ""),
                            "source_doc_id": getattr(rc.chunk, "source_doc_id", ""),
                            "relevance_score": round(rc.relevance_score, 4),
                            "text_preview": getattr(rc.chunk, "text", "")[:120],
                        }
                        for rc in planner_out.targeted_context.chunks[:5]
                    ]

            # --- Capture reasoner trace ---
            rt = reasoner_output.trace
            cr.trace_docs_in_scope = rt.documents_in_scope
            cr.trace_relationships_inferred = rt.relationships_inferred
            cr.trace_evidence_grounded = rt.evidence_grounded
            cr.trace_llm_time_ms = rt.llm_time_ms
            cr.trace_total_time_ms = rt.total_time_ms
            cr.trace_raw_response = reasoner_output.raw_response or ""
            cr.trace_verdict_answer = finding.verdict.answer or ""
            cr.trace_verdict_summary = finding.verdict.summary or ""
            cr.trace_reasoning_trace = finding.reasoning_trace or ""

            # Confidence factors
            cf = finding.confidence_factors
            cr.trace_confidence_overall = cf.overall
            cr.trace_confidence_evidence_quality = cf.evidence_quality
            cr.trace_confidence_source_agreement = cf.source_agreement
            cr.trace_confidence_coverage = cf.coverage

            # Relationship details
            cr.trace_relationships_detail = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relationship": r.relationship.value,
                    "confidence": round(r.confidence, 4),
                    "reasoning": r.reasoning[:120],
                }
                for r in finding.relationships
            ]

            # Evidence details
            cr.trace_evidence_detail = [
                {
                    "claim": e.claim[:120],
                    "source_document": e.source_document,
                    "exact_quote": (e.exact_quote or "")[:120],
                }
                for e in finding.evidence
            ]

            # --- Print per-claim diagnostic ---
            check = "\u2713" if cr.correct else "\u2717"
            _print(
                f"  [{ci}] {finding.verdict.verdict_type.value}"
                f"\u2192{cr.mapped_verdict.value} "
                f"(exp: {claim.label.value}) {check}  "
                f"scope_recall={cr.scope_recall:.2f}  "
                f"conf={cr.confidence:.2f}"
            )
            _print(
                f"      planner: seeds={cr.trace_seeds_found} "
                f"scope={cr.trace_scope_nodes}nodes/{cr.trace_scope_documents}docs "
                f"lane={cr.trace_selected_lane} "
                f"ppr_mass={cr.trace_ppr_total_mass:.2f}"
            )
            _print(
                f"      reasoner: docs={cr.trace_docs_in_scope} "
                f"rels={cr.trace_relationships_inferred} "
                f"ev={cr.trace_evidence_grounded} "
                f"llm={cr.trace_llm_time_ms:.0f}ms"
            )
            _print(
                f"      confidence: overall={cr.trace_confidence_overall:.2f} "
                f"ev_quality={cr.trace_confidence_evidence_quality:.2f} "
                f"agreement={cr.trace_confidence_source_agreement:.2f} "
                f"coverage={cr.trace_confidence_coverage:.2f}"
            )
            _print(
                f"      verdict_answer: {cr.trace_verdict_answer[:200]}"
            )
            _print(
                f"      raw_response: {cr.trace_raw_response[:300]}"
            )

        except Exception as e:
            cr.error = f"{type(e).__name__}: {e}"
            _print(f"  [{ci}] ERROR: {cr.error}")

        row_result.claim_results.append(cr)

    return row_result


# ============================================================================
# SUMMARY BUILDER
# ============================================================================

def build_summary(
    row_results: List[WikiRowResult],
    llm_model: str,
    embed_model: str,
) -> WikiRunSummary:
    """Aggregate per-row results into a WikiRunSummary."""

    all_claims: List[WikiClaimResult] = []
    for rr in row_results:
        all_claims.extend(rr.claim_results)

    completed = [c for c in all_claims if c.error is None]
    errored = [c for c in all_claims if c.error is not None]

    verdict_acc = (
        sum(1 for c in completed if c.correct) / len(completed)
        if completed else 0.0
    )
    mean_scope_recall = (
        sum(c.scope_recall for c in completed) / len(completed)
        if completed else 0.0
    )
    mean_confidence = (
        sum(c.confidence for c in completed) / len(completed)
        if completed else 0.0
    )

    # Per-verdict accuracy
    verdicts = [v.value for v in SciFactVerdict]
    per_verdict: Dict[str, Dict[str, Any]] = {}
    for v in verdicts:
        subset = [c for c in completed if c.expected_verdict.value == v]
        if subset:
            correct = sum(1 for c in subset if c.correct)
            per_verdict[v] = {
                "count": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
            }

    # Per question-type accuracy
    per_type: Dict[str, float] = {}
    for qtype in ("comparison", "bridge"):
        subset = [
            c for rr in row_results if rr.question_type == qtype
            for c in rr.claim_results if c.error is None
        ]
        if subset:
            per_type[qtype] = sum(1 for c in subset if c.correct) / len(subset)

    # Per level accuracy
    per_level: Dict[str, float] = {}
    for level in ("medium", "hard"):
        subset = [
            c for rr in row_results if rr.level == level
            for c in rr.claim_results if c.error is None
        ]
        if subset:
            per_level[level] = sum(1 for c in subset if c.correct) / len(subset)

    # Confusion matrix
    cm: Dict[str, Dict[str, int]] = {v: {v2: 0 for v2 in verdicts} for v in verdicts}
    for c in completed:
        if c.mapped_verdict:
            cm[c.expected_verdict.value][c.mapped_verdict.value] += 1

    total_index_time = sum(rr.index_time_seconds for rr in row_results)

    return WikiRunSummary(
        total_rows=len(row_results),
        total_claims=len(all_claims),
        completed_claims=len(completed),
        errored_claims=len(errored),
        verdict_accuracy=verdict_acc,
        mean_scope_recall=mean_scope_recall,
        mean_confidence=mean_confidence,
        per_verdict_accuracy=per_verdict,
        per_type_accuracy=per_type,
        per_level_accuracy=per_level,
        confusion_matrix=cm,
        total_index_time=total_index_time,
        llm_model=llm_model,
        embed_model=embed_model,
        row_results=row_results,
    )


# ============================================================================
# MARKDOWN FORMATTER
# ============================================================================

def format_output_md(summary: WikiRunSummary) -> str:
    """Render WikiFactClaims evaluation results as Markdown."""
    lines: List[str] = []

    lines.append("# WikiFactClaims v5 Evaluation Results\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append(f"- **Pipeline:** v5 (per-row indexing)")
    lines.append(f"- **LLM:** {summary.llm_model}")
    lines.append(f"- **Embedding:** {summary.embed_model}")
    lines.append(f"- **Rows evaluated:** {summary.total_rows}")
    lines.append(f"- **Total claims:** {summary.total_claims} "
                 f"(completed: {summary.completed_claims}, errors: {summary.errored_claims})")
    lines.append(f"- **Total index time:** {summary.total_index_time:.1f}s")
    lines.append(f"- **Timestamp:** {datetime.now().isoformat()}")
    lines.append("")

    # Summary Metrics
    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Verdict Accuracy | {summary.verdict_accuracy:.3f} |")
    lines.append(f"| Planner Scope Recall | {summary.mean_scope_recall:.3f} |")
    lines.append(f"| Mean Confidence | {summary.mean_confidence:.3f} |")
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

    # Per-Type Breakdown
    if summary.per_type_accuracy:
        lines.append("## Per-Type Accuracy\n")
        lines.append("| Question Type | Accuracy |")
        lines.append("|---------------|----------|")
        for qtype, acc in sorted(summary.per_type_accuracy.items()):
            lines.append(f"| {qtype} | {acc:.3f} |")
        lines.append("")

    # Per-Level Breakdown
    if summary.per_level_accuracy:
        lines.append("## Per-Level Accuracy\n")
        lines.append("| Level | Accuracy |")
        lines.append("|-------|----------|")
        for level, acc in sorted(summary.per_level_accuracy.items()):
            lines.append(f"| {level} | {acc:.3f} |")
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

    # Per-Row Summary
    lines.append("## Per-Row Summary\n")
    lines.append("| Row ID | Type | Level | Nodes | Edges | Index(s) | Claims | Correct |")
    lines.append("|--------|------|-------|-------|-------|----------|--------|---------|")
    for rr in summary.row_results:
        completed = [c for c in rr.claim_results if c.error is None]
        correct = sum(1 for c in completed if c.correct)
        total = len(rr.claim_results)
        lines.append(
            f"| {rr.row_id} | {rr.question_type} | {rr.level} "
            f"| {rr.graph_nodes} | {rr.graph_edges} | {rr.index_time_seconds:.1f} "
            f"| {total} | {correct}/{len(completed)} |"
        )
    lines.append("")

    # Per-Claim Detail
    lines.append("## Per-Claim Results\n")
    lines.append("| Row | Idx | Claim (60 chars) | Expected | Predicted | OK | Conf | Scope Recall |")
    lines.append("|-----|-----|------------------|----------|-----------|-----|------|--------------|")
    for rr in summary.row_results:
        for cr in rr.claim_results:
            claim_short = (cr.claim_text[:57] + "...") if len(cr.claim_text) > 60 else cr.claim_text
            if cr.error is not None:
                lines.append(
                    f"| {cr.row_id} | {cr.claim_index} | {claim_short} "
                    f"| {cr.expected_verdict.value} | ERROR | - | - | - |"
                )
            else:
                check = "\u2713" if cr.correct else "\u2717"
                pred = cr.mapped_verdict.value if cr.mapped_verdict else "?"
                lines.append(
                    f"| {cr.row_id} | {cr.claim_index} | {claim_short} "
                    f"| {cr.expected_verdict.value} | {pred} | {check} "
                    f"| {cr.confidence:.2f} | {cr.scope_recall:.2f} |"
                )
    lines.append("")

    # Diagnostic Trace
    lines.append("## Diagnostic Trace\n")
    for rr in summary.row_results:
        lines.append(f"### Row {rr.row_id} ({rr.question_type} / {rr.level})\n")
        lines.append(f"- **Index:** {rr.graph_nodes} nodes, {rr.graph_edges} edges, "
                     f"{rr.graph_bridge_count} bridges, {rr.graph_community_count} communities")
        lines.append(f"- **Extractor:** {rr.extractor_entities} entities, "
                     f"{rr.extractor_claims} claims, "
                     f"{rr.extractor_llm_calls} LLM calls ({rr.extractor_time_seconds:.1f}s)")
        lines.append("")
        for cr in rr.claim_results:
            if cr.error is not None:
                lines.append(f"**Claim {cr.claim_index}:** ERROR — {cr.error}\n")
                continue
            check = "\u2713" if cr.correct else "\u2717"
            pred = cr.mapped_verdict.value if cr.mapped_verdict else "?"
            lines.append(
                f"**Claim {cr.claim_index}:** {cr.predicted_verdict_type.value}"
                f"\u2192{pred} (exp: {cr.expected_verdict.value}) {check}  "
                f"conf={cr.confidence:.2f}  scope_recall={cr.scope_recall:.2f}"
            )
            lines.append("")
            lines.append(
                f"- **Planner:** seeds={cr.trace_seeds_found} "
                f"scope={cr.trace_scope_nodes}nodes/{cr.trace_scope_documents}docs "
                f"lane={cr.trace_selected_lane} "
                f"ppr_mass={cr.trace_ppr_total_mass:.2f} "
                f"router_conf={cr.trace_router_confidence:.2f} "
                f"intent={cr.trace_intent}"
            )
            lines.append(
                f"- **Reasoner:** docs={cr.trace_docs_in_scope} "
                f"rels={cr.trace_relationships_inferred} "
                f"ev={cr.trace_evidence_grounded} "
                f"llm={cr.trace_llm_time_ms:.0f}ms "
                f"total={cr.trace_total_time_ms:.0f}ms"
            )
            lines.append(
                f"- **Confidence:** overall={cr.trace_confidence_overall:.2f} "
                f"ev_quality={cr.trace_confidence_evidence_quality:.2f} "
                f"agreement={cr.trace_confidence_source_agreement:.2f} "
                f"coverage={cr.trace_confidence_coverage:.2f}"
            )
            if cr.trace_verdict_answer:
                lines.append(f"- **Verdict answer:** {cr.trace_verdict_answer[:500]}")
            if cr.trace_verdict_summary:
                lines.append(f"- **Verdict summary:** {cr.trace_verdict_summary[:500]}")
            if cr.trace_raw_response:
                lines.append(f"- **Raw LLM response:** `{cr.trace_raw_response[:500]}`")
            if cr.trace_seed_details:
                lines.append("- **Top seeds:**")
                for sd in cr.trace_seed_details:
                    lines.append(
                        f"  - {sd['node_id']} sim={sd['similarity']:.4f} "
                        f"doc={sd['source_doc_id']} — {sd['canonical_text']}"
                    )
            if cr.trace_ranked_chunks:
                lines.append("- **Top ranked chunks:**")
                for rc in cr.trace_ranked_chunks:
                    lines.append(
                        f"  - {rc['chunk_id']} rel={rc['relevance_score']:.4f} "
                        f"doc={rc['source_doc_id']} — {rc['text_preview'][:80]}"
                    )
            if cr.trace_relationships_detail:
                lines.append("- **Relationships:**")
                for rd in cr.trace_relationships_detail:
                    lines.append(
                        f"  - {rd['source_id']} \u2192 {rd['target_id']} "
                        f"[{rd['relationship']}] conf={rd['confidence']:.4f}"
                    )
            if cr.trace_evidence_detail:
                lines.append("- **Evidence:**")
                for ed in cr.trace_evidence_detail:
                    lines.append(
                        f"  - doc={ed['source_document']} — {ed['claim'][:100]}"
                    )
            lines.append("")
    lines.append("")

    # Error Log
    errors = [
        cr for rr in summary.row_results
        for cr in rr.claim_results if cr.error is not None
    ]
    if errors:
        lines.append("## Error Log\n")
        for cr in errors:
            lines.append(f"- **Row {cr.row_id} claim {cr.claim_index}:** {cr.error}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main(
    limit: Optional[int] = None,
    output_path: str = "wiki_output.md",
    model: Optional[str] = None,
    embed_model: Optional[str] = None,
    use_mock: bool = False,
    data_path: Optional[str] = None,
) -> None:
    """Run the WikiFactClaims v5 per-row evaluation."""
    from config.logging_config import setup_logging, log_file_from_output_path
    setup_logging(log_file=log_file_from_output_path(output_path))

    _print("=" * 70)
    _print("WikiFactClaims v5 Per-Row Evaluation")
    _print("=" * 70)

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

    # 3. Load dataset
    _print("\nLoading WikiFactClaims dataset...")
    loader = WikiFactClaimsLoader(path=data_path)
    rows = loader.get_rows(limit=limit)
    _print(f"  Rows: {len(rows)}" + (f" (limited to {limit})" if limit else ""))
    total_claims = sum(len(r.claims) for r in rows)
    _print(f"  Total claims: {total_claims}")

    # 4. Process each row
    _print(f"\nProcessing {len(rows)} rows...\n")
    row_results: List[WikiRowResult] = []

    try:
        for i, row in enumerate(rows, 1):
            rr = process_row(orchestrator, row, i, len(rows))
            row_results.append(rr)
    except KeyboardInterrupt:
        _print(f"\n\nInterrupted after {len(row_results)} rows. Writing partial results...")
    finally:
        # 5. Build summary and write output
        summary = build_summary(row_results, llm_model_name, embed_model_name)
        md = format_output_md(summary)
        with open(output_path, "w") as f:
            f.write(md)
        _print(f"\nResults written to {output_path}")

        # Print summary to stdout
        _print(f"\n{'=' * 70}")
        _print(f"SUMMARY")
        _print(f"{'=' * 70}")
        _print(f"  Rows:             {summary.total_rows}")
        _print(f"  Claims:           {summary.total_claims} "
               f"({summary.completed_claims} completed, {summary.errored_claims} errors)")
        _print(f"  Verdict Accuracy: {summary.verdict_accuracy:.3f}")
        _print(f"  Scope Recall:     {summary.mean_scope_recall:.3f}")
        _print(f"  Mean Confidence:  {summary.mean_confidence:.3f}")
        _print(f"  Total Index Time: {summary.total_index_time:.1f}s")
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
    parser = argparse.ArgumentParser(
        description="WikiFactClaims v5 Per-Row Evaluation",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of rows to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="wiki_output.md",
        help="Output markdown file path (default: wiki_output.md)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama LLM model name (default: from OLLAMA_MODEL env or qwen2.5:14b)",
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
        "--data-path", type=str, default=None,
        help="Path to wikifactclaims.jsonl (default: data/raw/wikifactclaims/wikifactclaims.jsonl)",
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        output_path=args.output,
        model=args.model,
        embed_model=args.embed_model,
        use_mock=args.mock,
        data_path=args.data_path,
    )
