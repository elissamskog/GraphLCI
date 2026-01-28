"""
Two-Stage Metric Tracking: Retrieval vs. Reasoning Failure Isolation
Location: evaluation/two_stage_metrics.py
Owner: ROOT Instance

Isolates whether errors come from the Graph/Retrieval layer (LIBRARIAN/PLANNER)
or the Reasoning layer (REASONER).

Failure taxonomy:
    SUCCESS              - gold docs in scope AND verdict correct
    RETRIEVAL_FAILURE    - gold docs NOT in scope
      LIBRARIAN_FAILURE  - gold doc has NO nodes in KnowledgeGraph
      PLANNER_FAILURE    - gold doc has nodes but PPR did not include them
    REASONING_FAILURE    - gold docs in scope but verdict wrong
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from schemas.contracts import (
    KnowledgeGraph,
    PlannerOutput,
    ReasonerOutput,
)
from schemas.evaluation_contracts import (
    FailureStage,
    GraphRecallResult,
    RetrievalFailureAttribution,
    StageAResult,
    StageBResult,
    TwoStageReport,
    TwoStageResult,
)


def evaluate_stage_a(
    case_id: str,
    gold_doc_ids: List[str],
    graph: KnowledgeGraph,
    planner_output: Optional[PlannerOutput],
) -> StageAResult:
    """Stage A: Check whether gold documents made it into the retrieval scope.

    For each gold_doc_id:
      1. Check graph.node_by_doc for presence -> in_graph, node_count_in_graph
      2. Check scope.document_ids for presence -> in_scope
      3. Count intersection of doc's graph nodes with scope.nodes -> node_count_in_scope
      4. Attribution: in_scope -> NOT_APPLICABLE,
                      not in graph -> LIBRARIAN_FAILURE,
                      in graph but not in scope -> PLANNER_FAILURE

    Edge case: empty gold_doc_ids -> vacuous pass (recall=1.0)
    """
    # Handle empty gold_doc_ids (e.g. NOT_ENOUGH_INFO)
    if not gold_doc_ids:
        return StageAResult(
            gold_doc_ids=[],
            scope_doc_ids=_get_scope_doc_ids(planner_output),
            per_doc_results=[],
            graph_recall_at_k=1.0,
            all_gold_in_scope=True,
            failure_attribution=RetrievalFailureAttribution.NOT_APPLICABLE,
            has_scope=_has_scope(planner_output),
        )

    has_scope = _has_scope(planner_output)
    scope_doc_ids = _get_scope_doc_ids(planner_output)
    scope_doc_set = set(scope_doc_ids)
    scope_node_ids = _get_scope_node_ids(planner_output)

    per_doc_results: List[GraphRecallResult] = []
    gold_in_scope_count = 0

    for doc_id in gold_doc_ids:
        # Check graph presence
        graph_node_ids = graph.node_by_doc.get(doc_id, [])
        in_graph = len(graph_node_ids) > 0
        node_count_in_graph = len(graph_node_ids)

        # Check scope presence
        in_scope = doc_id in scope_doc_set

        # Count nodes from this doc that are in scope
        node_count_in_scope = len(
            set(graph_node_ids) & scope_node_ids
        ) if in_graph else 0

        # Attribution
        if in_scope:
            attribution = RetrievalFailureAttribution.NOT_APPLICABLE
            gold_in_scope_count += 1
        elif not in_graph:
            attribution = RetrievalFailureAttribution.LIBRARIAN_FAILURE
        else:
            attribution = RetrievalFailureAttribution.PLANNER_FAILURE

        per_doc_results.append(GraphRecallResult(
            gold_doc_id=doc_id,
            in_graph=in_graph,
            in_scope=in_scope,
            node_count_in_graph=node_count_in_graph,
            node_count_in_scope=node_count_in_scope,
            attribution=attribution,
        ))

    graph_recall_at_k = gold_in_scope_count / len(gold_doc_ids)
    all_gold_in_scope = gold_in_scope_count == len(gold_doc_ids)

    # Aggregate failure attribution: worst failure wins
    if all_gold_in_scope:
        agg_attribution = RetrievalFailureAttribution.NOT_APPLICABLE
    else:
        has_librarian = any(
            r.attribution == RetrievalFailureAttribution.LIBRARIAN_FAILURE
            for r in per_doc_results
        )
        has_planner = any(
            r.attribution == RetrievalFailureAttribution.PLANNER_FAILURE
            for r in per_doc_results
        )
        if has_librarian and has_planner:
            agg_attribution = RetrievalFailureAttribution.LIBRARIAN_FAILURE
        elif has_librarian:
            agg_attribution = RetrievalFailureAttribution.LIBRARIAN_FAILURE
        else:
            agg_attribution = RetrievalFailureAttribution.PLANNER_FAILURE

    return StageAResult(
        gold_doc_ids=gold_doc_ids,
        scope_doc_ids=scope_doc_ids,
        per_doc_results=per_doc_results,
        graph_recall_at_k=graph_recall_at_k,
        all_gold_in_scope=all_gold_in_scope,
        failure_attribution=agg_attribution,
        has_scope=has_scope,
    )


def evaluate_stage_b(
    case_id: str,
    expected_verdict: str,
    reasoner_output: Optional[ReasonerOutput],
    stage_a_passed: bool,
    verdict_mapper: Optional[Callable] = None,
) -> Optional[StageBResult]:
    """Stage B: Check whether the verdict is correct, given retrieval succeeded.

    If Stage A failed, returns None (skip reasoning audit).
    If no ReasonerOutput, returns verdict_correct=False with actual_verdict="NO_OUTPUT".
    """
    if not stage_a_passed:
        return None

    if reasoner_output is None:
        return StageBResult(
            expected_verdict=expected_verdict,
            actual_verdict="NO_OUTPUT",
            verdict_correct=False,
            confidence=0.0,
            stage_a_passed=True,
        )

    finding = reasoner_output.finding
    actual_raw = finding.verdict.verdict_type

    if verdict_mapper is not None:
        actual_verdict = str(verdict_mapper(actual_raw))
    else:
        actual_verdict = actual_raw.value

    verdict_correct = actual_verdict == expected_verdict
    confidence = finding.verdict.confidence

    return StageBResult(
        expected_verdict=expected_verdict,
        actual_verdict=actual_verdict,
        verdict_correct=verdict_correct,
        confidence=confidence,
        stage_a_passed=True,
    )


def evaluate_two_stage(
    case_id: str,
    gold_doc_ids: List[str],
    graph: KnowledgeGraph,
    planner_output: Optional[PlannerOutput],
    reasoner_output: Optional[ReasonerOutput],
    expected_verdict: str,
    verdict_mapper: Optional[Callable] = None,
) -> TwoStageResult:
    """Run both Stage A (retrieval) and Stage B (reasoning) and classify failure.

    Parameters
    ----------
    case_id : str
    gold_doc_ids : list of doc IDs that should be in scope
    graph : KnowledgeGraph from the LIBRARIAN
    planner_output : PlannerOutput (may be None on pipeline error)
    reasoner_output : ReasonerOutput (may be None on pipeline error)
    expected_verdict : str — the expected verdict to compare against
    verdict_mapper : optional callable to map VerdictType -> comparable value
    """
    stage_a = evaluate_stage_a(case_id, gold_doc_ids, graph, planner_output)

    stage_b = evaluate_stage_b(
        case_id,
        expected_verdict,
        reasoner_output,
        stage_a_passed=stage_a.all_gold_in_scope,
        verdict_mapper=verdict_mapper,
    )

    # Classify failure stage
    if not stage_a.all_gold_in_scope:
        failure_stage = FailureStage.RETRIEVAL_FAILURE
        retrieval_attribution = stage_a.failure_attribution
    elif stage_b is not None and stage_b.verdict_correct:
        failure_stage = FailureStage.SUCCESS
        retrieval_attribution = RetrievalFailureAttribution.NOT_APPLICABLE
    else:
        failure_stage = FailureStage.REASONING_FAILURE
        retrieval_attribution = RetrievalFailureAttribution.NOT_APPLICABLE

    return TwoStageResult(
        failure_stage=failure_stage,
        stage_a=stage_a,
        stage_b=stage_b,
        retrieval_attribution=retrieval_attribution,
    )


def build_two_stage_report(results: List[TwoStageResult]) -> TwoStageReport:
    """Aggregate TwoStageResults across all cases into a TwoStageReport."""
    total = len(results)
    if total == 0:
        return TwoStageReport(
            total_cases=0,
            success_count=0,
            retrieval_failure_count=0,
            reasoning_failure_count=0,
            librarian_failure_count=0,
            planner_failure_count=0,
            success_rate=0.0,
            retrieval_failure_rate=0.0,
            reasoning_failure_rate=0.0,
            graph_recall_at_k_mean=0.0,
            verdict_accuracy_when_retrieved=0.0,
        )

    success_count = sum(1 for r in results if r.failure_stage == FailureStage.SUCCESS)
    retrieval_failure_count = sum(
        1 for r in results if r.failure_stage == FailureStage.RETRIEVAL_FAILURE
    )
    reasoning_failure_count = sum(
        1 for r in results if r.failure_stage == FailureStage.REASONING_FAILURE
    )

    librarian_failure_count = sum(
        1 for r in results
        if r.retrieval_attribution == RetrievalFailureAttribution.LIBRARIAN_FAILURE
    )
    planner_failure_count = sum(
        1 for r in results
        if r.retrieval_attribution == RetrievalFailureAttribution.PLANNER_FAILURE
    )

    graph_recall_at_k_mean = sum(
        r.stage_a.graph_recall_at_k for r in results
    ) / total

    # Verdict accuracy when retrieved: among cases where Stage A passed
    stage_b_cases = [r for r in results if r.stage_b is not None]
    if stage_b_cases:
        verdict_accuracy_when_retrieved = sum(
            1 for r in stage_b_cases if r.stage_b.verdict_correct
        ) / len(stage_b_cases)
    else:
        verdict_accuracy_when_retrieved = 0.0

    return TwoStageReport(
        total_cases=total,
        success_count=success_count,
        retrieval_failure_count=retrieval_failure_count,
        reasoning_failure_count=reasoning_failure_count,
        librarian_failure_count=librarian_failure_count,
        planner_failure_count=planner_failure_count,
        success_rate=success_count / total,
        retrieval_failure_rate=retrieval_failure_count / total,
        reasoning_failure_rate=reasoning_failure_count / total,
        graph_recall_at_k_mean=graph_recall_at_k_mean,
        verdict_accuracy_when_retrieved=verdict_accuracy_when_retrieved,
    )


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _has_scope(planner_output: Optional[PlannerOutput]) -> bool:
    """Check if planner output has a scope."""
    if planner_output is None:
        return False
    return planner_output.router_decision.scope is not None


def _get_scope_doc_ids(planner_output: Optional[PlannerOutput]) -> List[str]:
    """Extract document_ids from planner scope, or empty list."""
    if planner_output is None:
        return []
    scope = planner_output.router_decision.scope
    if scope is None:
        return []
    return list(scope.document_ids)


def _get_scope_node_ids(planner_output: Optional[PlannerOutput]) -> set:
    """Extract node IDs from planner scope, or empty set."""
    if planner_output is None:
        return set()
    scope = planner_output.router_decision.scope
    if scope is None:
        return set()
    return set(scope.nodes.keys())
