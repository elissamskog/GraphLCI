"""
Lane Router Module
Responsibility: Select processing lane based on intent and targets.

Deterministic routing logic — no LLM needed. Uses a decision matrix
mapping (intent, graph_availability, target_count) to a LaneType.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from schemas.contracts import (
    DomainType,
    IdentifiedTargets,
    IntentClassification,
    IntentType,
    LaneType,
)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class RoutingInput:
    """Input to the lane router."""
    intent: IntentClassification
    targets: IdentifiedTargets
    corpus_metadata: Dict[str, Any]  # keys: has_graph, domain


@dataclass
class RoutingOutput:
    """Output from the lane router."""
    selected_lane: LaneType
    requires_graph_scope: bool
    domain_expectation: DomainType
    routing_rationale: str


# ============================================================================
# ROUTER
# ============================================================================

class LaneRouter:
    """
    Deterministic lane router.

    Selects a processing lane based on intent classification, target count,
    and graph availability using a fixed decision matrix.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.fallback_lane = LaneType(
            self.config.get('fallback_lane', LaneType.LANE_C_VECTOR_ONLY.value)
        )

    def route(self, input_data: RoutingInput) -> RoutingOutput:
        """
        Main routing entry point.

        1. Low-confidence guard: if confidence < threshold, return fallback lane
        2. Select lane via decision matrix
        3. Set requires_graph_scope based on lane
        4. Infer domain from corpus metadata
        """
        confidence = input_data.intent.confidence
        intent_type = input_data.intent.classification

        # Low-confidence guard
        if confidence < self.confidence_threshold:
            domain = self._infer_domain(input_data.corpus_metadata)
            return RoutingOutput(
                selected_lane=self.fallback_lane,
                requires_graph_scope=False,
                domain_expectation=domain,
                routing_rationale=(
                    f"Confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}; falling back to "
                    f"{self.fallback_lane.value}"
                ),
            )

        has_graph = input_data.corpus_metadata.get('has_graph', False)
        target_count = (
            len(input_data.targets.explicit_docs)
            + len(input_data.targets.implicit_entities)
        )

        selected_lane = self._select_lane(intent_type, has_graph, target_count)
        requires_graph_scope = selected_lane in (
            LaneType.LANE_A_GRAPH_LCI,
            LaneType.LANE_B_GRAPH_ONLY,
        )
        domain = self._infer_domain(input_data.corpus_metadata)

        return RoutingOutput(
            selected_lane=selected_lane,
            requires_graph_scope=requires_graph_scope,
            domain_expectation=domain,
            routing_rationale=(
                f"Intent={intent_type.value}, has_graph={has_graph}, "
                f"target_count={target_count} -> {selected_lane.value}"
            ),
        )

    def _select_lane(
        self,
        intent: IntentType,
        has_graph: bool,
        target_count: int,
    ) -> LaneType:
        """
        Core decision matrix from PLAN.md §3.2.3.

        COMPARISON + graph + targets>=2 -> LANE_A
        COMPARISON otherwise             -> LANE_C
        SENSEMAKING + graph              -> LANE_B
        SENSEMAKING otherwise            -> LANE_C
        LOOKUP                           -> always LANE_C
        """
        if intent == IntentType.COMPARISON:
            if has_graph and target_count >= 2:
                return LaneType.LANE_A_GRAPH_LCI
            return LaneType.LANE_C_VECTOR_ONLY

        if intent == IntentType.SENSEMAKING:
            if has_graph:
                return LaneType.LANE_B_GRAPH_ONLY
            return LaneType.LANE_C_VECTOR_ONLY

        if intent == IntentType.LOOKUP:
            return LaneType.LANE_C_VECTOR_ONLY

        # Safe default
        return LaneType.LANE_C_VECTOR_ONLY

    def _infer_domain(self, corpus_metadata: Dict[str, Any]) -> DomainType:
        """Infer domain from corpus metadata, defaulting to LEGAL."""
        domain_str = corpus_metadata.get('domain', 'LEGAL')
        try:
            return DomainType(domain_str)
        except ValueError:
            return DomainType.LEGAL


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_lane_router() -> None:
    """
    Test the lane router with all decision matrix cells, low-confidence
    fallback, requires_graph_scope validation, and domain inference.
    """
    router = LaneRouter()

    def _make_input(
        intent: IntentType,
        confidence: float,
        explicit_docs: list,
        implicit_entities: list,
        has_graph: bool,
        domain: str = 'LEGAL',
    ) -> RoutingInput:
        return RoutingInput(
            intent=IntentClassification(
                classification=intent,
                confidence=confidence,
                reasoning="test",
            ),
            targets=IdentifiedTargets(
                explicit_docs=explicit_docs,
                implicit_entities=implicit_entities,
            ),
            corpus_metadata={'has_graph': has_graph, 'domain': domain},
        )

    test_cases = [
        # 1. COMPARISON + graph + targets>=2 -> LANE_A
        {
            "name": "COMPARISON with graph and 2+ targets -> LANE_A",
            "input": _make_input(IntentType.COMPARISON, 0.9, ["MSA", "DORA"], [], True),
            "expected_lane": LaneType.LANE_A_GRAPH_LCI,
            "expected_graph_scope": True,
        },
        # 2. COMPARISON + graph + targets<2 -> LANE_C
        {
            "name": "COMPARISON with graph but <2 targets -> LANE_C",
            "input": _make_input(IntentType.COMPARISON, 0.9, ["MSA"], [], True),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 3. COMPARISON + no graph -> LANE_C
        {
            "name": "COMPARISON without graph -> LANE_C",
            "input": _make_input(IntentType.COMPARISON, 0.9, ["MSA", "DORA"], [], False),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 4. SENSEMAKING + graph -> LANE_B
        {
            "name": "SENSEMAKING with graph -> LANE_B",
            "input": _make_input(IntentType.SENSEMAKING, 0.85, [], ["risk"], True),
            "expected_lane": LaneType.LANE_B_GRAPH_ONLY,
            "expected_graph_scope": True,
        },
        # 5. SENSEMAKING + no graph -> LANE_C
        {
            "name": "SENSEMAKING without graph -> LANE_C",
            "input": _make_input(IntentType.SENSEMAKING, 0.85, [], ["risk"], False),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 6. LOOKUP -> always LANE_C
        {
            "name": "LOOKUP always -> LANE_C",
            "input": _make_input(IntentType.LOOKUP, 0.95, [], [], True),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 7. LOOKUP with targets -> still LANE_C
        {
            "name": "LOOKUP with targets -> LANE_C",
            "input": _make_input(IntentType.LOOKUP, 0.9, ["MSA"], ["fee"], True),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 8. Low confidence -> fallback LANE_C
        {
            "name": "Low confidence COMPARISON -> fallback LANE_C",
            "input": _make_input(IntentType.COMPARISON, 0.5, ["MSA", "DORA"], [], True),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 9. Low confidence SENSEMAKING -> fallback LANE_C
        {
            "name": "Low confidence SENSEMAKING -> fallback LANE_C",
            "input": _make_input(IntentType.SENSEMAKING, 0.3, [], ["risk"], True),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
        },
        # 10. Domain inference: REGULATORY
        {
            "name": "Domain inference: REGULATORY",
            "input": _make_input(IntentType.LOOKUP, 0.9, [], [], True, 'REGULATORY'),
            "expected_lane": LaneType.LANE_C_VECTOR_ONLY,
            "expected_graph_scope": False,
            "expected_domain": DomainType.REGULATORY,
        },
        # 11. COMPARISON with implicit entities counting towards targets
        {
            "name": "COMPARISON with implicit entities contributing to target count -> LANE_A",
            "input": _make_input(IntentType.COMPARISON, 0.9, ["MSA"], ["liability"], True),
            "expected_lane": LaneType.LANE_A_GRAPH_LCI,
            "expected_graph_scope": True,
        },
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Lane Router Test Suite")
    print("=" * 70)

    for case in test_cases:
        output = router.route(case["input"])
        lane_ok = output.selected_lane == case["expected_lane"]
        graph_ok = output.requires_graph_scope == case["expected_graph_scope"]

        domain_ok = True
        if "expected_domain" in case:
            domain_ok = output.domain_expectation == case["expected_domain"]

        all_ok = lane_ok and graph_ok and domain_ok
        status = "PASS" if all_ok else "FAIL"

        if all_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {case['name']}")
        print(f"  Expected lane: {case['expected_lane'].value}")
        print(f"  Got lane:      {output.selected_lane.value}")
        print(f"  Expected graph_scope: {case['expected_graph_scope']}")
        print(f"  Got graph_scope:      {output.requires_graph_scope}")
        if "expected_domain" in case:
            print(f"  Expected domain: {case['expected_domain'].value}")
            print(f"  Got domain:      {output.domain_expectation.value}")
        print(f"  Rationale: {output.routing_rationale}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    assert failed == 0, f"{failed} test(s) failed"
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_lane_router()
