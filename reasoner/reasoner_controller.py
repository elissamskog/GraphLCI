"""
REASONER Controller (v5)
Location: reasoner/reasoner_controller.py
Owner: REASONER Instance

Main entry point for the REASONER module. Coordinates prompt building,
epistemic extraction, evidence grounding, and verdict resolution.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

from schemas.contracts import (
    ConfidenceFactors,
    EpistemicRelationship,
    Finding,
    GroundedEvidence,
    InferredRelationship,
    LaneType,
    LLMClient,
    RawDocument,
    ReasonerInput,
    ReasonerOutput,
    ReasonerTrace,
    Scope,
    ScopeSummary,
    TextChunk,
    Verdict,
    VerdictType,
)

from reasoner.prompt_builder import (
    build_focused_prompt,
    build_targeted_prompt,
    build_hybrid_prompt,
)
from reasoner.epistemic_extractor import extract_relationships
from reasoner.evidence_grounder import ground_evidence
from reasoner.verdict_resolver import (
    calculate_confidence,
    determine_verdict_type,
)


def _parse_response_json(raw_response: str) -> Dict[str, Any]:
    """Extract and parse JSON from raw LLM response. Returns empty dict on failure."""
    try:
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


class ReasonerController:
    """Orchestrates the REASONER pipeline.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    config:
        Optional configuration dictionary.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None) -> None:
        self._llm = llm_client
        self._config = config or {}

    def _empty_context_output(
        self,
        input: ReasonerInput,
        selected_lane: LaneType,
        start_time: float,
    ) -> ReasonerOutput:
        """Build a NOT_FOUND output when the lane's context is empty."""
        total_time_ms = (time.time() - start_time) * 1000

        verdict = Verdict(
            verdict_type=VerdictType.NOT_FOUND,
            answer="",
            confidence=0.0,
            summary="No evidence provided to evaluate claim.",
            primary_relationship=None,
        )
        confidence_factors = ConfidenceFactors(
            overall=0.0,
            evidence_quality=0.0,
            source_agreement=0.0,
            coverage=0.0,
        )
        scope_summary = ScopeSummary(
            document_ids=[],
            document_count=0,
            node_count=0,
            edge_count=0,
            bridge_edges_used=0,
        )

        finding = Finding(
            finding_id=str(uuid4()),
            query_id=input.query_id,
            timestamp=datetime.utcnow(),
            verdict=verdict,
            relationships=[],
            evidence=[],
            scope=scope_summary,
            lane_used=selected_lane.value,
            search_strategy=input.router_decision.search_strategy.classification.value,
            confidence_factors=confidence_factors,
            reasoning_trace="Empty context detected — no LLM call made.",
        )

        trace = ReasonerTrace(
            trace_id=str(uuid4()),
            query_id=input.query_id,
            lane_used=selected_lane.value,
            search_strategy=input.router_decision.search_strategy.classification.value,
            documents_in_scope=0,
            relationships_inferred=0,
            evidence_grounded=0,
            verdict_type="NOT_FOUND",
            verdict_confidence=0.0,
            llm_time_ms=0.0,
            total_time_ms=total_time_ms,
            final_prompt=None,
            prompt_token_count=0,
        )

        return ReasonerOutput(
            finding=finding,
            trace=trace,
            raw_response="",
        )

    def process(self, input: ReasonerInput) -> ReasonerOutput:
        """Run the full REASONER pipeline.

        Steps:
        1. Build hybrid prompt from available contexts (LANE_A + LANE_C)
        2. Call LLM for epistemic analysis
        3. Extract relationships from response
        4. Ground evidence to source documents/chunks
        5. Determine verdict and confidence
        6. Apply Epistemic Calibration

        Returns
        -------
        ReasonerOutput
            Finding with verdict, relationships, evidence, and trace.
        """
        start_time = time.time()
        selected_lane = input.router_decision.selected_lane

        # --- Step 1: Build Hybrid Prompt ---
        # We always synthesize focused (A) and targeted (C) if available
        prompt = build_hybrid_prompt(
            input.query,
            focused=input.focused_context,
            targeted=input.targeted_context
        )

        # --- Guard: detect empty context before calling LLM ---
        has_focused = input.focused_context is not None and bool(input.focused_context.documents)
        has_targeted = input.targeted_context is not None and bool(input.targeted_context.chunks)

        if not has_focused and not has_targeted:
            logger.warning("Hybrid Reasoner: 0 context for query: %.80s", input.query.query_text)
            return self._empty_context_output(input, selected_lane, start_time)

        # --- Step 2: Call LLM ---
        llm_start = time.time()
        raw_response = self._llm.generate(prompt)
        llm_time_ms = (time.time() - llm_start) * 1000

        # --- Step 3: Extract relationships ---
        relationships = extract_relationships(raw_response)

        # --- Step 4: Gather documents, chunks, and scope for grounding ---
        documents: Dict[str, RawDocument] = {}
        chunks: Optional[List[TextChunk]] = None
        scope: Optional[Scope] = None

        if input.focused_context:
            documents = input.focused_context.documents
            scope = input.focused_context.scope
        
        if input.targeted_context:
            chunks = [rc.chunk for rc in input.targeted_context.chunks]
            if not scope:
                scope = input.targeted_context.scope

        # --- Step 5: Gather claims for grounding ---
        claims: List[str] = []
        for rel in relationships:
            if rel.evidence_quote:
                claims.append(rel.evidence_quote)

        # Also extract evidence claims from parsed JSON
        parsed = _parse_response_json(raw_response)
        for ev in parsed.get("evidence", []):
            claim = ev.get("exact_quote") or ev.get("claim", "")
            if claim and claim not in claims:
                claims.append(claim)

        # --- Step 6: Ground evidence ---
        evidence = ground_evidence(claims, documents, chunks)

        # --- Step 7: Determine verdict type ---
        verdict_type = determine_verdict_type(relationships, evidence)

        # --- Step 8: Calculate raw confidence ---
        confidence_factors = calculate_confidence(relationships, evidence)
        final_confidence = confidence_factors.overall

        # --- Step 9: Epistemic Calibration ---
        # Cap confidence if claim is temporal/relational but only has LANE_C evidence
        temporal_relational_types = {
            EpistemicRelationship.PRECEDES,
            EpistemicRelationship.SUPERSEDES,
            EpistemicRelationship.EXTENDS,
        }
        has_temporal_relational = any(
            rel.relationship in temporal_relational_types for rel in relationships
        )

        if has_temporal_relational and not has_focused and has_targeted:
            # Cap confidence at 0.4 for temporal/relational claims lacking structural evidence
            final_confidence = min(final_confidence, 0.4)
            logger.info("Epistemic Calibration: Capped confidence for temporal claim without Lane A.")

        # --- Step 10: Build Verdict ---
        verdict_block = parsed.get("verdict", {})
        answer = verdict_block.get("answer", "")
        summary = verdict_block.get("summary", "")

        primary_relationship: Optional[EpistemicRelationship] = None
        if relationships:
            counter = Counter(r.relationship for r in relationships)
            primary_relationship = counter.most_common(1)[0][0]

        verdict = Verdict(
            verdict_type=verdict_type,
            answer=answer,
            confidence=final_confidence,
            summary=summary,
            primary_relationship=primary_relationship,
        )

        # --- Step 11: Build ScopeSummary ---
        if scope:
            bridge_count = sum(
                1 for e in scope.edges.values() if getattr(e, "is_cross_document", False)
            )
            scope_summary = ScopeSummary(
                document_ids=scope.document_ids,
                document_count=len(scope.document_ids),
                node_count=len(scope.nodes),
                edge_count=len(scope.edges),
                bridge_edges_used=bridge_count,
            )
        else:
            scope_summary = ScopeSummary(
                document_ids=[],
                document_count=0,
                node_count=0,
                edge_count=0,
                bridge_edges_used=0,
            )

        # --- Step 12: Extract reasoning trace ---
        reasoning_trace = parsed.get("reasoning_trace", "")

        # --- Step 13: Build Finding ---
        total_time_ms = (time.time() - start_time) * 1000

        finding = Finding(
            finding_id=str(uuid4()),
            query_id=input.query_id,
            timestamp=datetime.utcnow(),
            verdict=verdict,
            relationships=relationships,
            evidence=evidence,
            scope=scope_summary,
            lane_used=selected_lane.value,
            search_strategy=input.router_decision.search_strategy.classification.value,
            confidence_factors=confidence_factors,
            reasoning_trace=reasoning_trace,
        )

        # --- Step 14: Build ReasonerTrace ---
        trace = ReasonerTrace(
            trace_id=str(uuid4()),
            query_id=input.query_id,
            lane_used=selected_lane.value,
            search_strategy=input.router_decision.search_strategy.classification.value,
            documents_in_scope=scope_summary.document_count,
            relationships_inferred=len(relationships),
            evidence_grounded=sum(1 for e in evidence if e.exact_quote),
            verdict_type=verdict_type.value,
            verdict_confidence=final_confidence,
            llm_time_ms=llm_time_ms,
            total_time_ms=total_time_ms,
            final_prompt=prompt,
            prompt_token_count=len(prompt) // 4,
        )

        return ReasonerOutput(
            finding=finding,
            trace=trace,
            raw_response=raw_response,
        )
