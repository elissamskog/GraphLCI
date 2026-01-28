"""
v5 Pipeline Evaluation Feedback Loop
Location: evaluation/v5_evaluator.py
Owner: ROOT Instance

Analyzes v5 module outputs for structural correctness and generates
actionable feedback per module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from schemas.contracts import (
    ExplorerOutput,
    LibrarianOutput,
    PlannerOutput,
    ReasonerOutput,
)


@dataclass
class Feedback:
    """Single evaluation finding."""
    severity: str       # "HIGH" | "MEDIUM" | "LOW"
    module: str         # "EXPLORER" | "LIBRARIAN" | "PLANNER"
    file: str           # path to file needing the fix
    issue: str          # one-line summary
    detail: str         # full explanation
    suggested_fix: str  # actionable guidance


class V5PipelineEvaluator:
    """Evaluate v5 pipeline stage outputs and produce actionable feedback.

    Each ``evaluate_*`` method checks structural correctness of one
    module's output and returns a list of ``Feedback`` items with
    severity (HIGH/MEDIUM/LOW), the responsible module, the relevant
    file, a description of the issue, and a suggested fix.
    """

    # ------------------------------------------------------------------
    # EXPLORER
    # ------------------------------------------------------------------

    def evaluate_explorer(self, output: ExplorerOutput) -> List[Feedback]:
        """Evaluate EXPLORER output for structural completeness."""
        feedback: List[Feedback] = []
        ns = output.node_set

        # Check 1: No nodes produced
        if not ns.nodes:
            feedback.append(Feedback(
                severity="HIGH",
                module="EXPLORER",
                file="explorer/explorer_controller.py",
                issue="Explorer produced zero nodes",
                detail=(
                    "NodeSet.nodes is empty. The explorer should produce "
                    "at least one EmbeddedNode per input document."
                ),
                suggested_fix=(
                    "Verify the chunking and entity extraction pipeline "
                    "produces at least one node per document."
                ),
            ))

        # Check 2: Zero entities
        if ns.entity_count == 0:
            feedback.append(Feedback(
                severity="HIGH",
                module="EXPLORER",
                file="explorer/explorer_controller.py",
                issue="Explorer extracted zero entities",
                detail=(
                    f"NodeSet.entity_count is 0 with {ns.chunk_count} chunks. "
                    "Entity extraction should find at least some entities."
                ),
                suggested_fix=(
                    "Check the entity extraction prompt and LLM response "
                    "parsing in the explorer pipeline."
                ),
            ))

        # Check 3: Zero claims
        if ns.claim_count == 0:
            feedback.append(Feedback(
                severity="MEDIUM",
                module="EXPLORER",
                file="explorer/explorer_controller.py",
                issue="Explorer extracted zero claims",
                detail=(
                    f"NodeSet.claim_count is 0 with {ns.chunk_count} chunks. "
                    "Claim extraction should find at least some claims."
                ),
                suggested_fix=(
                    "Check the claim extraction prompt and ensure the "
                    "documents contain verifiable claims."
                ),
            ))

        # Check 4: Nodes without embeddings
        nodes_without_embedding = [
            n for n in ns.nodes if not n.embedding or len(n.embedding) == 0
        ]
        if nodes_without_embedding:
            feedback.append(Feedback(
                severity="HIGH",
                module="EXPLORER",
                file="explorer/explorer_controller.py",
                issue=f"{len(nodes_without_embedding)} node(s) have no embedding",
                detail=(
                    f"Found {len(nodes_without_embedding)} EmbeddedNode(s) with "
                    "empty or missing embedding vectors. All nodes must have "
                    "embeddings for downstream similarity search."
                ),
                suggested_fix=(
                    "Ensure the embedding client is called for all nodes "
                    "and the results are correctly assigned."
                ),
            ))

        # Check 5: No source documents referenced
        if not ns.source_doc_ids:
            feedback.append(Feedback(
                severity="HIGH",
                module="EXPLORER",
                file="explorer/explorer_controller.py",
                issue="NodeSet has no source_doc_ids",
                detail=(
                    "NodeSet.source_doc_ids is empty. This should list all "
                    "input documents that contributed nodes."
                ),
                suggested_fix=(
                    "Populate source_doc_ids from the documents that were "
                    "chunked during exploration."
                ),
            ))

        return feedback

    # ------------------------------------------------------------------
    # LIBRARIAN
    # ------------------------------------------------------------------

    def evaluate_librarian(self, output: LibrarianOutput) -> List[Feedback]:
        """Evaluate LIBRARIAN output for structural completeness."""
        feedback: List[Feedback] = []
        graph = output.graph

        # Check 1: Empty graph
        if not graph.nodes:
            feedback.append(Feedback(
                severity="HIGH",
                module="LIBRARIAN",
                file="librarian/librarian_controller.py",
                issue="Librarian produced an empty graph (zero nodes)",
                detail=(
                    "KnowledgeGraph.nodes is empty. The librarian should "
                    "index all nodes from the explorer output."
                ),
                suggested_fix=(
                    "Verify LibrarianControllerV5 ingests nodes from "
                    "the LibrarianInput.node_set."
                ),
            ))

        # Check 2: Zero edges
        if not graph.edges:
            feedback.append(Feedback(
                severity="HIGH",
                module="LIBRARIAN",
                file="librarian/librarian_controller.py",
                issue="Librarian produced zero edges",
                detail=(
                    "KnowledgeGraph.edges is empty. The librarian should "
                    "create edges based on synonymy, citations, and "
                    "semantic similarity."
                ),
                suggested_fix=(
                    "Check edge creation logic in the librarian pipeline, "
                    "including synonymy detection and similarity thresholds."
                ),
            ))

        # Check 3: No bridge edges
        if graph.bridge_edges is None or len(graph.bridge_edges) == 0:
            feedback.append(Feedback(
                severity="MEDIUM",
                module="LIBRARIAN",
                file="librarian/librarian_controller.py",
                issue="No bridge edges identified",
                detail=(
                    "KnowledgeGraph.bridge_edges is empty. Bridge edges "
                    "connect different document communities and are "
                    "critical for cross-document reasoning."
                ),
                suggested_fix=(
                    "Verify the bridge edge identification logic runs "
                    "after graph construction and correctly identifies "
                    "edges connecting different document subgraphs."
                ),
            ))

        # Check 4: Empty adjacency
        if not graph.adjacency:
            feedback.append(Feedback(
                severity="HIGH",
                module="LIBRARIAN",
                file="librarian/librarian_controller.py",
                issue="Empty adjacency map",
                detail=(
                    "KnowledgeGraph.adjacency is empty. The adjacency "
                    "map must be populated for BFS/PPR traversal."
                ),
                suggested_fix=(
                    "Ensure the adjacency map is built from the edge "
                    "list during graph construction."
                ),
            ))

        # Check 5: node_by_doc missing documents
        if not graph.node_by_doc:
            feedback.append(Feedback(
                severity="MEDIUM",
                module="LIBRARIAN",
                file="librarian/librarian_controller.py",
                issue="Empty node_by_doc mapping",
                detail=(
                    "KnowledgeGraph.node_by_doc is empty. This mapping "
                    "is needed for document-level coverage analysis."
                ),
                suggested_fix=(
                    "Populate node_by_doc by grouping graph nodes by "
                    "their source_doc_id during graph construction."
                ),
            ))

        return feedback

    # ------------------------------------------------------------------
    # PLANNER
    # ------------------------------------------------------------------

    def evaluate_planner(self, output: PlannerOutput) -> List[Feedback]:
        """Evaluate PLANNER output for structural completeness."""
        feedback: List[Feedback] = []
        rd = output.router_decision

        # Check 1: No seed nodes
        if not rd.seed_nodes:
            feedback.append(Feedback(
                severity="HIGH",
                module="PLANNER",
                file="planner_controller.py",
                issue="Planner found zero seed nodes",
                detail=(
                    "RouterDecisionV5.seed_nodes is empty. The planner "
                    "should find at least one seed node via embedding "
                    "similarity to the query."
                ),
                suggested_fix=(
                    "Check query embedding and similarity search against "
                    "the graph's node embeddings. Lower the similarity "
                    "threshold if needed."
                ),
            ))

        # Check 2: No scope
        if rd.scope is None:
            feedback.append(Feedback(
                severity="HIGH",
                module="PLANNER",
                file="planner_controller.py",
                issue="Planner produced no scope",
                detail=(
                    "RouterDecisionV5.scope is None. PPR expansion should "
                    "produce a scope from the seed nodes."
                ),
                suggested_fix=(
                    "Verify PPR expansion runs on the seed nodes and "
                    "produces a non-empty scope."
                ),
            ))
        else:
            # Check 3: Scope with zero nodes
            if not rd.scope.nodes:
                feedback.append(Feedback(
                    severity="HIGH",
                    module="PLANNER",
                    file="planner_controller.py",
                    issue="Scope contains zero nodes",
                    detail=(
                        "Scope.nodes is empty after PPR expansion. "
                        "The scope should contain at least the seed nodes."
                    ),
                    suggested_fix=(
                        "Ensure PPR expansion includes the seed nodes "
                        "in the output scope."
                    ),
                ))

            # Check 4: No documents in scope
            if not rd.scope.document_ids:
                feedback.append(Feedback(
                    severity="MEDIUM",
                    module="PLANNER",
                    file="planner_controller.py",
                    issue="Scope has no document_ids",
                    detail=(
                        "Scope.document_ids is empty. The scope should "
                        "list documents covered by scoped nodes."
                    ),
                    suggested_fix=(
                        "Derive document_ids from the source_doc_id of "
                        "each ScopedNode in the scope."
                    ),
                ))

        # Check 5: No context assembled for Lane A
        from schemas.contracts import LaneType
        if rd.selected_lane == LaneType.LANE_A_GRAPH_LCI and output.focused_context is None:
            feedback.append(Feedback(
                severity="HIGH",
                module="PLANNER",
                file="planner_controller.py",
                issue="Lane A selected but no focused_context assembled",
                detail=(
                    "The planner selected LANE_A_GRAPH_LCI but "
                    "PlannerOutput.focused_context is None."
                ),
                suggested_fix=(
                    "Ensure the context assembly step runs for Lane A "
                    "queries and produces a FocusedContext."
                ),
            ))

        if rd.selected_lane == LaneType.LANE_B_GRAPH_ONLY and output.global_context is None:
            feedback.append(Feedback(
                severity="HIGH",
                module="PLANNER",
                file="planner_controller.py",
                issue="Lane B selected but no global_context assembled",
                detail=(
                    "The planner selected LANE_B_GRAPH_ONLY but "
                    "PlannerOutput.global_context is None."
                ),
                suggested_fix=(
                    "Ensure the context assembly step runs for Lane B "
                    "queries and produces a GlobalContext."
                ),
            ))

        if rd.selected_lane == LaneType.LANE_C_VECTOR_ONLY and output.targeted_context is None:
            feedback.append(Feedback(
                severity="HIGH",
                module="PLANNER",
                file="planner_controller.py",
                issue="Lane C selected but no targeted_context assembled",
                detail=(
                    "The planner selected LANE_C_VECTOR_ONLY but "
                    "PlannerOutput.targeted_context is None."
                ),
                suggested_fix=(
                    "Ensure the context assembly step runs for Lane C "
                    "queries and produces a TargetedContext."
                ),
            ))

        return feedback

    # ------------------------------------------------------------------
    # REASONER
    # ------------------------------------------------------------------

    def evaluate_reasoner(self, output: ReasonerOutput) -> List[Feedback]:
        """Evaluate REASONER output for structural completeness."""
        feedback: List[Feedback] = []
        finding = output.finding

        # Check 1: No relationships
        if not finding.relationships:
            feedback.append(Feedback(
                severity="HIGH",
                module="REASONER",
                file="reasoner/reasoner_controller.py",
                issue="Reasoner inferred zero relationships",
                detail=(
                    "Finding.relationships is empty. The reasoner should "
                    "infer at least one epistemic relationship from the "
                    "evidence."
                ),
                suggested_fix=(
                    "Check the relationship inference prompt and ensure "
                    "the reasoner receives non-empty context."
                ),
            ))

        # Check 2: No evidence
        if not finding.evidence:
            feedback.append(Feedback(
                severity="HIGH",
                module="REASONER",
                file="reasoner/reasoner_controller.py",
                issue="Reasoner produced zero evidence items",
                detail=(
                    "Finding.evidence is empty. The reasoner should "
                    "ground its verdict in specific evidence."
                ),
                suggested_fix=(
                    "Check the evidence extraction logic in the reasoner "
                    "and ensure it populates GroundedEvidence items."
                ),
            ))

        # Check 3: Ungrounded evidence (no exact_quote)
        if finding.evidence:
            ungrounded = [
                e for e in finding.evidence
                if not e.exact_quote or not e.exact_quote.strip()
            ]
            if len(ungrounded) > len(finding.evidence) * 0.5:
                feedback.append(Feedback(
                    severity="MEDIUM",
                    module="REASONER",
                    file="reasoner/reasoner_controller.py",
                    issue=f"{len(ungrounded)}/{len(finding.evidence)} evidence items lack exact_quote",
                    detail=(
                        "More than half of evidence items have empty "
                        "exact_quote. Grounded evidence improves "
                        "auditability."
                    ),
                    suggested_fix=(
                        "Ensure the reasoner prompt instructs the LLM to "
                        "provide exact quotations from source documents."
                    ),
                ))

        # Check 4: Zero confidence
        if finding.confidence_factors.overall <= 0:
            feedback.append(Feedback(
                severity="MEDIUM",
                module="REASONER",
                file="reasoner/reasoner_controller.py",
                issue="Overall confidence is zero or negative",
                detail=(
                    f"ConfidenceFactors.overall = "
                    f"{finding.confidence_factors.overall}. A valid "
                    "finding should have positive confidence."
                ),
                suggested_fix=(
                    "Check the confidence computation logic. Ensure "
                    "evidence quality, source agreement, and coverage "
                    "factors are computed correctly."
                ),
            ))

        # Check 5: Empty reasoning trace
        if not finding.reasoning_trace or not finding.reasoning_trace.strip():
            feedback.append(Feedback(
                severity="LOW",
                module="REASONER",
                file="reasoner/reasoner_controller.py",
                issue="Empty reasoning trace",
                detail=(
                    "Finding.reasoning_trace is empty. The trace improves "
                    "debuggability and auditability of the verdict."
                ),
                suggested_fix=(
                    "Ensure the reasoner captures and stores the LLM's "
                    "reasoning chain in the finding."
                ),
            ))

        return feedback
