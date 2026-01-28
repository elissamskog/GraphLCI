"""
KnowledgeAudit Orchestrator
Location: orchestrator.py
Owner: ROOT Instance

Coordinates PlannerController and LibrarianController via direct in-process
imports. Includes a self-contained test function with a mock LLM.
"""

from __future__ import annotations

import json
import logging
import sys
import os
import time as _time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- sys.path setup ----------------------------------------------------------
# Project root: resolves schemas.*, librarian.*
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# planner/ directory: so planner_controller's inner `from planner.intent_classifier`
# resolves to planner/planner/ (the inner package with __init__.py)
_PLANNER_DIR = os.path.join(_PROJECT_ROOT, "planner")
if _PLANNER_DIR not in sys.path:
    sys.path.insert(0, _PLANNER_DIR)

from schemas.contracts import (
    AssembledContext,
    ExplorationRequest,
    ExplorationResponse,
    ExplorationResult,
    IndexedCorpus,
    IndexRequest,
    LLMClient,
    LibrarianRequest,
    PlannerRequest,
    QueryRequest,
    RawCorpus,
    RawDocument,
    RouterDecision,
    ScopeRequest,
    # v5 contracts
    ChunkerInput,
    ChunkerOutput,
    EmbeddingClient,
    ExplorerInput,
    ExplorerOutput,
    ExtractorInput,
    LibrarianInput,
    LibrarianOutput,
    PlannerInput,
    PlannerOutput,
    ReasonerInput,
    ReasonerOutput,
    Finding,
)
from planner_controller import PlannerController
from librarian.librarian_controller import LibrarianController
from explorer.explorer_controller import ExplorerController
from llm_client import get_client, get_embedding_client, OllamaClient

# v5 controllers
from explorer.explorer_controller import ExplorerControllerV5
from chunker.chunker_controller import ChunkerController
from extractor.extractor_controller import ExtractorController
from librarian.librarian_controller import LibrarianControllerV5
from planner_controller import PlannerControllerV5
from reasoner.reasoner_controller import ReasonerController as ReasonerControllerV5
from librarian.similarity_index import SimilarityIndex


logger = logging.getLogger("knowledgeaudit.orchestrator")


def _trace_summary(trace: Any) -> str:
    """Format a trace dataclass into a single-line summary for DEBUG logging."""
    if trace is None:
        return "(no trace)"
    fields = []
    for attr in sorted(vars(trace)):
        if attr.startswith("_"):
            continue
        val = getattr(trace, attr)
        if isinstance(val, (list, dict)):
            fields.append(f"{attr}={len(val)}")
        elif isinstance(val, float):
            fields.append(f"{attr}={val:.3f}")
        else:
            fields.append(f"{attr}={val}")
    return " ".join(fields)


# ============================================================================
# ERROR CLASSES
# ============================================================================

class PlanningError(Exception):
    """Raised when PlannerResponse.success is False."""


class IndexingError(Exception):
    """Raised when INDEX LibrarianResponse.success is False."""


class ExplorationError(Exception):
    """Raised when ExplorationResponse.success is False."""


class AssemblyError(Exception):
    """Raised when RESOLVE_AND_ASSEMBLE LibrarianResponse.success is False."""


# ============================================================================
# TOKEN COUNTING WRAPPERS (v5 evaluation)
# ============================================================================

class TokenCountingWrapper:
    """Wraps an LLMClient to accumulate per-stage token usage."""

    def __init__(self, client):
        self._client = client
        self._accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        result = self._client.generate(prompt)
        self._call_count += 1
        usage = getattr(self._client, "last_usage", None)
        if usage:
            self._accumulated["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._accumulated["completion_tokens"] += usage.get("completion_tokens", 0)
            self._accumulated["total_tokens"] += usage.get("total_tokens", 0)
        return result

    def snapshot_and_reset(self) -> Dict[str, int]:
        result = dict(self._accumulated)
        result["call_count"] = self._call_count
        self._accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._call_count = 0
        return result

    def __getattr__(self, name):
        return getattr(self._client, name)


class EmbeddingCountingWrapper:
    """Wraps an EmbeddingClient to accumulate per-stage token usage."""

    def __init__(self, client):
        self._client = client
        self._accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._call_count = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        result = self._client.embed(texts)
        self._call_count += 1
        usage = getattr(self._client, "last_usage", None)
        if usage:
            self._accumulated["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._accumulated["completion_tokens"] += usage.get("completion_tokens", 0)
            self._accumulated["total_tokens"] += usage.get("total_tokens", 0)
        return result

    def snapshot_and_reset(self) -> Dict[str, int]:
        result = dict(self._accumulated)
        result["call_count"] = self._call_count
        self._accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._call_count = 0
        return result

    def __getattr__(self, name):
        return getattr(self._client, name)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class Orchestrator:
    """Coordinate PlannerController and LibrarianController.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    config:
        Dictionary with keys consumed by planner and librarian.
    """

    def __init__(self, llm_client: LLMClient, config: Dict[str, Any]) -> None:
        self._config = config
        self._last_exploration_result: Optional[ExplorationResult] = None

        # -- Planner config --
        self._planner_config: Dict[str, Any] = {
            "available_documents": config.get("available_documents", []),
            "known_entities": config.get("known_entities", []),
            "corpus_metadata": config.get("corpus_metadata", {}),
        }
        self._planner = PlannerController(self._planner_config)

        # -- Explorer config --
        explorer_config: Dict[str, Any] = config.get("explorer", {})
        self._explorer = ExplorerController(llm_client, explorer_config)

        # -- Librarian config --
        librarian_config: Dict[str, Any] = {
            "indexer": config.get("indexer", {}),
            "resolver": config.get("resolver", {}),
            "assembler": config.get("assembler", {}),
        }
        self._librarian = LibrarianController(llm_client, librarian_config)

    # ------------------------------------------------------------------
    # INDEX pipeline
    # ------------------------------------------------------------------

    def index_corpus(self, corpus: RawCorpus) -> IndexedCorpus:
        """Index a corpus of documents.

        Two-stage pipeline:
        1. **Exploration** — EXPLORER produces document summaries and a
           domain taxonomy.
        2. **Indexing** — LIBRARIAN indexes the corpus, enriched by
           exploration results.

        After indexing, updates the planner config with the newly-available
        document IDs so that TargetExtractor can match them.
        """
        pipeline_t0 = _time.monotonic()
        doc_count = len(corpus.documents)
        logger.info("index_corpus START — %d documents", doc_count)

        # Stage 1: Exploration
        logger.info("EXPLORER START — %d documents", doc_count)
        t0 = _time.monotonic()
        try:
            exploration_request = ExplorationRequest(corpus=corpus)
            exploration_response = self._explorer.process(exploration_request)
        except Exception:
            logger.exception("EXPLORER FAILED")
            raise
        elapsed = _time.monotonic() - t0
        if not exploration_response.success:
            logger.error("EXPLORER returned errors: %s", exploration_response.errors)
            raise ExplorationError(
                f"Exploration failed: {exploration_response.errors}"
            )
        exploration_result = exploration_response.result
        self._last_exploration_result = exploration_result
        logger.info("EXPLORER DONE — %.2fs", elapsed)

        # Stage 2: Indexing with exploration context
        logger.info("LIBRARIAN INDEX START")
        t0 = _time.monotonic()
        try:
            index_request = IndexRequest(corpus=corpus, exploration=exploration_result)
            request = LibrarianRequest(request_type="INDEX", payload=index_request)
            response = self._librarian.process(request)
        except Exception:
            logger.exception("LIBRARIAN INDEX FAILED")
            raise
        elapsed = _time.monotonic() - t0

        if not response.success:
            logger.error("LIBRARIAN INDEX returned errors: %s", response.errors)
            raise IndexingError(
                f"Indexing failed: {response.errors}"
            )

        node_count = len(response.payload.graph_state.nodes)
        edge_count = len(response.payload.graph_state.edges)
        logger.info(
            "LIBRARIAN INDEX DONE — nodes=%d edges=%d elapsed=%.2fs",
            node_count, edge_count, elapsed,
        )

        # Update planner with newly-available doc IDs
        doc_ids = [doc.doc_id for doc in corpus.documents]
        self._planner_config["available_documents"] = doc_ids
        self._planner = PlannerController(self._planner_config)

        total_elapsed = _time.monotonic() - pipeline_t0
        logger.info("index_corpus DONE — total elapsed=%.2fs", total_elapsed)
        return response.payload

    @property
    def last_exploration_result(self) -> Optional[ExplorationResult]:
        """The most recent ``ExplorationResult`` from ``index_corpus``."""
        return self._last_exploration_result

    # ------------------------------------------------------------------
    # QUERY pipeline
    # ------------------------------------------------------------------

    def process_query(
        self, query: QueryRequest
    ) -> Tuple[RouterDecision, Optional[AssembledContext]]:
        """Route a query through planner and optionally assemble context.

        Returns
        -------
        (RouterDecision, AssembledContext | None)
            The router decision and, if graph scope is required, the
            assembled LCI context.
        """
        pipeline_t0 = _time.monotonic()
        logger.info("process_query START — query_id=%s", query.query_id)

        # Step 1: Plan
        logger.info("PLANNER START — query_id=%s", query.query_id)
        t0 = _time.monotonic()
        try:
            planner_request = PlannerRequest(query_request=query)
            planner_response = self._planner.process(planner_request)
        except Exception:
            logger.exception("PLANNER FAILED")
            raise
        elapsed = _time.monotonic() - t0

        if not planner_response.success:
            logger.error("PLANNER returned errors: %s", planner_response.errors)
            raise PlanningError(
                f"Planning failed: {planner_response.errors}"
            )

        router_decision = planner_response.router_decision
        logger.info(
            "PLANNER DONE — lane=%s intent=%s scope=%s elapsed=%.2fs",
            router_decision.selected_lane.value,
            router_decision.intent.classification.value,
            router_decision.requires_graph_scope,
            elapsed,
        )

        # Step 2: Optionally resolve scope and assemble context
        if router_decision.requires_graph_scope:
            logger.info("LIBRARIAN RESOLVE_AND_ASSEMBLE START — query_id=%s", query.query_id)
            t0 = _time.monotonic()
            try:
                scope_request = ScopeRequest(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    router_decision=router_decision,
                    traversal_config=self._config.get("traversal_config"),
                )
                lib_request = LibrarianRequest(
                    request_type="RESOLVE_AND_ASSEMBLE",
                    payload=scope_request,
                )
                lib_response = self._librarian.process(lib_request)
            except Exception:
                logger.exception("LIBRARIAN RESOLVE_AND_ASSEMBLE FAILED")
                raise
            elapsed = _time.monotonic() - t0

            if not lib_response.success:
                logger.error("LIBRARIAN ASSEMBLY returned errors: %s", lib_response.errors)
                raise AssemblyError(
                    f"Assembly failed: {lib_response.errors}"
                )

            logger.info("LIBRARIAN RESOLVE_AND_ASSEMBLE DONE — elapsed=%.2fs", elapsed)
            total_elapsed = _time.monotonic() - pipeline_t0
            logger.info("process_query DONE — total elapsed=%.2fs", total_elapsed)
            return router_decision, lib_response.payload

        total_elapsed = _time.monotonic() - pipeline_t0
        logger.info("process_query DONE — no graph scope — total elapsed=%.2fs", total_elapsed)
        return router_decision, None


# ============================================================================
# v5 ORCHESTRATOR
# ============================================================================

class ReasoningError(Exception):
    """Raised when ReasonerController fails."""


class OrchestratorV5:
    """v5 pipeline orchestrator.

    Pipeline:
      Index:  CHUNKER(split) → EXTRACTOR(NER+embed) → LIBRARIAN(edges+communities) → [INDEX]
      Query:  PLANNER(scope+route+assemble) → REASONER(verdict)

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    embedding_client:
        Object satisfying the ``EmbeddingClient`` protocol.
    config:
        Configuration dictionary.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        config: Dict[str, Any],
    ) -> None:
        self._llm_raw = llm_client
        self._embedding_raw = embedding_client
        self._llm = TokenCountingWrapper(llm_client)
        self._embedding = EmbeddingCountingWrapper(embedding_client)
        self._config = config

        # Controllers are lazily imported to avoid circular deps
        # and because the stubs may not be fully implemented yet.
        self._index: Optional[LibrarianOutput] = None

        # Stashed intermediate outputs for evaluation
        self._last_chunker_output: Optional[ChunkerOutput] = None
        self._last_explorer_output: Optional[ExplorerOutput] = None
        self._last_planner_output: Optional[PlannerOutput] = None

        # Per-stage performance stashes (v5 evaluation)
        self._stage_timings: Dict[str, float] = {}
        self._stage_token_usage: Dict[str, Dict] = {}

    def index_corpus_v5(self, documents: List[RawDocument]) -> LibrarianOutput:
        """Index a corpus through the v5 pipeline.

        Steps:
        1. CHUNKER: split documents into text chunks (pure Python)
        2. EXTRACTOR: extract entities/claims, resolve coreferences, embed nodes
        3. LIBRARIAN: create edges, build graph, detect communities

        Returns
        -------
        LibrarianOutput
            Graph, community structure, and similarity index.
        """
        pipeline_t0 = _time.monotonic()
        doc_count = len(documents)
        logger.info("===== index_corpus_v5 START — %d documents =====", doc_count)

        # --- Stage 1: CHUNKER ---
        logger.info("CHUNKER START — %d documents", doc_count)
        t0 = _time.monotonic()
        try:
            chunker_input = ChunkerInput(
                documents=documents,
                config=self._config.get("chunker", self._config.get("explorer", {})),
            )
            chunker = ChunkerController(
                self._config.get("chunker", self._config.get("explorer", {})),
            )
            chunker_output = chunker.process(chunker_input)
        except Exception:
            logger.exception("CHUNKER FAILED")
            raise
        elapsed = _time.monotonic() - t0
        self._last_chunker_output = chunker_output
        self._stage_timings["chunker"] = elapsed * 1000
        self._stage_token_usage["chunker"] = self._llm.snapshot_and_reset()
        self._stage_token_usage["chunker_embed"] = self._embedding.snapshot_and_reset()

        chunk_count = len(chunker_output.chunks)
        src_doc_count = len(chunker_output.source_doc_ids)
        logger.info(
            "CHUNKER DONE — chunks=%d source_doc_ids=%d elapsed=%.2fs",
            chunk_count, src_doc_count, elapsed,
        )
        if chunker_output.trace:
            logger.debug("CHUNKER trace: %s", _trace_summary(chunker_output.trace))

        # --- Stage 2: EXTRACTOR ---
        logger.info("EXTRACTOR START — %d chunks", chunk_count)
        t0 = _time.monotonic()
        try:
            extractor_input = ExtractorInput(
                chunks=chunker_output.chunks,
                source_doc_ids=chunker_output.source_doc_ids,
                config=self._config.get("extractor", self._config.get("explorer", {})),
            )
            extractor = ExtractorController(
                self._llm, self._embedding,
                self._config.get("extractor", self._config.get("explorer", {})),
            )
            explorer_output = extractor.process(extractor_input)
        except Exception:
            logger.exception("EXTRACTOR FAILED")
            raise
        elapsed = _time.monotonic() - t0
        self._last_explorer_output = explorer_output
        self._stage_timings["extractor"] = elapsed * 1000
        self._stage_token_usage["extractor"] = self._llm.snapshot_and_reset()
        self._stage_token_usage["extractor_embed"] = self._embedding.snapshot_and_reset()

        entity_count = explorer_output.node_set.entity_count if explorer_output.node_set else 0
        claim_count = explorer_output.node_set.claim_count if explorer_output.node_set else 0
        node_count = len(explorer_output.node_set.nodes) if explorer_output.node_set else 0
        llm_calls = explorer_output.trace.llm_calls if explorer_output.trace else 0
        logger.info(
            "EXTRACTOR DONE — entities=%d claims=%d nodes=%d llm_calls=%d elapsed=%.2fs",
            entity_count, claim_count, node_count, llm_calls, elapsed,
        )
        if explorer_output.trace:
            logger.debug("EXTRACTOR trace: %s", _trace_summary(explorer_output.trace))

        # --- Stage 3: LIBRARIAN ---
        logger.info("LIBRARIAN START — %d nodes", node_count)
        t0 = _time.monotonic()
        try:
            doc_store = {doc.doc_id: doc for doc in documents}
            librarian_input = LibrarianInput(
                node_set=explorer_output.node_set,
                chunk_node_map=explorer_output.chunk_node_map,
                document_store=doc_store,
            )
            librarian = LibrarianControllerV5(
                self._llm, self._config.get("librarian", {}),
            )
            librarian_output = librarian.process(librarian_input)
        except Exception:
            logger.exception("LIBRARIAN FAILED")
            raise
        elapsed = _time.monotonic() - t0
        self._stage_timings["librarian"] = elapsed * 1000
        self._stage_token_usage["librarian"] = self._llm.snapshot_and_reset()
        self._stage_token_usage["librarian_embed"] = self._embedding.snapshot_and_reset()

        graph_nodes = len(librarian_output.graph.nodes)
        graph_edges = len(librarian_output.graph.edges)
        comm_count = (
            len(librarian_output.community_structure.communities)
            if librarian_output.community_structure else 0
        )
        logger.info(
            "LIBRARIAN DONE — graph_nodes=%d graph_edges=%d communities=%d elapsed=%.2fs",
            graph_nodes, graph_edges, comm_count, elapsed,
        )

        self._index = librarian_output

        total_elapsed = _time.monotonic() - pipeline_t0
        self._stage_timings["index_total"] = total_elapsed * 1000
        logger.info(
            "===== index_corpus_v5 DONE — total elapsed=%.2fs =====",
            total_elapsed,
        )
        return librarian_output

    def process_query_v5(
        self,
        query: QueryRequest,
        index: LibrarianOutput,
    ) -> ReasonerOutput:
        """Process a query through the v5 pipeline.

        Steps:
        1. PLANNER: embed query, find seeds, PPR expand, route, assemble context
        2. REASONER: build prompt, infer relationships, ground evidence, verdict

        Parameters
        ----------
        query:
            The user's query.
        index:
            The indexed corpus from ``index_corpus_v5``.

        Returns
        -------
        ReasonerOutput
            Finding with verdict, evidence, and trace.
        """
        pipeline_t0 = _time.monotonic()
        graph_nodes = len(index.graph.nodes)
        graph_edges = len(index.graph.edges)
        logger.info(
            "===== process_query_v5 START — query_id=%s graph=%d nodes/%d edges =====",
            query.query_id, graph_nodes, graph_edges,
        )

        # --- Build SimilarityIndex from graph nodes ---
        sim_index = SimilarityIndex()
        sim_index.add_nodes(index.graph.nodes)

        # --- Stage 1: PLANNER ---
        logger.info("PLANNER START — query_id=%s", query.query_id)
        t0 = _time.monotonic()
        try:
            planner_input = PlannerInput(
                query=query,
                graph=index.graph,
                similarity_index=sim_index,
                community_structure=index.community_structure,
                document_store=index.document_store,
                chunks=self._last_chunker_output.chunks if self._last_chunker_output else None,
            )
            planner_config = {
                **self._config.get("planner", {}),
                "embedding_client": self._embedding,
            }
            planner = PlannerControllerV5(planner_config)
            planner_output = planner.process(planner_input)
        except Exception:
            logger.exception("PLANNER FAILED")
            raise
        elapsed = _time.monotonic() - t0
        self._last_planner_output = planner_output
        self._stage_timings["planner"] = elapsed * 1000
        self._stage_token_usage["planner"] = self._llm.snapshot_and_reset()
        self._stage_token_usage["planner_embed"] = self._embedding.snapshot_and_reset()

        rd = planner_output.router_decision
        seed_count = len(rd.seed_nodes)
        logger.info(
            "PLANNER DONE — lane=%s seeds=%d intent=%s elapsed=%.2fs",
            rd.selected_lane.value, seed_count,
            rd.intent.classification.value, elapsed,
        )

        # --- Stage 2: REASONER ---
        logger.info(
            "REASONER START — lane=%s query_id=%s",
            rd.selected_lane.value, query.query_id,
        )
        t0 = _time.monotonic()
        try:
            reasoner_input = ReasonerInput(
                query=query,
                query_id=query.query_id,
                router_decision=planner_output.router_decision,
                focused_context=planner_output.focused_context,
                global_context=planner_output.global_context,
                targeted_context=planner_output.targeted_context,
            )
            reasoner = ReasonerControllerV5(self._llm, self._config.get("reasoner", {}))
            reasoner_output = reasoner.process(reasoner_input)
        except Exception:
            logger.exception("REASONER FAILED")
            raise
        elapsed = _time.monotonic() - t0
        self._stage_timings["reasoner"] = elapsed * 1000
        self._stage_token_usage["reasoner"] = self._llm.snapshot_and_reset()
        self._stage_token_usage["reasoner_embed"] = self._embedding.snapshot_and_reset()

        finding = reasoner_output.finding
        logger.info(
            "REASONER DONE — verdict=%s confidence=%.2f relationships=%d evidence=%d elapsed=%.2fs",
            finding.verdict.verdict_type.value,
            finding.verdict.confidence,
            len(finding.relationships),
            len(finding.evidence),
            elapsed,
        )

        total_elapsed = _time.monotonic() - pipeline_t0
        self._stage_timings["query_total"] = total_elapsed * 1000
        logger.info(
            "===== process_query_v5 DONE — total elapsed=%.2fs =====",
            total_elapsed,
        )
        return reasoner_output

    # ------------------------------------------------------------------
    # Stashed outputs for evaluation
    # ------------------------------------------------------------------

    @property
    def last_chunker_output(self) -> Optional[ChunkerOutput]:
        """The most recent ``ChunkerOutput`` from ``index_corpus_v5``."""
        return self._last_chunker_output

    @property
    def last_explorer_output(self) -> Optional[ExplorerOutput]:
        """The most recent ``ExplorerOutput`` from ``index_corpus_v5``."""
        return self._last_explorer_output

    @property
    def last_planner_output(self) -> Optional[PlannerOutput]:
        """The most recent ``PlannerOutput`` from ``process_query_v5``."""
        return self._last_planner_output

    @property
    def stage_timings(self) -> Dict[str, float]:
        """Per-stage elapsed time in milliseconds."""
        return dict(self._stage_timings)

    @property
    def stage_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Per-stage accumulated token usage."""
        return dict(self._stage_token_usage)

    def run_and_snapshot(
        self,
        documents: List[RawDocument],
        query: QueryRequest,
    ):
        """Run the full v5 pipeline and return a V5PipelineSnapshot.

        Convenience method for evaluation. Indexes the documents, then
        processes the query, capturing all intermediate outputs.

        Parameters
        ----------
        documents:
            Input documents.
        query:
            The user query.

        Returns
        -------
        V5PipelineSnapshot
            Snapshot of all intermediate outputs.
        ReasonerOutput
            The final pipeline result.
        """
        from evaluation.v5_metrics import V5PipelineSnapshot

        index = self.index_corpus_v5(documents)
        reasoner_output = self.process_query_v5(query, index)

        snapshot = V5PipelineSnapshot(
            documents=documents,
            query=query,
            chunker_output=self._last_chunker_output,
            explorer_output=self._last_explorer_output,
            librarian_output=index,
            planner_output=self._last_planner_output,
            reasoner_output=reasoner_output,
        )

        return snapshot, reasoner_output


# ============================================================================
# MOCK LLM CLIENT
# ============================================================================

class MockLLMClient:
    """Mock LLM that routes based on prompt content to return canned JSON.

    Handles four prompt types:
    - Initial extraction (per-document)
    - Gleaning (self-reflection)
    - Edge interpretation (scope resolution)
    - Decontextualization (definition substitution)
    """

    def generate(self, prompt: str) -> str:
        if "You are a legal-document analyst" in prompt:
            return self._handle_summarizer(prompt)
        if "You are a legal-domain taxonomy expert" in prompt:
            return self._handle_domain_reasoner(prompt)
        if "Extract entities and relationships" in prompt:
            return self._handle_extraction(prompt)
        if "Review your extraction for missed" in prompt:
            return self._handle_gleaning(prompt)
        if "scope resolver determining which documents" in prompt:
            return self._handle_edge_interpretation(prompt)
        if "legal text processor performing definition substitution" in prompt:
            return self._handle_decontextualization(prompt)
        return "{}"

    # -- Summarizer (Explorer) ------------------------------------------------

    @staticmethod
    def _handle_summarizer(prompt: str) -> str:
        """Return canned DocumentSummary JSON for known doc IDs."""
        summaries = {
            "MSA": {
                "summary": (
                    "Master Service Agreement between a vendor and client. "
                    "Defines key terms, limitation of liability, and "
                    "indemnification obligations."
                ),
                "key_themes": [
                    "liability limitation",
                    "service provision",
                    "indemnification",
                    "contractual obligations",
                ],
                "apparent_domain": "corporate law",
                "key_terms": ["Vendor", "Client", "Liability Cap"],
                "document_type": "contract",
            },
            "DORA": {
                "summary": (
                    "Digital Operational Resilience Act establishing ICT risk "
                    "management requirements for financial entities and their "
                    "critical third-party service providers."
                ),
                "key_themes": [
                    "operational resilience",
                    "ICT risk management",
                    "financial regulation",
                    "third-party oversight",
                ],
                "apparent_domain": "financial regulation",
                "key_terms": [
                    "ICT risk",
                    "operational resilience",
                    "financial entities",
                ],
                "document_type": "regulation",
            },
            "CLINICAL_PROTOCOL": {
                "summary": (
                    "Clinical trial protocol for a randomized double-blind "
                    "placebo-controlled study with electronic data capture "
                    "requirements."
                ),
                "key_themes": [
                    "clinical trials",
                    "data management",
                    "study design",
                    "regulatory compliance",
                ],
                "apparent_domain": "clinical research",
                "key_terms": [
                    "randomized",
                    "double-blind",
                    "electronic data capture",
                ],
                "document_type": "protocol",
            },
        }

        # Find which doc_id appears in the prompt
        for doc_id, data in summaries.items():
            if f"DOCUMENT ID: {doc_id}" in prompt:
                return json.dumps(data)

        # Fallback for unknown documents
        return json.dumps({
            "summary": "Document summary not available.",
            "key_themes": ["general"],
            "apparent_domain": "unknown",
            "key_terms": [],
            "document_type": "other",
        })

    # -- Domain Reasoner (Explorer) -------------------------------------------

    @staticmethod
    def _handle_domain_reasoner(prompt: str) -> str:
        """Return canned DomainTaxonomy JSON."""
        return json.dumps({
            "domains": [
                {
                    "domain_id": "dom_corporate_law",
                    "name": "Corporate/Contract Law",
                    "description": (
                        "Governs contractual agreements, liability, and "
                        "service-level obligations between parties."
                    ),
                    "indicators": [
                        "Master Service Agreement",
                        "liability",
                        "indemnification",
                        "Vendor",
                    ],
                },
                {
                    "domain_id": "dom_financial_reg",
                    "name": "Financial Regulation",
                    "description": (
                        "Regulatory framework for financial entities "
                        "covering ICT risk and operational resilience."
                    ),
                    "indicators": [
                        "DORA",
                        "operational resilience",
                        "ICT risk",
                        "financial entities",
                    ],
                },
            ],
            "cross_domain_relationships": [
                {
                    "source_domain": "dom_financial_reg",
                    "target_domain": "dom_corporate_law",
                    "relationship_type": "regulates",
                    "reasoning": (
                        "Financial regulation imposes mandatory requirements "
                        "on service agreements with financial entities."
                    ),
                },
            ],
        })

    # -- Extraction -----------------------------------------------------------

    def _handle_extraction(self, prompt: str) -> str:
        if "DOCUMENT ID: MSA" in prompt:
            return json.dumps({
                "document_node": {
                    "node_id": "doc_msa",
                    "domain": "LEGAL",
                    "content_summary": "Master Service Agreement between parties",
                    "defined_terms": ["Vendor"],
                },
                "entities": [],
                "definitions": [
                    {
                        "node_id": "def_vendor",
                        "term": "Vendor",
                        "definition_text": "Acme Corp",
                    }
                ],
                "clauses": [
                    {
                        "node_id": "cls_liability",
                        "clause_type": "Limitation of Liability",
                        "content_summary": (
                            "Vendor's total liability shall not exceed $1,000,000"
                        ),
                    }
                ],
                "edges": [],
            })

        if "DOCUMENT ID: DORA" in prompt:
            return json.dumps({
                "document_node": {
                    "node_id": "doc_dora",
                    "domain": "REGULATORY",
                    "content_summary": "Digital Operational Resilience Act",
                    "defined_terms": [],
                },
                "entities": [],
                "definitions": [],
                "clauses": [
                    {
                        "node_id": "cls_resilience",
                        "clause_type": "Resilience Requirements",
                        "content_summary": (
                            "Service providers must maintain operational "
                            "resilience standards"
                        ),
                    }
                ],
                "edges": [
                    {
                        "edge_id": "edge_governs_1",
                        "source_node": "doc_dora",
                        "target_node": "doc_msa",
                        "edge_type": "GOVERNS",
                        "confidence": 0.95,
                        "reasoning_hint": (
                            "DORA regulatory requirements govern the MSA"
                        ),
                    }
                ],
            })

        return json.dumps({
            "document_node": {
                "node_id": "doc_unknown",
                "domain": "LEGAL",
                "content_summary": "Unknown document",
                "defined_terms": [],
            },
            "entities": [],
            "definitions": [],
            "clauses": [],
            "edges": [],
        })

    # -- Gleaning -------------------------------------------------------------

    @staticmethod
    def _handle_gleaning(prompt: str) -> str:
        return json.dumps({
            "additional_entities": [],
            "additional_edges": [],
            "corrections": [],
            "confidence_complete": 1.0,
        })

    # -- Edge interpretation --------------------------------------------------

    @staticmethod
    def _handle_edge_interpretation(prompt: str) -> str:
        return json.dumps({
            "relevant_edges": [
                {
                    "edge_id": "edge_governs_1",
                    "relevance_score": 0.95,
                    "reasoning_hint": (
                        "DORA regulatory requirements govern the MSA"
                    ),
                    "precedence_rule": (
                        "DORA requirements are mandatory minimums; "
                        "MSA can be stricter but not more permissive."
                    ),
                }
            ],
            "supporting_documents": [
                {
                    "doc_id": "DORA",
                    "dependency_type": "REGULATION",
                    "required_sections": None,
                }
            ],
            "domain_pair": {
                "primary_domain": "REGULATORY",
                "secondary_domain": "LEGAL",
                "cross_domain_logic": (
                    "Regulatory floor supersedes contract"
                ),
            },
            "completeness_confidence": 0.9,
            "warnings": [],
        })

    # -- Decontextualization --------------------------------------------------

    @staticmethod
    def _handle_decontextualization(prompt: str) -> str:
        if "MASTER SERVICE AGREEMENT" in prompt:
            return json.dumps({
                "decontextualized_content": (
                    'MASTER SERVICE AGREEMENT\n\n'
                    'Section 1. DEFINITIONS\n'
                    '"Vendor" means Acme Corp.\n\n'
                    'Section 2. LIMITATION OF LIABILITY\n'
                    'The Vendor [defined as: Acme Corp] shall not be liable '
                    'for any indirect, incidental, or consequential damages. '
                    'The Vendor\'s total aggregate liability shall not exceed '
                    '$1,000,000.'
                ),
                "substitutions_made": [
                    {
                        "term": "Vendor",
                        "definition": "Acme Corp",
                        "location": "Section 2",
                    }
                ],
                "failures": [],
                "final_token_count": 60,
            })

        # DORA or any other document: no Vendor term in content
        return json.dumps({
            "decontextualized_content": prompt.split("ORIGINAL TEXT:\n")[1].split(
                "\n\nDEFINITIONS AVAILABLE:"
            )[0] if "ORIGINAL TEXT:" in prompt else "",
            "substitutions_made": [],
            "failures": [
                {"term": "Vendor", "reason": "NOT_FOUND"},
            ],
            "final_token_count": 50,
        })


# ============================================================================
# v5 MOCK CLIENTS
# ============================================================================

class MockLLMClientV5:
    """Mock LLM client for v5 pipeline testing.

    Routes prompts by substring to return canned JSON responses for:
    - Entity extraction (EXPLORER)
    - Claim extraction (EXPLORER)
    - Gleaning (EXPLORER)
    - Coreference resolution (EXPLORER)
    - Community summarization (LIBRARIAN)
    - Epistemic analysis (REASONER — all 3 lanes)
    """

    def __init__(self):
        self.last_usage = None

    def generate(self, prompt: str) -> str:
        # --- v5 EXPLORER prompts ---
        if "Extract all named entities" in prompt:
            result = self._handle_entity_extraction(prompt)
        elif "Extract factual claims" in prompt:
            result = self._handle_claim_extraction(prompt)
        elif "Examine the text again carefully" in prompt:
            result = self._handle_gleaning(prompt)
        elif "Identify which entities" in prompt:
            result = self._handle_coreference(prompt)
        # --- v5 LIBRARIAN prompts ---
        elif "Summarize the following group" in prompt:
            result = self._handle_community_summary(prompt)
        # --- v5 REASONER prompts ---
        elif "epistemic analysis engine" in prompt:
            result = self._handle_reasoner_focused(prompt)
        elif "epistemic synthesis engine" in prompt:
            result = self._handle_reasoner_global(prompt)
        elif "evidence lookup engine" in prompt:
            result = self._handle_reasoner_targeted(prompt)
        else:
            result = "{}"

        prompt_tokens = len(prompt) // 4
        completion_tokens = len(result) // 4
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return result

    @staticmethod
    def _handle_entity_extraction(prompt: str) -> str:
        if "MASTER SERVICE AGREEMENT" in prompt or "Vendor" in prompt:
            return json.dumps([
                {"surface_form": "Vendor", "entity_type": "ACTOR"},
                {"surface_form": "Acme Corp", "entity_type": "ORGANIZATION"},
                {"surface_form": "$1,000,000", "entity_type": "QUANTITY"},
            ])
        if "DIGITAL OPERATIONAL RESILIENCE" in prompt or "DORA" in prompt:
            return json.dumps([
                {"surface_form": "DORA", "entity_type": "ARTIFACT"},
                {"surface_form": "financial entities", "entity_type": "ACTOR"},
                {"surface_form": "ICT risk", "entity_type": "CONCEPT"},
            ])
        return json.dumps([
            {"surface_form": "Unknown Entity", "entity_type": "UNKNOWN"},
        ])

    @staticmethod
    def _handle_claim_extraction(prompt: str) -> str:
        if "MASTER SERVICE AGREEMENT" in prompt or "Vendor" in prompt:
            return json.dumps([{
                "claim": "Vendor's total aggregate liability shall not exceed $1,000,000",
                "subject": "Vendor",
                "object": "$1,000,000",
                "predicate": "liability_capped_at",
                "description": "Liability cap for vendor",
                "confidence": 0.95,
            }])
        if "DIGITAL OPERATIONAL RESILIENCE" in prompt or "DORA" in prompt:
            return json.dumps([{
                "claim": "Financial entities shall ensure operational resilience at all times",
                "subject": "financial entities",
                "object": "operational resilience",
                "predicate": "must_ensure",
                "description": "Operational resilience requirement",
                "confidence": 0.9,
            }])
        return json.dumps([])

    @staticmethod
    def _handle_gleaning(prompt: str) -> str:
        return "EXTRACTION COMPLETE"

    @staticmethod
    def _handle_coreference(prompt: str) -> str:
        return json.dumps([
            {"canonical": "Vendor", "variants": ["Vendor", "Acme Corp"]},
            {"canonical": "$1,000,000", "variants": ["$1,000,000"]},
            {"canonical": "DORA", "variants": ["DORA"]},
            {"canonical": "financial entities", "variants": ["financial entities"]},
            {"canonical": "ICT risk", "variants": ["ICT risk"]},
        ])

    @staticmethod
    def _handle_community_summary(prompt: str) -> str:
        return json.dumps({
            "summary": (
                "Legal and regulatory concepts related to operational "
                "obligations and liability provisions."
            ),
            "key_themes": [
                "liability",
                "operational resilience",
                "regulatory compliance",
            ],
        })

    @staticmethod
    def _handle_reasoner_focused(prompt: str) -> str:
        return json.dumps({
            "relationships": [{
                "source_id": "MSA",
                "target_id": "DORA",
                "relationship": "CONTRADICTS",
                "confidence": 0.85,
                "reasoning": (
                    "MSA caps liability while DORA requires "
                    "operational resilience"
                ),
                "evidence_quote": (
                    "Vendor's total aggregate liability shall "
                    "not exceed $1,000,000"
                ),
            }],
            "evidence": [{
                "claim": "MSA limits vendor liability to $1,000,000",
                "source_document": "MSA",
                "exact_quote": (
                    "Vendor's total aggregate liability shall "
                    "not exceed $1,000,000"
                ),
            }],
            "verdict": {
                "verdict_type": "CONFIRMED",
                "answer": "Yes, the MSA contradicts DORA requirements",
                "confidence": 0.85,
                "summary": (
                    "The MSA's liability cap contradicts DORA's "
                    "operational resilience requirements."
                ),
            },
            "reasoning_trace": (
                "Compared MSA Section 2 liability cap with "
                "DORA Article 2 requirements."
            ),
        })

    @staticmethod
    def _handle_reasoner_global(prompt: str) -> str:
        return json.dumps({
            "relationships": [{
                "source_id": "community_0",
                "target_id": "community_1",
                "relationship": "CONTRADICTS",
                "confidence": 0.75,
                "reasoning": (
                    "Contract provisions tension with "
                    "regulatory requirements"
                ),
                "evidence_quote": "",
            }],
            "evidence": [],
            "verdict": {
                "verdict_type": "PARTIALLY_CONFIRMED",
                "answer": (
                    "Tension exists between contract and "
                    "regulatory communities"
                ),
                "confidence": 0.75,
                "summary": (
                    "Community-level analysis reveals tension between "
                    "contractual and regulatory provisions."
                ),
            },
            "reasoning_trace": "Analyzed inter-community relationships.",
        })

    @staticmethod
    def _handle_reasoner_targeted(prompt: str) -> str:
        return json.dumps({
            "relationships": [{
                "source_id": "chunk_MSA_0",
                "target_id": "chunk_DORA_0",
                "relationship": "CONTRADICTS",
                "confidence": 0.7,
                "reasoning": "Liability provisions conflict",
                "evidence_quote": (
                    "liability shall not exceed $1,000,000"
                ),
            }],
            "evidence": [{
                "claim": "Liability is capped at $1M",
                "source_document": "MSA",
                "exact_quote": (
                    "liability shall not exceed $1,000,000"
                ),
            }],
            "verdict": {
                "verdict_type": "PARTIALLY_CONFIRMED",
                "answer": "Evidence suggests contradiction",
                "confidence": 0.7,
                "summary": (
                    "Chunk-level analysis identifies "
                    "liability cap contradiction."
                ),
            },
            "reasoning_trace": (
                "Searched relevant chunks for liability provisions."
            ),
        })


class MockEmbeddingClientV5:
    """Mock embedding client for v5 pipeline testing.

    Returns deterministic 8-dimensional embeddings based on text hashing.
    All vectors have high mutual cosine similarity to ensure seed discovery.
    """

    def __init__(self):
        self.last_usage = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib

        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            # Base direction [0.5, ...] with small hash-based perturbation
            vec = [
                0.5 + int(h[i * 2 : i * 2 + 2], 16) / 2550.0
                for i in range(8)
            ]
            results.append(vec)

        total_prompt_tokens = sum(len(t) // 4 for t in texts)
        self.last_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_prompt_tokens,
        }
        return results


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_orchestrator() -> None:
    """End-to-end orchestrator test with mock LLM."""

    print("=" * 70)
    print("Orchestrator End-to-End Test")
    print("=" * 70)

    # 1. Create MockLLMClient
    mock_llm = MockLLMClient()

    # 2. Create config
    config: Dict[str, Any] = {
        "corpus_metadata": {"has_graph": True, "domain": "LEGAL"},
        "available_documents": [],
        "known_entities": [],
        "indexer": {
            "extraction": {
                "max_gleaning_iterations": 1,
                "min_confidence_threshold": 0.9,
            },
        },
        "resolver": {},
        "assembler": {
            "decontextualization": {},
            "context": {},
        },
    }

    # 3. Create Orchestrator
    orchestrator = Orchestrator(mock_llm, config)

    # 4. Build RawCorpus with 2 documents
    corpus = RawCorpus(
        corpus_id="test-corpus-1",
        documents=[
            RawDocument(
                doc_id="MSA",
                content=(
                    'MASTER SERVICE AGREEMENT\n\n'
                    'Section 1. DEFINITIONS\n'
                    '"Vendor" means Acme Corp.\n\n'
                    'Section 2. LIMITATION OF LIABILITY\n'
                    'The Vendor shall not be liable for any indirect, '
                    'incidental, or consequential damages. '
                    "The Vendor's total aggregate liability shall not "
                    'exceed $1,000,000.'
                ),
                metadata={"type": "contract", "domain": "LEGAL"},
            ),
            RawDocument(
                doc_id="DORA",
                content=(
                    'DIGITAL OPERATIONAL RESILIENCE ACT (DORA)\n\n'
                    'Article 1. SCOPE\n'
                    'This regulation establishes requirements for ICT risk '
                    'management applicable to financial entities and their '
                    'critical third-party service providers.\n\n'
                    'Article 2. OPERATIONAL RESILIENCE REQUIREMENTS\n'
                    'Financial entities shall ensure that their ICT systems '
                    'and service providers maintain adequate operational '
                    'resilience at all times.'
                ),
                metadata={"type": "regulation", "domain": "REGULATORY"},
            ),
        ],
        ingestion_timestamp=datetime.now(),
    )

    # 5. Index corpus
    print("\n--- Indexing Corpus ---")
    indexed = orchestrator.index_corpus(corpus)
    node_count = len(indexed.graph_state.nodes)
    edge_count = len(indexed.graph_state.edges)
    print(f"Index complete: {node_count} nodes, {edge_count} edges")
    print(f"Coverage score: {indexed.extraction_metadata.coverage_score:.2f}")

    # 6. Build query
    query = QueryRequest(
        query_id="test-q1",
        query_text="Does the MSA contradict DORA?",
    )

    # 7. Process query
    print("\n--- Processing Query ---")
    print(f'Query: "{query.query_text}"')
    router_decision, assembled_context = orchestrator.process_query(query)

    # 8. Assert router decision
    from schemas.contracts import LaneType
    assert router_decision.selected_lane == LaneType.LANE_A_GRAPH_LCI, (
        f"Expected LANE_A_GRAPH_LCI, got {router_decision.selected_lane.value}"
    )
    print(f"\nRouterDecision:")
    print(f"  Lane:            {router_decision.selected_lane.value}")
    print(f"  Graph scope:     {router_decision.requires_graph_scope}")
    print(f"  Intent:          {router_decision.intent.classification.value} "
          f"(conf={router_decision.intent.confidence})")
    print(f"  Targets:         docs={router_decision.identified_targets.explicit_docs}, "
          f"entities={router_decision.identified_targets.implicit_entities}")
    print(f"  Domain:          {router_decision.domain_expectation.value}")

    # 9. Assert assembled context exists
    assert assembled_context is not None, "Expected assembled context for graph query"

    # 10. Print edge grammar
    print(f"\n--- Edge Grammar ---")
    print(assembled_context.lci_payload.preamble.edge_grammar)

    # 11. Print document ordering and token count
    print(f"\n--- Document Ordering ---")
    for doc in assembled_context.lci_payload.documents:
        print(f"  [{doc.injection_order}] {doc.doc_id} (role={doc.role})")
    print(f"  Total tokens: {assembled_context.lci_payload.total_tokens}")

    # 12. Print decontextualization report
    report = assembled_context.decontextualization_report
    print(f"\n--- Decontextualization Report ---")
    print(f"  Terms resolved:  {report.terms_resolved}")
    print(f"  Terms failed:    {len(report.terms_failed)}")
    for f in report.terms_failed:
        print(f"    - {f.term}: {f.failure_reason.value}")
    print(f"  Success rate:    {report.substitution_success_rate:.2f}")

    # Final assertions
    assert router_decision.requires_graph_scope is True
    assert "GOVERNS" in assembled_context.lci_payload.preamble.edge_grammar

    print("\n" + "=" * 70)
    print("All assertions passed.")
    print("=" * 70)


def test_with_explorer() -> None:
    """Test the EXPLORER → LIBRARIAN pipeline.

    Uses a real LLM (via LLM_PROVIDER env var or get_client) when
    available, otherwise falls back to MockLLMClient.  The explorer
    controller is called for real — summaries and taxonomy come from
    the LLM.
    """
    print("=" * 70)
    print("Explorer → Librarian Pipeline Test")
    print("=" * 70)

    # 1. Choose LLM client
    llm_provider = os.getenv("LLM_PROVIDER", "")
    if llm_provider:
        llm = get_client(llm_provider)
        print(f"Using real LLM: {llm_provider} ({type(llm).__name__})")
    else:
        try:
            llm = get_client("ollama")
            print(f"Using real LLM: ollama ({type(llm).__name__})")
        except Exception:
            llm = MockLLMClient()
            print("Using MockLLMClient (no LLM_PROVIDER set, Ollama unavailable)")

    # 2. Create config
    config: Dict[str, Any] = {
        "corpus_metadata": {"has_graph": True, "domain": "CROSS_DOMAIN"},
        "available_documents": [],
        "known_entities": [],
        "indexer": {
            "extraction": {
                "max_gleaning_iterations": 1,
                "min_confidence_threshold": 0.9,
            },
        },
        "resolver": {},
        "assembler": {
            "decontextualization": {},
            "context": {},
        },
    }

    # 3. Build a 3-document cross-domain corpus
    corpus = RawCorpus(
        corpus_id="explorer-test-corpus",
        documents=[
            RawDocument(
                doc_id="MSA",
                content=(
                    'MASTER SERVICE AGREEMENT\n\n'
                    'Section 1. DEFINITIONS\n'
                    '"Vendor" means Acme Corp.\n\n'
                    'Section 2. LIMITATION OF LIABILITY\n'
                    'The Vendor shall not be liable for any indirect, '
                    'incidental, or consequential damages. '
                    "The Vendor's total aggregate liability shall not "
                    'exceed $1,000,000.'
                ),
                metadata={"type": "contract", "domain": "LEGAL"},
            ),
            RawDocument(
                doc_id="DORA",
                content=(
                    'DIGITAL OPERATIONAL RESILIENCE ACT (DORA)\n\n'
                    'Article 1. SCOPE\n'
                    'This regulation establishes requirements for ICT risk '
                    'management applicable to financial entities.\n\n'
                    'Article 2. OPERATIONAL RESILIENCE REQUIREMENTS\n'
                    'Financial entities shall ensure operational resilience.'
                ),
                metadata={"type": "regulation", "domain": "REGULATORY"},
            ),
            RawDocument(
                doc_id="CLINICAL_PROTOCOL",
                content=(
                    'CLINICAL TRIAL PROTOCOL CP-2024-001\n\n'
                    'Section 1. STUDY DESIGN\n'
                    'Randomized double-blind placebo-controlled trial.\n\n'
                    'Section 2. DATA MANAGEMENT\n'
                    'Electronic data capture systems must comply with '
                    'applicable ICT regulations.'
                ),
                metadata={"type": "protocol", "domain": "SCIENTIFIC"},
            ),
        ],
        ingestion_timestamp=datetime.now(),
    )

    # 4. Run exploration via ExplorerController (real LLM path)
    explorer = ExplorerController(llm, config.get("explorer", {}))
    exploration_request = ExplorationRequest(corpus=corpus)

    print("\n--- Running Exploration ---")
    exploration_response = explorer.process(exploration_request)

    if not exploration_response.success:
        print(f"  Exploration FAILED: {exploration_response.errors}")
        print("  Falling back without exploration data.")
        exploration_result = None
    else:
        exploration_result = exploration_response.result
        print(f"  Exploration succeeded.")

        # Print summaries
        print(f"\n--- Document Summaries ({len(exploration_result.summaries)}) ---")
        for s in exploration_result.summaries:
            print(f"\n  [{s.doc_id}]")
            print(f"    Summary:  {s.summary}")
            print(f"    Domain:   {s.apparent_domain}")
            print(f"    Type:     {s.document_type}")
            print(f"    Themes:   {', '.join(s.key_themes)}")
            print(f"    Terms:    {', '.join(s.key_terms)}")

        # Print taxonomy
        taxonomy = exploration_result.taxonomy
        print(f"\n--- Domain Taxonomy ({len(taxonomy.domains)} domains) ---")
        for d in taxonomy.domains:
            print(f"\n  [{d.domain_id}] {d.name}")
            print(f"    {d.description}")
            print(f"    Indicators: {', '.join(d.indicators)}")

        print(f"\n--- Cross-Domain Relationships ({len(taxonomy.cross_domain_relationships)}) ---")
        for r in taxonomy.cross_domain_relationships:
            print(f"  {r.source_domain} --{r.relationship_type}--> {r.target_domain}")
            print(f"    {r.reasoning}")

        print(f"\n  Metadata: {exploration_result.exploration_metadata}")

    # 5. Index with exploration context
    index_request = IndexRequest(corpus=corpus, exploration=exploration_result)
    librarian = LibrarianController(llm, config)
    request = LibrarianRequest(request_type="INDEX", payload=index_request)

    print("\n--- Indexing with IndexRequest ---")
    response = librarian.process(request)

    assert response.success, f"Indexing failed: {response.errors}"
    indexed = response.payload
    node_count = len(indexed.graph_state.nodes)
    edge_count = len(indexed.graph_state.edges)
    print(f"Index complete: {node_count} nodes, {edge_count} edges")
    print(f"Coverage score: {indexed.extraction_metadata.coverage_score:.2f}")

    if exploration_result is not None:
        assert index_request.exploration is exploration_result
        print(f"\nExploration attached: corpus_id={index_request.exploration.corpus_id}")

    print("\n" + "=" * 70)
    print("Explorer pipeline test complete. Review output above.")
    print("=" * 70)


def test_with_real_llm() -> None:
    """End-to-end orchestrator test with a real Ollama LLM."""

    print("=" * 70)
    print("Orchestrator Real-LLM Test (Ollama)")
    print("=" * 70)

    # 1. Create real LLM client
    llm = get_client("ollama")
    assert isinstance(llm, OllamaClient)
    print(f"Using model: {llm.model}")

    # 2. Create config
    config: Dict[str, Any] = {
        "corpus_metadata": {"has_graph": True, "domain": "LEGAL"},
        "available_documents": [],
        "known_entities": [],
        "indexer": {
            "extraction": {
                "max_gleaning_iterations": 1,
                "min_confidence_threshold": 0.9,
            },
        },
        "resolver": {},
        "assembler": {
            "decontextualization": {},
            "context": {},
        },
    }

    # 3. Create Orchestrator
    orchestrator = Orchestrator(llm, config)

    # 4. Build RawCorpus with 2 documents designed to produce a clear contradiction
    corpus = RawCorpus(
        corpus_id="real-llm-test-corpus",
        documents=[
            RawDocument(
                doc_id="MSA",
                content=(
                    'MASTER SERVICE AGREEMENT\n\n'
                    'Section 1. DEFINITIONS\n'
                    '"Service Provider" means TechCo Ltd, a technology '
                    'services company.\n'
                    '"Client" means FinanceBank plc.\n\n'
                    'Section 2. LIMITATION OF LIABILITY\n'
                    "The Service Provider's total aggregate liability under "
                    'this Agreement shall not exceed $1,000,000 (one million '
                    'dollars) regardless of the nature of the claim.\n\n'
                    'Section 3. INDEMNIFICATION\n'
                    'The Service Provider shall indemnify the Client against '
                    "losses arising from the Service Provider's negligence, "
                    'subject to the liability cap in Section 2.'
                ),
                metadata={"type": "contract", "domain": "LEGAL"},
            ),
            RawDocument(
                doc_id="REGULATION",
                content=(
                    'CRITICAL SERVICE PROVIDER REGULATION (CSPR)\n\n'
                    'Article 1. SCOPE\n'
                    'This regulation applies to all critical third-party '
                    'service providers serving regulated financial entities.\n\n'
                    'Article 2. LIABILITY REQUIREMENTS\n'
                    'Critical service providers shall maintain unlimited '
                    'liability for operational failures that impact the '
                    'financial stability of their clients. No contractual '
                    'limitation of liability shall override this '
                    'requirement.\n\n'
                    'Article 3. COMPLIANCE\n'
                    'All existing service agreements must be reviewed for '
                    "compliance with Article 2 within 12 months of this "
                    "regulation's effective date."
                ),
                metadata={"type": "regulation", "domain": "REGULATORY"},
            ),
        ],
        ingestion_timestamp=datetime.now(),
    )

    # 5. Index corpus
    print("\n--- Indexing Corpus ---")
    indexed = orchestrator.index_corpus(corpus)
    node_count = len(indexed.graph_state.nodes)
    edge_count = len(indexed.graph_state.edges)
    print(f"Index complete: {node_count} nodes, {edge_count} edges")
    print(f"Coverage score: {indexed.extraction_metadata.coverage_score:.2f}")

    # 6. Build query
    query = QueryRequest(
        query_id="real-llm-q1",
        query_text=(
            "Does the MSA's liability cap contradict the regulation?"
        ),
    )

    # 7. Process query
    print("\n--- Processing Query ---")
    print(f'Query: "{query.query_text}"')
    router_decision, assembled_context = orchestrator.process_query(query)

    # 8. Print router decision (no hard assertions -- LLM output is nondeterministic)
    print(f"\nRouterDecision:")
    print(f"  Lane:            {router_decision.selected_lane.value}")
    print(f"  Graph scope:     {router_decision.requires_graph_scope}")
    print(f"  Intent:          {router_decision.intent.classification.value} "
          f"(conf={router_decision.intent.confidence})")
    print(f"  Targets:         docs={router_decision.identified_targets.explicit_docs}, "
          f"entities={router_decision.identified_targets.implicit_entities}")
    print(f"  Domain:          {router_decision.domain_expectation.value}")

    # 9. Print assembled context if available
    if assembled_context is not None:
        print(f"\n--- Edge Grammar ---")
        print(assembled_context.lci_payload.preamble.edge_grammar)

        print(f"\n--- Document Ordering ---")
        for doc in assembled_context.lci_payload.documents:
            print(f"  [{doc.injection_order}] {doc.doc_id} (role={doc.role})")
        print(f"  Total tokens: {assembled_context.lci_payload.total_tokens}")

        report = assembled_context.decontextualization_report
        print(f"\n--- Decontextualization Report ---")
        print(f"  Terms resolved:  {report.terms_resolved}")
        print(f"  Terms failed:    {len(report.terms_failed)}")
        for f in report.terms_failed:
            print(f"    - {f.term}: {f.failure_reason.value}")
        print(f"  Success rate:    {report.substitution_success_rate:.2f}")
    else:
        print("\n  (No assembled context -- query did not require graph scope)")

    print("\n" + "=" * 70)
    print("Real-LLM test complete. Review output above for correctness.")
    print("=" * 70)


def test_v5_pipeline() -> None:
    """End-to-end v5 pipeline test with mock LLM and embedding clients."""

    print("=" * 70)
    print("V5 Pipeline End-to-End Test")
    print("=" * 70)

    # 1. Create mock clients
    mock_llm = MockLLMClientV5()
    mock_embed = MockEmbeddingClientV5()

    # 2. Create config
    config: Dict[str, Any] = {
        "explorer": {},
        "librarian": {},
        "planner": {},
        "reasoner": {},
    }

    # 3. Create OrchestratorV5
    orchestrator = OrchestratorV5(mock_llm, mock_embed, config)

    # 4. Build documents (MSA + DORA)
    documents = [
        RawDocument(
            doc_id="MSA",
            content=(
                'MASTER SERVICE AGREEMENT\n\n'
                'Section 1. DEFINITIONS\n'
                '"Vendor" means Acme Corp.\n\n'
                'Section 2. LIMITATION OF LIABILITY\n'
                'The Vendor shall not be liable for any indirect, '
                'incidental, or consequential damages. '
                "The Vendor's total aggregate liability shall not "
                'exceed $1,000,000.'
            ),
            metadata={"type": "contract", "domain": "LEGAL"},
        ),
        RawDocument(
            doc_id="DORA",
            content=(
                'DIGITAL OPERATIONAL RESILIENCE ACT (DORA)\n\n'
                'Article 1. SCOPE\n'
                'This regulation establishes requirements for ICT risk '
                'management applicable to financial entities and their '
                'critical third-party service providers.\n\n'
                'Article 2. OPERATIONAL RESILIENCE REQUIREMENTS\n'
                'Financial entities shall ensure that their ICT systems '
                'and service providers maintain adequate operational '
                'resilience at all times.'
            ),
            metadata={"type": "regulation", "domain": "REGULATORY"},
        ),
    ]

    # 5. Index corpus
    print("\n--- Indexing Corpus (v5) ---")
    index = orchestrator.index_corpus_v5(documents)

    # Verify LibrarianOutput
    assert index.graph is not None, "Graph is None"
    node_count = len(index.graph.nodes)
    edge_count = len(index.graph.edges)
    print(f"  Nodes: {node_count}")
    print(f"  Edges: {edge_count}")
    assert node_count > 0, f"Expected nodes in graph, got {node_count}"
    assert edge_count > 0, f"Expected edges in graph, got {edge_count}"

    if index.community_structure:
        comm_count = len(index.community_structure.communities)
        print(f"  Communities: {comm_count}")
    else:
        print("  Communities: None")

    if index.document_store:
        print(f"  Document store: {len(index.document_store)} documents")

    # 6. Process query
    query = QueryRequest(
        query_id="v5-test-q1",
        query_text="Does the MSA contradict DORA?",
    )

    print(f"\n--- Processing Query (v5) ---")
    print(f'  Query: "{query.query_text}"')
    result = orchestrator.process_query_v5(query, index)

    # 7. Verify ReasonerOutput
    assert result.finding is not None, "Finding is None"
    finding = result.finding

    print(f"\n--- Finding ---")
    print(f"  Verdict:       {finding.verdict.verdict_type.value}")
    print(f"  Answer:        {finding.verdict.answer}")
    print(f"  Confidence:    {finding.verdict.confidence:.2f}")
    print(f"  Lane:          {finding.lane_used}")
    print(f"  Intent:        {finding.intent_classified}")
    print(f"  Relationships: {len(finding.relationships)}")
    print(f"  Evidence:      {len(finding.evidence)}")

    # Assert meaningful output
    assert len(finding.relationships) > 0, "Expected at least one relationship"
    assert finding.confidence_factors.overall > 0, "Expected confidence > 0"

    # 8. Print trace
    trace = result.trace
    print(f"\n--- Trace ---")
    print(f"  Lane:            {trace.lane_used}")
    print(f"  Relationships:   {trace.relationships_inferred}")
    print(f"  Evidence:        {trace.evidence_grounded}")
    print(f"  Verdict:         {trace.verdict_type} "
          f"(conf={trace.verdict_confidence:.2f})")

    print("\n" + "=" * 70)
    print("V5 pipeline test: All assertions passed.")
    print("=" * 70)


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging(log_file="pipeline_run.log")

    import sys as _sys
    if "--v5" in _sys.argv:
        test_v5_pipeline()
    elif "--explorer" in _sys.argv:
        test_with_explorer()
    elif "--real" in _sys.argv or os.getenv("LLM_PROVIDER") == "ollama":
        test_with_real_llm()
    else:
        test_orchestrator()
