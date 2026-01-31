"""
Librarian Controller
Location: librarian/librarian_controller.py
Owner: LIBRARIAN Instance

Responsibility: Orchestrate INDEXER, RESOLVER, and ASSEMBLER pipelines.
Receives ``LibrarianRequest`` from ROOT, dispatches to the appropriate
sub-pipeline, and returns ``LibrarianResponse``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from schemas.contracts import (
    AssembledContext,
    DecontextualizationReport,
    DocumentSummary,
    DomainTaxonomy,
    ExplorationResult,
    ExtractionMetadata,
    GraphState,
    IndexedCorpus,
    IndexRequest,
    LibrarianRequest,
    LibrarianResponse,
    NodeType,
    RawCorpus,
    RawDocument,
    ScopeRequest,
    SubstitutionFailure,
    VectorState,
)

# Indexer
from librarian.indexer.extractor import (
    ExtractionInput,
    ExtractionOutput,
    Extractor,
)

# Resolver
from librarian.resolver.scope_resolver import (
    ResolverInput,
    ScopeResolver,
)

# Assembler
from librarian.assembler.decontextualizer import (
    DecontextInput,
    Decontextualizer,
)
from librarian.assembler.context_builder import (
    ContextBuildInput,
    ContextBuilder,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM CLIENT PROTOCOL
# ============================================================================

class LLMClient(Protocol):
    """Protocol for the underlying LLM call."""
    def generate(self, prompt: str) -> str: ...


# ============================================================================
# CONTROLLER
# ============================================================================

class LibrarianController:
    """Main controller for the LIBRARIAN instance.

    Receives ``LibrarianRequest`` messages from ROOT and returns
    ``LibrarianResponse``.  Internally orchestrates:

    * **INDEX** — extract entities/edges from every document, assemble
      ``GraphState`` and ``VectorState``, persist as ``IndexedCorpus``.
    * **RESOLVE_AND_ASSEMBLE** — run scope resolution over the indexed
      graph, decontextualize documents, and build the LCI payload.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol (shared by all
        LLM-using sub-components).
    config:
        System-level configuration dictionary.  Relevant sub-keys:

        - ``indexer.extraction.*`` — forwarded to ``Extractor``
        - ``resolver.*`` — forwarded to ``ScopeResolver``
        - ``assembler.decontextualization.*`` — forwarded to ``Decontextualizer``
        - ``assembler.context.*`` — forwarded to ``ContextBuilder``
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

        # --- Sub-component configs ---
        indexer_cfg = self._config.get("indexer", {}).get("extraction", {})
        resolver_cfg = self._config.get("resolver", {})
        decontext_cfg = self._config.get("assembler", {}).get("decontextualization", {})
        context_cfg = self._config.get("assembler", {}).get("context", {})

        # --- Indexer components ---
        self.extractor = Extractor(llm_client, indexer_cfg)

        # --- Resolver components ---
        resolver_traversal_cfg = {
            **resolver_cfg.get("traversal", {}),
            **resolver_cfg.get("scope", {}),
        }
        self.scope_resolver = ScopeResolver(llm_client, resolver_traversal_cfg)

        # --- Assembler components ---
        self.decontextualizer = Decontextualizer(llm_client, decontext_cfg)
        self.context_builder = ContextBuilder(context_cfg)

        # --- State ---
        self._indexed_corpus: Optional[IndexedCorpus] = None
        self._document_store: Dict[str, RawDocument] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, request: LibrarianRequest) -> LibrarianResponse:
        """Route a request to the appropriate pipeline."""
        if request.request_type == "INDEX":
            return self._handle_index(request.payload)
        elif request.request_type == "RESOLVE_AND_ASSEMBLE":
            return self._handle_resolve_and_assemble(request.payload)
        else:
            return LibrarianResponse(
                success=False,
                payload=None,
                errors=[f"Unknown request type: {request.request_type}"],
            )

    @property
    def indexed_corpus(self) -> Optional[IndexedCorpus]:
        """Access the currently indexed corpus (read-only)."""
        return self._indexed_corpus

    # ------------------------------------------------------------------
    # INDEX pipeline
    # ------------------------------------------------------------------

    def _handle_index(self, payload: Any) -> LibrarianResponse:
        """Offline indexing pipeline.

        Accepts either an ``IndexRequest`` (new two-stage pipeline) or a
        bare ``RawCorpus`` (legacy callers).

        Flow
        ----
        1. Extract entities/edges from each document (with gleaning)
        2. Merge extractions into a unified ``GraphState``
        3. Build a placeholder ``VectorState`` (full implementation in
           ``indexer/vector_store.py``)
        4. Persist ``IndexedCorpus`` and raw documents for runtime use
        """
        try:
            # Unwrap payload — support both IndexRequest and legacy RawCorpus
            if isinstance(payload, IndexRequest):
                corpus = payload.corpus
                exploration = payload.exploration
            else:
                corpus = payload  # Legacy: bare RawCorpus
                exploration = None

            # Extract exploration data when available
            taxonomy: Optional[DomainTaxonomy] = None
            summary_lookup: Dict[str, DocumentSummary] = {}
            if exploration is not None:
                taxonomy = exploration.taxonomy
                summary_lookup = {s.doc_id: s for s in exploration.summaries}

            all_extractions: List[ExtractionOutput] = []
            extraction_errors: List[str] = []

            # Build corpus context for cross-document extraction.
            # Use exploration summaries when available for richer data.
            if summary_lookup:
                corpus_context_full: List[Dict[str, Any]] = []
                for doc in corpus.documents:
                    entry: Dict[str, str] = {"doc_id": doc.doc_id}
                    ds = summary_lookup.get(doc.doc_id)
                    if ds is not None:
                        entry["summary"] = ds.summary
                        entry["document_type"] = ds.document_type
                        entry["key_themes"] = ds.key_themes
                    else:
                        entry["summary"] = doc.content[:200].replace("\n", " ")
                    corpus_context_full.append(entry)
            else:
                corpus_context_full = [
                    {"doc_id": doc.doc_id, "summary": doc.content[:200].replace("\n", " ")}
                    for doc in corpus.documents
                ]

            for doc in corpus.documents:
                logger.info("Extracting from document: %s", doc.doc_id)
                try:
                    other_docs = [c for c in corpus_context_full if c["doc_id"] != doc.doc_id]
                    extraction = self.extractor.extract(
                        ExtractionInput(
                            document=doc,
                            extraction_config={},
                            corpus_context=other_docs if other_docs else None,
                            taxonomy=taxonomy,
                            document_summary=summary_lookup.get(doc.doc_id),
                        )
                    )
                    all_extractions.append(extraction)
                except Exception as exc:
                    msg = f"Extraction failed for {doc.doc_id}: {exc}"
                    logger.error(msg)
                    extraction_errors.append(msg)

            # Merge into unified graph
            graph_state = self._merge_graph_state(all_extractions)

            # Placeholder vector state (vector_store.py not yet implemented)
            vector_state = VectorState(chunks=[])

            # Coverage metric: fraction of documents that extracted at
            # least one non-document node
            coverage = self._calculate_coverage(all_extractions, corpus)

            self._indexed_corpus = IndexedCorpus(
                corpus_id=corpus.corpus_id,
                graph_state=graph_state,
                vector_state=vector_state,
                extraction_metadata=ExtractionMetadata(
                    gleaning_iterations=sum(
                        e.gleaning_iterations for e in all_extractions
                    ),
                    extraction_errors=extraction_errors,
                    coverage_score=coverage,
                ),
            )

            # Cache raw documents for runtime retrieval
            self._document_store = {doc.doc_id: doc for doc in corpus.documents}

            logger.info(
                "Indexing complete: %d nodes, %d edges, coverage=%.2f",
                len(graph_state.nodes),
                len(graph_state.edges),
                coverage,
            )

            return LibrarianResponse(
                success=True,
                payload=self._indexed_corpus,
                errors=extraction_errors,
            )

        except Exception as exc:
            logger.exception("Indexing pipeline failed")
            return LibrarianResponse(
                success=False,
                payload=None,
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # RESOLVE_AND_ASSEMBLE pipeline
    # ------------------------------------------------------------------

    def _handle_resolve_and_assemble(
        self,
        scope_request: ScopeRequest,
    ) -> LibrarianResponse:
        """Runtime scope resolution and context assembly.

        Flow
        ----
        1. Resolve scope (PPR traversal + LLM edge interpretation)
        2. Fetch raw documents in scope
        3. Collect definitions and decontextualize each document
        4. Build the LCI payload via ``ContextBuilder``
        5. Compose ``AssembledContext``
        """
        try:
            if not self._indexed_corpus:
                raise ValueError("Corpus not indexed. Run INDEX first.")

            router_decision = scope_request.router_decision

            # --- Step 1: Scope resolution ---
            resolver_output = self.scope_resolver.resolve(
                ResolverInput(
                    router_decision=router_decision,
                    graph_state=self._indexed_corpus.graph_state,
                    traversal_config=scope_request.traversal_config or {},
                    query_text=scope_request.query_text,
                )
            )
            scope_object = resolver_output.scope_object

            logger.info(
                "Scope resolved: %d primary, %d supporting, %d edges — [%s]",
                len(scope_object.primary_documents),
                len(scope_object.supporting_documents),
                len(scope_object.edge_context),
                ", ".join(resolver_output.traversal_trace),
            )

            # --- Step 2: Fetch documents ---
            all_doc_ids = (
                [d.doc_id for d in scope_object.primary_documents]
                + [d.doc_id for d in scope_object.supporting_documents]
            )
            raw_documents = self._fetch_documents(all_doc_ids)

            # --- Step 3: Decontextualize ---
            definitions = self._collect_definitions(self._indexed_corpus.graph_state)
            max_tokens_per_unit = (
                self._config
                .get("assembler", {})
                .get("decontextualization", {})
                .get("max_tokens_per_unit", 512)
            )

            decontextualized: Dict[str, str] = {}
            agg_resolved = 0
            agg_failed: List[SubstitutionFailure] = []

            for doc in raw_documents:
                decontext_output = self.decontextualizer.process(
                    DecontextInput(
                        content=doc.content,
                        definitions=definitions,
                        max_tokens=max_tokens_per_unit,
                    )
                )
                decontextualized[doc.doc_id] = decontext_output.decontextualized_content
                agg_resolved += decontext_output.report.terms_resolved
                agg_failed.extend(decontext_output.report.terms_failed)

            total_terms = agg_resolved + len(agg_failed)
            decontext_report = DecontextualizationReport(
                terms_resolved=agg_resolved,
                terms_failed=agg_failed,
                substitution_success_rate=(
                    agg_resolved / total_terms if total_terms > 0 else 1.0
                ),
            )

            # --- Step 4: Build LCI payload ---
            max_context_tokens = (
                self._config
                .get("assembler", {})
                .get("context", {})
                .get("max_total_tokens", 200_000)
            )

            context_output = self.context_builder.build(
                ContextBuildInput(
                    scope_object=scope_object,
                    raw_documents=raw_documents,
                    decontextualized_contents=decontextualized,
                    max_context_tokens=max_context_tokens,
                )
            )

            # --- Step 5: Compose result ---
            assembled_context = AssembledContext(
                query_id=scope_request.query_id,
                lci_payload=context_output.lci_payload,
                decontextualization_report=decontext_report,
                assembly_warnings=context_output.assembly_warnings,
            )

            return LibrarianResponse(
                success=True,
                payload=assembled_context,
                errors=[],
            )

        except Exception as exc:
            logger.exception("Resolve-and-assemble pipeline failed")
            return LibrarianResponse(
                success=False,
                payload=None,
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_graph_state(extractions: List[ExtractionOutput]) -> GraphState:
        """Merge per-document extractions into a single ``GraphState``."""
        from schemas.contracts import GraphNode, GraphEdge

        all_nodes: List[GraphNode] = []
        all_edges: List[GraphEdge] = []
        seen_node_ids: set[str] = set()
        seen_edge_ids: set[str] = set()

        for extraction in extractions:
            for node in extraction.nodes:
                if node.node_id not in seen_node_ids:
                    all_nodes.append(node)
                    seen_node_ids.add(node.node_id)
            for edge in extraction.edges:
                if edge.edge_id not in seen_edge_ids:
                    all_edges.append(edge)
                    seen_edge_ids.add(edge.edge_id)

        return GraphState(nodes=all_nodes, edges=all_edges)

    @staticmethod
    def _calculate_coverage(
        extractions: List[ExtractionOutput],
        corpus: RawCorpus,
    ) -> float:
        """Fraction of documents that produced at least one entity/clause."""
        if not corpus.documents:
            return 0.0

        docs_with_entities = sum(
            1 for ext in extractions
            if any(
                n.node_type != NodeType.DOCUMENT
                for n in ext.nodes
            )
        )
        return docs_with_entities / len(corpus.documents)

    def _fetch_documents(self, doc_ids: List[str]) -> List[RawDocument]:
        """Retrieve raw documents from the in-memory store."""
        documents: List[RawDocument] = []
        for doc_id in doc_ids:
            doc = self._document_store.get(doc_id)
            if doc:
                documents.append(doc)
            else:
                logger.warning("Document %s not found in store", doc_id)
        return documents

    @staticmethod
    def _collect_definitions(graph_state: GraphState) -> Dict[str, str]:
        """Extract a term→definition mapping from DEFINITION nodes."""
        definitions: Dict[str, str] = {}
        for node in graph_state.nodes:
            if node.node_type == NodeType.DEFINITION:
                for term in node.defined_terms:
                    if term and term not in definitions:
                        definitions[term] = node.content_summary
        return definitions


# ============================================================================
# v5 CONTROLLER
# ============================================================================

from schemas.contracts import (
    CommunityStructure,
    EmbeddedNode,
    KnowledgeGraph,
    LibrarianInput,
    LibrarianOutput,
    NodeSet,
    StructuralEdge,
)

from librarian.similarity_index import SimilarityIndex
from librarian.edge_builder import (
    extract_explicit_edges,
    create_synonymy_edges,
    boost_bridge_edges,
    discount_synonymy_by_degree,
)
from librarian.graph_assembler import assemble_graph
from librarian.community_detector import detect_communities
from librarian.community_summarizer import summarize_communities


class LibrarianControllerV5:
    """v5 LIBRARIAN pipeline controller.

    Chains: NodeSet → SimilarityIndex → edges → KnowledgeGraph
    → community detection → summarization → LibrarianOutput.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None) -> None:
        self._llm = llm_client
        self._config = config or {}

    def process(self, inp: LibrarianInput) -> LibrarianOutput:
        """Run the full v5 LIBRARIAN pipeline."""
        # 1. Build nodes dict
        nodes_dict: Dict[str, EmbeddedNode] = {
            n.node_id: n for n in inp.node_set.nodes
        }

        # 2. Build similarity index
        sim_index = SimilarityIndex()
        sim_index.add_nodes(nodes_dict)

        # 3. Extract explicit edges (pass chunk_node_map when available)
        explicit_edges = extract_explicit_edges(
            nodes_dict,
            chunk_node_map=inp.chunk_node_map,
        )

        # 4. Create synonymy edges (pass similarity_index)
        coreference_groups = inp.node_set.coreference_groups or []
        synonymy_cfg = self._config.get("synonymy", {})
        threshold = synonymy_cfg.get("threshold", 0.80)
        cross_doc_threshold = synonymy_cfg.get("cross_doc_threshold", None)
        max_edges_per_node = synonymy_cfg.get("max_edges_per_node", 10)
        min_cross_doc_edges = synonymy_cfg.get("min_cross_doc_edges", 3)
        synonymy_edges = create_synonymy_edges(
            nodes_dict,
            coreference_groups=coreference_groups if coreference_groups else None,
            similarity_threshold=threshold,
            cross_doc_threshold=cross_doc_threshold,
            similarity_index=sim_index,
            max_edges_per_node=max_edges_per_node,
            min_cross_doc_edges=min_cross_doc_edges,
        )

        # 5. Combine edges, apply degree-discounting, boost bridges, assemble graph
        all_edges = explicit_edges + synonymy_edges

        # P2b: Discount SYNONYMY edges by node degree to prevent similarity clusters
        synonymy_discount = self._config.get("synonymy_degree_discount", True)
        if synonymy_discount:
            discount_synonymy_by_degree(all_edges, nodes_dict)

        # P3: Boost bridge edge weights
        bridge_multiplier = self._config.get("bridge_edge_multiplier", 1.5)
        if bridge_multiplier != 1.0:
            boost_bridge_edges(all_edges, multiplier=bridge_multiplier)

        graph = assemble_graph(nodes_dict, all_edges)

        # 6. Community detection
        community_structure: Optional[CommunityStructure] = None
        community_cfg = self._config.get("community_detection", {})
        if community_cfg.get("enabled", True):
            max_levels = community_cfg.get("max_levels", 3)
            resolution = community_cfg.get("resolution", 1.0)
            min_community_size = community_cfg.get("min_community_size", 3)
            community_structure = detect_communities(
                graph,
                max_levels=max_levels,
                resolution=resolution,
                min_community_size=min_community_size,
            )

            # 7. Community summarization
            summarization_cfg = self._config.get("summarization", {})
            if summarization_cfg.get("enabled", True) and community_structure:
                summarize_communities(graph, community_structure, self._llm)

        # 8. Persist similarity index if configured
        similarity_index_path: Optional[str] = None
        persist_path = (
            self._config
            .get("similarity_index", {})
            .get("persist_path")
        )
        if persist_path:
            sim_index.save(persist_path)
            similarity_index_path = persist_path

        # 9. Return output
        return LibrarianOutput(
            graph=graph,
            community_structure=community_structure,
            document_store=inp.document_store,
            similarity_index_path=similarity_index_path,
        )
