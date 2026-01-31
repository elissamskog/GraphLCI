"""
Index Diagnostics for v5 Evaluation
Location: evaluation/diagnostics.py
Owner: Audit Agent

Provides diagnostic data collection and logging for the indexing pipeline.
Extracted from run_evaluation_v5.py for modularity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import OrchestratorV5
    from schemas.contracts import StructuralEdgeType

# Default printer (unbuffered)
def _default_print(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True so output appears immediately in pipes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================================
# INDEX DIAGNOSTICS
# ============================================================================

@dataclass
class IndexDiagnostics:
    """Diagnostic data captured from the indexing pipeline."""
    chunker_total_chunks: int = 0
    chunker_doc_count: int = 0
    chunker_token_stats: Dict[str, float] = field(default_factory=dict)
    chunker_section_headers_detected: int = 0
    chunker_chunks_per_doc: Dict[str, int] = field(default_factory=dict)
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

    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_document_count: int = 0
    graph_bridge_count: int = 0
    community_count: int = 0
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    docs_with_nodes: int = 0
    docs_without_nodes: int = 0

    coref_group_details: List[Dict[str, Any]] = field(default_factory=list)
    coref_multi_doc_groups: int = 0
    cross_doc_entity_overlap: List[Dict[str, Any]] = field(default_factory=list)
    cross_doc_overlap_entity_count: int = 0
    unique_entities_total: int = 0
    # Extracted triples (v5 evaluation)
    extracted_triples_count: int = 0
    extracted_triples_sample: List[Dict[str, Any]] = field(default_factory=list)
    # Synonymy edges (v5 evaluation)
    synonymy_edge_count: int = 0
    synonymy_edge_mean_score: float = 0.0
    synonymy_edge_min_score: float = 0.0
    synonymy_edge_max_score: float = 0.0
    synonymy_edges_sample: List[Dict[str, Any]] = field(default_factory=list)
    # Community summaries (v5 evaluation)
    community_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Per-stage performance (v5 evaluation)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_token_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)


def collect_index_diagnostics(
    orchestrator: "OrchestratorV5",
    index: Any,
    corpus_size: int,
) -> IndexDiagnostics:
    """Collect diagnostic data from Chunker, Extractor, Explorer, and Librarian."""
    from schemas.contracts import StructuralEdgeType

    diag = IndexDiagnostics()

    # --- Chunker trace ---
    chunker_out = orchestrator.last_chunker_output
    if chunker_out and chunker_out.trace:
        ct = chunker_out.trace
        diag.chunker_total_chunks = ct.total_chunks
        diag.chunker_doc_count = len(ct.input_doc_ids)
        if ct.chunks:
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
            cpc = Counter(c.source_doc_id for c in ct.chunks)
            diag.chunker_chunks_per_doc = dict(cpc.most_common(10))

    # --- Extractor trace ---
    explorer_out = orchestrator.last_explorer_output
    if explorer_out and explorer_out.trace:
        et = explorer_out.trace
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

    if explorer_out and explorer_out.node_set:
        ns = explorer_out.node_set
        if ns.extraction_config and isinstance(ns.extraction_config, dict):
            diag.extractor_error = ns.extraction_config.get("error")



    # --- Librarian / graph diagnostics ---
    graph = index.graph
    diag.graph_node_count = len(graph.nodes)
    diag.graph_edge_count = len(graph.edges)
    diag.graph_document_count = graph.document_count
    diag.graph_bridge_count = graph.bridge_count

    edge_types: Dict[str, int] = Counter()
    for edge in graph.edges.values():
        edge_types[edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type)] += 1
    diag.edge_type_distribution = dict(edge_types)

    node_types: Dict[str, int] = Counter()
    for node in graph.nodes.values():
        ntype = node.node_type if isinstance(node.node_type, str) else str(node.node_type)
        node_types[ntype] += 1
    diag.node_type_distribution = dict(node_types)

    if graph.node_by_doc:
        diag.docs_with_nodes = len(graph.node_by_doc)
        diag.docs_without_nodes = corpus_size - diag.docs_with_nodes

    if index.community_structure:
        diag.community_count = len(index.community_structure.communities)

    # --- Coreference analysis ---
    if explorer_out and explorer_out.node_set:
        ns = explorer_out.node_set
        if ns.coreference_groups:
            for cg in ns.coreference_groups:
                diag.coref_group_details.append({
                    "canonical_form": cg.canonical_form,
                    "variant_count": len(cg.variants),
                    "variants": cg.variants,
                })
            canonical_to_docs: Dict[str, set] = defaultdict(set)
            for node in ns.nodes:
                canonical_to_docs[node.canonical_text].add(node.source_doc_id)
            coref_canonicals = {cg.canonical_form for cg in ns.coreference_groups}
            diag.coref_multi_doc_groups = sum(
                1 for cf in coref_canonicals
                if len(canonical_to_docs.get(cf, set())) > 1
            )

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

    # --- Extracted triples ---
    if explorer_out and explorer_out.trace and explorer_out.trace.chunk_details:
        all_triples: List[Dict[str, Any]] = []
        for cd in explorer_out.trace.chunk_details:
            for claim in cd.claims:
                all_triples.append({
                    "subject": claim.subject_entity,
                    "predicate": claim.predicate,
                    "object": claim.object_entity,
                    "claim_text": claim.claim_text[:120],
                    "chunk_id": claim.chunk_id,
                    "confidence": claim.confidence,
                })
        diag.extracted_triples_count = len(all_triples)
        diag.extracted_triples_sample = all_triples[:100]

    # --- Synonymy edges ---
    synonymy_edges = [
        e for e in index.graph.edges.values()
        if e.edge_type == StructuralEdgeType.SYNONYMY
    ]
    diag.synonymy_edge_count = len(synonymy_edges)
    if synonymy_edges:
        scores = [e.similarity for e in synonymy_edges if e.similarity is not None]
        if scores:
            diag.synonymy_edge_min_score = min(scores)
            diag.synonymy_edge_max_score = max(scores)
            diag.synonymy_edge_mean_score = sum(scores) / len(scores)
        sorted_syn = sorted(
            synonymy_edges,
            key=lambda e: e.similarity if e.similarity is not None else 0.0,
            reverse=True,
        )
        for e in sorted_syn[:50]:
            diag.synonymy_edges_sample.append({
                "edge_id": e.edge_id,
                "source": e.source,
                "target": e.target,
                "similarity": e.similarity,
                "weight": e.weight,
                "is_cross_document": e.is_cross_document,
            })

    # --- Community summaries ---
    if index.community_structure:
        for cid, comm in index.community_structure.communities.items():
            diag.community_summaries.append({
                "community_id": comm.community_id,
                "level": comm.level,
                "node_count": len(comm.node_ids),
                "summary": comm.summary,
                "key_themes": comm.key_themes,
            })

    # --- Stage timings / token usage ---
    diag.stage_timings = orchestrator.stage_timings
    diag.stage_token_usage = orchestrator.stage_token_usage

    return diag


def log_index_diagnostics(
    diag: IndexDiagnostics,
    printer: Callable[..., None] = _default_print,
) -> None:
    """Print index diagnostics to stdout."""
    printer(f"\n  --- Chunker Diagnostics ---")
    printer(f"    Total chunks:          {diag.chunker_total_chunks}")
    printer(f"    Documents chunked:     {diag.chunker_doc_count}")
    if diag.chunker_token_stats:
        ts = diag.chunker_token_stats
        printer(f"    Token stats:           min={ts.get('min', 0):.0f} max={ts.get('max', 0):.0f} "
                f"mean={ts.get('mean', 0):.0f} median={ts.get('median', 0):.0f}")
    printer(f"    Section headers:       {diag.chunker_section_headers_detected} chunks")

    printer(f"\n  --- Extractor Diagnostics ---")
    if diag.extractor_error:
        printer(f"    *** EXTRACTOR ERROR: {diag.extractor_error} ***")
    printer(f"    Entities extracted:    {diag.extractor_entities_extracted}")
    printer(f"    Claims extracted:      {diag.extractor_claims_extracted}")
    printer(f"    Gleaning iterations:   {diag.extractor_gleaning_iterations}")
    printer(f"    LLM calls:            {diag.extractor_llm_calls}")
    printer(f"    Extraction time:       {diag.extractor_time_seconds:.1f}s")
    printer(f"    Chunks with 0 entities: {diag.extractor_chunks_with_zero_entities}")
    printer(f"    Chunks with 0 claims:   {diag.extractor_chunks_with_zero_claims}")
    printer(f"    Mean entities/chunk:    {diag.extractor_mean_entities_per_chunk:.2f}")
    printer(f"    Mean claims/chunk:      {diag.extractor_mean_claims_per_chunk:.2f}")



    printer(f"\n  --- Librarian Diagnostics ---")
    printer(f"    Graph nodes:           {diag.graph_node_count}")
    printer(f"    Graph edges:           {diag.graph_edge_count}")
    printer(f"    Documents in graph:    {diag.graph_document_count}")
    printer(f"    Bridge edges:          {diag.graph_bridge_count}")
    printer(f"    Communities:           {diag.community_count}")
    printer(f"    Docs with nodes:       {diag.docs_with_nodes}")
    printer(f"    Docs without nodes:    {diag.docs_without_nodes}")

    if diag.edge_type_distribution:
        printer(f"    Edge types:")
        for etype, count in sorted(diag.edge_type_distribution.items(), key=lambda x: -x[1]):
            printer(f"      {etype}: {count}")

    if diag.node_type_distribution:
        printer(f"    Node types:")
        for ntype, count in sorted(diag.node_type_distribution.items(), key=lambda x: -x[1]):
            printer(f"      {ntype}: {count}")

    printer(f"\n  --- Coreference Analysis ---")
    printer(f"    Total coref groups:    {len(diag.coref_group_details)}")
    printer(f"    Multi-doc groups:      {diag.coref_multi_doc_groups}")
    if diag.unique_entities_total > 0:
        pct = diag.cross_doc_overlap_entity_count / diag.unique_entities_total * 100
        printer(f"    Cross-doc entities:    {diag.cross_doc_overlap_entity_count} / "
                f"{diag.unique_entities_total} ({pct:.1f}%)")
    else:
        printer(f"    Cross-doc entities:    0 / 0 (0.0%)")

    printer(f"\n  --- Extracted Triples ---")
    printer(f"    Total triples:         {diag.extracted_triples_count}")
    if diag.extracted_triples_sample:
        printer(f"    Sample (first 5):")
        for t in diag.extracted_triples_sample[:5]:
            printer(f"      ({t['subject']}) --[{t['predicate']}]--> ({t['object']})"
                    f"  conf={t['confidence']:.2f}")

    printer(f"\n  --- Synonymy Edges ---")
    printer(f"    Count:                 {diag.synonymy_edge_count}")
    if diag.synonymy_edge_count > 0:
        printer(f"    Similarity:            min={diag.synonymy_edge_min_score:.4f} "
                f"max={diag.synonymy_edge_max_score:.4f} "
                f"mean={diag.synonymy_edge_mean_score:.4f}")
        if diag.synonymy_edges_sample:
            printer(f"    Top 5 edges:")
            for e in diag.synonymy_edges_sample[:5]:
                printer(f"      {e['source']} <-> {e['target']}  "
                        f"sim={e['similarity']:.4f}  cross_doc={e['is_cross_document']}")

    printer(f"\n  --- Community Summaries ---")
    printer(f"    Communities:           {len(diag.community_summaries)}")
    for cs in diag.community_summaries[:5]:
        printer(f"    [{cs['community_id']}] level={cs['level']} nodes={cs['node_count']}")
        if cs.get('summary'):
            printer(f"      Summary: {cs['summary'][:120]}")
        if cs.get('key_themes'):
            printer(f"      Themes: {', '.join(cs['key_themes'][:5])}")

    if diag.stage_timings:
        printer(f"\n  --- Stage Timings ---")
        for stage, ms in sorted(diag.stage_timings.items()):
            printer(f"    {stage}: {ms:.0f}ms")

    if diag.stage_token_usage:
        printer(f"\n  --- Stage Token Usage ---")
        for stage, usage in sorted(diag.stage_token_usage.items()):
            if usage.get("total_tokens", 0) > 0 or usage.get("call_count", 0) > 0:
                printer(f"    {stage}: prompt={usage.get('prompt_tokens', 0)} "
                        f"completion={usage.get('completion_tokens', 0)} "
                        f"total={usage.get('total_tokens', 0)} "
                        f"calls={usage.get('call_count', 0)}")
