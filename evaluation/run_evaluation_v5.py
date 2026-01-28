"""
Unified v5 Evaluation Runner
Location: evaluation/run_evaluation_v5.py
Owner: ROOT Instance

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
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
from evaluation.data_loaders.scifact_loader import SciFactLoader
from evaluation.data_loaders.wikifactclaims_loader import WikiFactClaimsLoader
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

class DatasetAdapter(ABC):
    """Abstract base for dataset-specific loading and feature flags."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def load_groups(self, limit: Optional[int] = None, **kwargs) -> List[EvalGroup]: ...

    # Feature flags — concrete adapters override to return True
    has_two_stage: bool = False
    has_evidence_doc_recall: bool = False
    has_per_type_level: bool = False
    has_index_diagnostics: bool = False
    has_checkpointing: bool = False


class SciFactAdapter(DatasetAdapter):
    """Adapter for the SciFact full-corpus evaluation."""

    name = "scifact"
    display_name = "SciFact Full-Corpus"
    has_two_stage = True
    has_evidence_doc_recall = True
    has_index_diagnostics = True
    has_checkpointing = True

    def load_groups(
        self,
        limit: Optional[int] = None,
        small_corpus: bool = False,
        **kw,
    ) -> List[EvalGroup]:
        loader = SciFactLoader()
        test_cases = loader.get_test_cases(limit=limit)
        _print(f"  Claims: {len(test_cases)} test cases" +
               (f" (limited to {limit})" if limit else ""))

        if small_corpus and limit:
            target = limit * 5
            documents = build_small_corpus(loader, test_cases, target)
            _print(f"  Small corpus: {len(documents)} documents "
                   f"(target: {target}, evidence docs included)")
        else:
            if small_corpus and limit is None:
                _print("WARNING: --small-corpus requires --limit N. "
                       "Falling back to full corpus.")
            documents = loader.get_all_abstracts()

        _print(f"  Corpus: {len(documents)} documents")

        # Log evidence coverage in small-corpus mode
        evidence_ids: set = set()
        for case in test_cases:
            evidence_ids.update(case.expected_evidence.keys())
        if small_corpus and limit is not None:
            included_ids = {d.doc_id for d in documents}
            missing = evidence_ids - included_ids
            if missing:
                _print(f"  WARNING: {len(missing)} evidence docs missing "
                       f"from corpus: {missing}")
            else:
                _print(f"  All {len(evidence_ids)} evidence docs present "
                       "in corpus")

        claims = [
            EvalClaim(
                claim_id=str(c.claim_id),
                claim_text=c.claim_text,
                expected_verdict=c.expected_verdict,
                gold_doc_ids=list(c.expected_evidence.keys()),
                expected_evidence=c.expected_evidence,
            )
            for c in test_cases
        ]
        return [EvalGroup(
            group_id="scifact-full",
            claims=claims,
            documents=documents,
            metadata={
                "corpus_mode": "small-corpus" if (small_corpus and limit) else "full",
            },
        )]


class WikiFactClaimsAdapter(DatasetAdapter):
    """Adapter for the WikiFactClaims per-row evaluation."""

    name = "wikifactclaims"
    display_name = "WikiFactClaims Per-Row"
    has_per_type_level = True
    has_evidence_doc_recall = True

    def __init__(self, data_path: Optional[str] = None):
        self._data_path = data_path

    def load_groups(
        self,
        limit: Optional[int] = None,
        **kw,
    ) -> List[EvalGroup]:
        loader = WikiFactClaimsLoader(path=self._data_path)
        rows = loader.get_rows(limit=limit)
        _print(f"  Rows: {len(rows)}" +
               (f" (limited to {limit})" if limit else ""))
        total_claims = sum(len(r.claims) for r in rows)
        _print(f"  Total claims: {total_claims}")

        groups: List[EvalGroup] = []
        for row in rows:
            documents = WikiFactClaimsLoader.build_corpus_for_row(row)
            claims = [
                EvalClaim(
                    claim_id=f"wiki-{row.row_id}-c{i}",
                    claim_text=c.text,
                    expected_verdict=c.label,
                    gold_doc_ids=list(set(row.supporting_facts_titles)),
                    expected_evidence={t: [] for t in set(row.supporting_facts_titles)},
                    metadata={"claim_index": i},
                )
                for i, c in enumerate(row.claims)
            ]
            groups.append(EvalGroup(
                group_id=row.row_id,
                claims=claims,
                documents=documents,
                metadata={
                    "question": row.question,
                    "question_type": row.question_type,
                    "level": row.level,
                },
            ))
        return groups


_ADAPTER_REGISTRY: Dict[str, type] = {
    "scifact": SciFactAdapter,
    "wikifactclaims": WikiFactClaimsAdapter,
}


def get_adapter(
    dataset: str,
    data_path: Optional[str] = None,
) -> DatasetAdapter:
    """Instantiate the adapter for the given dataset name."""
    cls = _ADAPTER_REGISTRY.get(dataset)
    if cls is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {', '.join(_ADAPTER_REGISTRY)}"
        )
    if dataset == "wikifactclaims":
        return cls(data_path=data_path)
    return cls()


# ============================================================================
# SMALL-CORPUS BUILDER (SciFact only)
# ============================================================================

def build_small_corpus(
    loader: Any,
    test_cases: List[Any],
    target_size: int,
) -> list:
    """Build a focused corpus containing evidence docs + random distractors.

    Guarantees every evidence document from the selected test cases is included.
    Pads with random non-evidence documents to reach target_size.
    If evidence docs alone exceed target_size, returns all evidence docs (no cap).
    """
    import random

    all_docs = loader.get_all_abstracts()
    docs_by_id = {d.doc_id: d for d in all_docs}

    evidence_ids: set = set()
    for case in test_cases:
        evidence_ids.update(case.expected_evidence.keys())

    evidence_docs = [docs_by_id[did] for did in evidence_ids if did in docs_by_id]

    needed = max(0, target_size - len(evidence_docs))
    if needed > 0:
        non_evidence = [d for d in all_docs if d.doc_id not in evidence_ids]
        distractors = random.sample(non_evidence, min(needed, len(non_evidence)))
    else:
        distractors = []

    documents = evidence_docs + distractors
    random.shuffle(documents)
    return documents


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
# INDEX DIAGNOSTICS (SciFact)
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
    explorer_chunk_count: int = 0
    explorer_entity_count: int = 0
    explorer_claim_count: int = 0
    explorer_node_count: int = 0
    explorer_source_doc_count: int = 0
    explorer_coref_groups: int = 0
    explorer_avg_claims_per_chunk: float = 0.0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    graph_document_count: int = 0
    graph_bridge_count: int = 0
    community_count: int = 0
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    docs_with_nodes: int = 0
    docs_without_nodes: int = 0
    explorer_error: Optional[str] = None
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
    orchestrator: OrchestratorV5,
    index: Any,
    corpus_size: int,
) -> IndexDiagnostics:
    """Collect diagnostic data from Chunker, Extractor, Explorer, and Librarian."""
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

    # --- Explorer diagnostics ---
    if explorer_out and explorer_out.node_set:
        ns = explorer_out.node_set
        diag.explorer_chunk_count = ns.chunk_count
        diag.explorer_entity_count = ns.entity_count
        diag.explorer_claim_count = ns.claim_count
        diag.explorer_node_count = len(ns.nodes)
        diag.explorer_source_doc_count = len(ns.source_doc_ids)
        diag.explorer_avg_claims_per_chunk = ns.avg_claims_per_chunk or 0.0
        if ns.coreference_groups:
            diag.explorer_coref_groups = len(ns.coreference_groups)
        if ns.extraction_config and "error" in ns.extraction_config:
            diag.explorer_error = ns.extraction_config["error"]

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


def log_index_diagnostics(diag: IndexDiagnostics) -> None:
    """Print index diagnostics to stdout."""
    _print(f"\n  --- Chunker Diagnostics ---")
    _print(f"    Total chunks:          {diag.chunker_total_chunks}")
    _print(f"    Documents chunked:     {diag.chunker_doc_count}")
    if diag.chunker_token_stats:
        ts = diag.chunker_token_stats
        _print(f"    Token stats:           min={ts.get('min', 0):.0f} max={ts.get('max', 0):.0f} "
               f"mean={ts.get('mean', 0):.0f} median={ts.get('median', 0):.0f}")
    _print(f"    Section headers:       {diag.chunker_section_headers_detected} chunks")

    _print(f"\n  --- Extractor Diagnostics ---")
    if diag.extractor_error:
        _print(f"    *** EXTRACTOR ERROR: {diag.extractor_error} ***")
    _print(f"    Entities extracted:    {diag.extractor_entities_extracted}")
    _print(f"    Claims extracted:      {diag.extractor_claims_extracted}")
    _print(f"    Gleaning iterations:   {diag.extractor_gleaning_iterations}")
    _print(f"    LLM calls:            {diag.extractor_llm_calls}")
    _print(f"    Extraction time:       {diag.extractor_time_seconds:.1f}s")
    _print(f"    Chunks with 0 entities: {diag.extractor_chunks_with_zero_entities}")
    _print(f"    Chunks with 0 claims:   {diag.extractor_chunks_with_zero_claims}")
    _print(f"    Mean entities/chunk:    {diag.extractor_mean_entities_per_chunk:.2f}")
    _print(f"    Mean claims/chunk:      {diag.extractor_mean_claims_per_chunk:.2f}")

    _print(f"\n  --- Explorer Diagnostics ---")
    if diag.explorer_error:
        _print(f"    *** EXPLORER ERROR: {diag.explorer_error} ***")
    _print(f"    Chunks produced:       {diag.explorer_chunk_count}")
    _print(f"    Entities extracted:    {diag.explorer_entity_count}")
    _print(f"    Claims extracted:      {diag.explorer_claim_count}")
    _print(f"    Nodes built:           {diag.explorer_node_count}")
    _print(f"    Source docs covered:   {diag.explorer_source_doc_count}")
    _print(f"    Coref groups:          {diag.explorer_coref_groups}")
    _print(f"    Avg claims/chunk:      {diag.explorer_avg_claims_per_chunk:.2f}")

    _print(f"\n  --- Librarian Diagnostics ---")
    _print(f"    Graph nodes:           {diag.graph_node_count}")
    _print(f"    Graph edges:           {diag.graph_edge_count}")
    _print(f"    Documents in graph:    {diag.graph_document_count}")
    _print(f"    Bridge edges:          {diag.graph_bridge_count}")
    _print(f"    Communities:           {diag.community_count}")
    _print(f"    Docs with nodes:       {diag.docs_with_nodes}")
    _print(f"    Docs without nodes:    {diag.docs_without_nodes}")

    if diag.edge_type_distribution:
        _print(f"    Edge types:")
        for etype, count in sorted(diag.edge_type_distribution.items(), key=lambda x: -x[1]):
            _print(f"      {etype}: {count}")

    if diag.node_type_distribution:
        _print(f"    Node types:")
        for ntype, count in sorted(diag.node_type_distribution.items(), key=lambda x: -x[1]):
            _print(f"      {ntype}: {count}")

    _print(f"\n  --- Coreference Analysis ---")
    _print(f"    Total coref groups:    {len(diag.coref_group_details)}")
    _print(f"    Multi-doc groups:      {diag.coref_multi_doc_groups}")
    if diag.unique_entities_total > 0:
        pct = diag.cross_doc_overlap_entity_count / diag.unique_entities_total * 100
        _print(f"    Cross-doc entities:    {diag.cross_doc_overlap_entity_count} / "
               f"{diag.unique_entities_total} ({pct:.1f}%)")
    else:
        _print(f"    Cross-doc entities:    0 / 0 (0.0%)")

    _print(f"\n  --- Extracted Triples ---")
    _print(f"    Total triples:         {diag.extracted_triples_count}")
    if diag.extracted_triples_sample:
        _print(f"    Sample (first 5):")
        for t in diag.extracted_triples_sample[:5]:
            _print(f"      ({t['subject']}) --[{t['predicate']}]--> ({t['object']})"
                   f"  conf={t['confidence']:.2f}")

    _print(f"\n  --- Synonymy Edges ---")
    _print(f"    Count:                 {diag.synonymy_edge_count}")
    if diag.synonymy_edge_count > 0:
        _print(f"    Similarity:            min={diag.synonymy_edge_min_score:.4f} "
               f"max={diag.synonymy_edge_max_score:.4f} "
               f"mean={diag.synonymy_edge_mean_score:.4f}")
        if diag.synonymy_edges_sample:
            _print(f"    Top 5 edges:")
            for e in diag.synonymy_edges_sample[:5]:
                _print(f"      {e['source']} <-> {e['target']}  "
                       f"sim={e['similarity']:.4f}  cross_doc={e['is_cross_document']}")

    _print(f"\n  --- Community Summaries ---")
    _print(f"    Communities:           {len(diag.community_summaries)}")
    for cs in diag.community_summaries[:5]:
        _print(f"    [{cs['community_id']}] level={cs['level']} nodes={cs['node_count']}")
        if cs.get('summary'):
            _print(f"      Summary: {cs['summary'][:120]}")
        if cs.get('key_themes'):
            _print(f"      Themes: {', '.join(cs['key_themes'][:5])}")

    if diag.stage_timings:
        _print(f"\n  --- Stage Timings ---")
        for stage, ms in sorted(diag.stage_timings.items()):
            _print(f"    {stage}: {ms:.0f}ms")

    if diag.stage_token_usage:
        _print(f"\n  --- Stage Token Usage ---")
        for stage, usage in sorted(diag.stage_token_usage.items()):
            if usage.get("total_tokens", 0) > 0 or usage.get("call_count", 0) > 0:
                _print(f"    {stage}: prompt={usage.get('prompt_tokens', 0)} "
                       f"completion={usage.get('completion_tokens', 0)} "
                       f"total={usage.get('total_tokens', 0)} "
                       f"calls={usage.get('call_count', 0)}")


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

        # Evidence doc recall (SciFact)
        if adapter.has_evidence_doc_recall and claim.expected_evidence:
            result.evidence_doc_recall = compute_evidence_doc_recall(
                finding, claim.expected_evidence,
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
# MARKDOWN FORMATTER
# ============================================================================

def format_output_md(summary: UnifiedRunSummary, adapter: DatasetAdapter) -> str:
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
    orchestrator: Optional[OrchestratorV5] = None,
    trace_chunks: int = 50,
    trace_level: TraceLevel = TraceLevel.STANDARD,
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
                "explorer": {
                    "chunk_count": d.explorer_chunk_count,
                    "entity_count": d.explorer_entity_count,
                    "claim_count": d.explorer_claim_count,
                    "node_count": d.explorer_node_count,
                    "source_doc_count": d.explorer_source_doc_count,
                    "coref_groups": d.explorer_coref_groups,
                    "avg_claims_per_chunk": d.explorer_avg_claims_per_chunk,
                    "error": d.explorer_error,
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
                        # Build chunk_id → full text for FULL level
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
                    # Build chunk_id → full text lookup for FULL trace
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
    _print(f"  Trace written to {diag_path}")


# ============================================================================
# CHECKPOINT WRITER
# ============================================================================

def write_checkpoint(
    group_results: List[GroupResult],
    output_path: str,
) -> None:
    """Write partial results to a checkpoint JSON file."""
    checkpoint_path = output_path + ".checkpoint.json"
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
) -> None:
    """Run unified v5 evaluation for the specified dataset."""
    from config.logging_config import setup_logging, log_file_from_output_path

    # Resolve output path
    if output_path is None:
        output_path = f"{dataset}_output.md"
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
    _print(f"\nLoading {adapter.display_name} dataset...")
    groups = adapter.load_groups(limit=limit, small_corpus=small_corpus)

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
        checkpoint_path = output_path + ".checkpoint.json"
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
    parser = argparse.ArgumentParser(
        description="Unified v5 Evaluation Runner",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["scifact", "wikifactclaims"],
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
    )
