"""
Dataset Adapters for v5 Evaluation
Location: evaluation/adapters.py
Owner: Audit Agent

Provides dataset-specific loading and feature flags for evaluation runners.
Extracted from run_evaluation_v5.py for modularity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from schemas.evaluation_contracts import EvalClaim, EvalGroup

# Default printer (unbuffered)
def _default_print(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True so output appears immediately in pipes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


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
    def load_groups(
        self,
        limit: Optional[int] = None,
        printer: Callable[..., None] = _default_print,
        **kwargs,
    ) -> List[EvalGroup]: ...

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
        printer: Callable[..., None] = _default_print,
        small_corpus: bool = False,
        **kw,
    ) -> List[EvalGroup]:
        from evaluation.data_loaders.scifact_loader import SciFactLoader

        loader = SciFactLoader()
        test_cases = loader.get_test_cases(limit=limit)
        printer(f"  Claims: {len(test_cases)} test cases" +
                (f" (limited to {limit})" if limit else ""))

        if small_corpus and limit:
            target = limit * 1  # Minimal corpus size (just evidence docs)
            documents = build_small_corpus(loader, test_cases, target)
            printer(f"  Small corpus: {len(documents)} documents "
                    f"(target: {target}, evidence docs included)")
        else:
            if small_corpus and limit is None:
                printer("WARNING: --small-corpus requires --limit N. "
                        "Falling back to full corpus.")
            documents = loader.get_all_abstracts()

        printer(f"  Corpus: {len(documents)} documents")

        # Log evidence coverage in small-corpus mode
        evidence_ids: set = set()
        for case in test_cases:
            evidence_ids.update(case.expected_evidence.keys())
        if small_corpus and limit is not None:
            included_ids = {d.doc_id for d in documents}
            missing = evidence_ids - included_ids
            if missing:
                printer(f"  WARNING: {len(missing)} evidence docs missing "
                        f"from corpus: {missing}")
            else:
                printer(f"  All {len(evidence_ids)} evidence docs present "
                        "in corpus")

        claims = [
            EvalClaim(
                claim_id=str(c.claim_id),
                claim_text=c.claim_text,
                expected_verdict=c.expected_verdict,
                gold_doc_ids=c.cited_doc_ids if c.cited_doc_ids else list(c.expected_evidence.keys()),
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
        printer: Callable[..., None] = _default_print,
        **kw,
    ) -> List[EvalGroup]:
        from evaluation.data_loaders.wikifactclaims_loader import WikiFactClaimsLoader

        loader = WikiFactClaimsLoader(path=self._data_path)
        rows = loader.get_rows(limit=limit)
        printer(f"  Rows: {len(rows)}" +
                (f" (limited to {limit})" if limit else ""))
        total_claims = sum(len(r.claims) for r in rows)
        printer(f"  Total claims: {total_claims}")

        groups: List[EvalGroup] = []
        merge_on = kw.get("merge_groups", False)

        # Containers for merging
        merged_documents: Dict[str, Any] = {}
        merged_claims: List[EvalClaim] = []
        merged_questions: List[str] = []

        for row in rows:
            documents = WikiFactClaimsLoader.build_corpus_for_row(row)
            claims = [
                EvalClaim(
                    claim_id=f"wiki-{row.row_id}-c{i}",
                    claim_text=c.text,
                    expected_verdict=c.label,
                    gold_doc_ids=list(set(row.supporting_facts_titles)),
                    expected_evidence={t: [] for t in set(row.supporting_facts_titles)},
                    metadata={"claim_index": i, "original_group": row.row_id},
                )
                for i, c in enumerate(row.claims)
            ]

            if merge_on:
                # Merge documents (deduplicate by doc_id which is title)
                for d in documents:
                    if d.doc_id not in merged_documents:
                        merged_documents[d.doc_id] = d
                # Merge claims
                merged_claims.extend(claims)
                merged_questions.append(row.question)
            else:
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

        if merge_on and merged_claims:
            printer(f"  Merged Corpus: {len(merged_documents)} documents")
            groups.append(EvalGroup(
                group_id="merged-wiki-group",
                claims=merged_claims,
                documents=list(merged_documents.values()),
                metadata={
                    "questions_merged": len(merged_questions),
                    "context": "merged_multi_group",
                },
            ))

        return groups


class MultiHopAdapter(DatasetAdapter):
    """Adapter for the MultiHop reasoning test dataset."""

    name = "multihop"
    display_name = "MultiHop Reasoning Test"
    has_per_type_level = True
    has_evidence_doc_recall = True

    def __init__(self, data_path: Optional[str] = None):
        self._data_path = data_path

    def load_groups(
        self,
        limit: Optional[int] = None,
        printer: Callable[..., None] = _default_print,
        **kw,
    ) -> List[EvalGroup]:
        from evaluation.data_loaders.multihop_loader import MultiHopLoader

        loader = MultiHopLoader(path=self._data_path)
        rows = loader.get_rows(limit=limit)
        printer(f"  Rows: {len(rows)}" +
                (f" (limited to {limit})" if limit else ""))
        total_claims = sum(len(r.claims) for r in rows)
        printer(f"  Total claims: {total_claims}")

        groups: List[EvalGroup] = []
        for row in rows:
            documents = MultiHopLoader.build_corpus_for_row(row)
            claims = [
                EvalClaim(
                    claim_id=f"multihop-{row.row_id}-c{i}",
                    claim_text=c.text,
                    expected_verdict=c.label,
                    gold_doc_ids=list(set(row.supporting_facts_titles)),
                    expected_evidence={t: [] for t in set(row.supporting_facts_titles)},
                    metadata={
                        "claim_index": i,
                        "reasoning_test": c.reasoning_test,
                        "required_hops": c.required_hops,
                    },
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


class MusiqueAdapter(DatasetAdapter):
    """Adapter for the MuSiQue multi-hop dataset."""

    name = "musique"
    display_name = "MuSiQue Multi-Hop QA"
    has_evidence_doc_recall = True  # We have cited_doc_ids

    def load_groups(
        self,
        limit: Optional[int] = None,
        printer: Callable[..., None] = _default_print,
        **kw,
    ) -> List[EvalGroup]:
        from evaluation.data_loaders.musique_loader import load_musique_data

        test_cases, corpus_docs = load_musique_data(limit=limit)
        printer(f"  Cases: {len(test_cases)}" +
                (f" (limited to {limit})" if limit else ""))

        claims = [
            EvalClaim(
                claim_id=str(c.claim_id),
                claim_text=c.claim_text,
                expected_verdict=c.expected_verdict,
                gold_doc_ids=c.cited_doc_ids,
                metadata=c.metadata,
                expected_evidence={},  # We don't have sentence-level
            )
            for c in test_cases.values()
        ]

        return [EvalGroup(
            group_id="musique",
            claims=claims,
            documents=list(corpus_docs.values()),
            metadata={"source": "dgslibisey/MuSiQue"}
        )]


# ============================================================================
# ADAPTER REGISTRY
# ============================================================================

_ADAPTER_REGISTRY: Dict[str, type] = {
    "scifact": SciFactAdapter,
    "wikifactclaims": WikiFactClaimsAdapter,
    "multihop": MultiHopAdapter,
    "musique": MusiqueAdapter,
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
    if dataset in ("wikifactclaims", "multihop"):
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
        # Also include cited_doc_ids to ensure we have the target docs for NEI claims
        if hasattr(case, "cited_doc_ids"):
            evidence_ids.update(case.cited_doc_ids)

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
