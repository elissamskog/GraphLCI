"""
Minimal CUAD integration test.
Location: evaluation/test_cuad_minimal.py
Owner: ROOT Instance

Loads 2 CUAD contracts, indexes them, and runs a comparison query through
the full Orchestrator pipeline using a real Ollama LLM.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# planner/ directory for inner planner package resolution
_PLANNER_DIR = os.path.join(_PROJECT_ROOT, "planner")
if _PLANNER_DIR not in sys.path:
    sys.path.insert(0, _PLANNER_DIR)

from schemas.contracts import RawCorpus, QueryRequest
from orchestrator import Orchestrator
from llm_client import get_client
from evaluation.data_loaders.cuad_loader import CUADLoader


def main() -> None:
    print("=" * 70)
    print("CUAD Minimal Integration Test")
    print("=" * 70)

    # 1. Load 2 CUAD contracts
    print("\n--- Loading CUAD contracts ---")
    loader = CUADLoader()
    docs = loader.get_contracts(limit=2)
    print(f"Loaded {len(docs)} contracts:")
    for d in docs:
        print(f"  - {d.doc_id}  ({len(d.content)} chars)")

    # 2. Create Orchestrator with Ollama client
    llm = get_client("ollama")
    config = {
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
    orchestrator = Orchestrator(llm, config)

    # 3. Build RawCorpus and index
    corpus = RawCorpus(
        corpus_id="cuad-test-corpus",
        documents=docs,
        ingestion_timestamp=datetime.now(),
    )

    print("\n--- Indexing Corpus ---")
    indexed = orchestrator.index_corpus(corpus)
    node_count = len(indexed.graph_state.nodes)
    edge_count = len(indexed.graph_state.edges)
    print(f"Index complete: {node_count} nodes, {edge_count} edges")
    print(f"Coverage score: {indexed.extraction_metadata.coverage_score:.2f}")

    # 4. Run comparison query
    query = QueryRequest(
        query_id="cuad-q1",
        query_text="Do these contracts have conflicting liability terms?",
    )

    print("\n--- Processing Query ---")
    print(f'Query: "{query.query_text}"')
    router_decision, assembled_context = orchestrator.process_query(query)

    # 5. Print router decision
    print(f"\nRouterDecision:")
    print(f"  Lane:            {router_decision.selected_lane.value}")
    print(f"  Graph scope:     {router_decision.requires_graph_scope}")
    print(f"  Intent:          {router_decision.intent.classification.value} "
          f"(conf={router_decision.intent.confidence})")
    print(f"  Targets:         docs={router_decision.identified_targets.explicit_docs}, "
          f"entities={router_decision.identified_targets.implicit_entities}")
    print(f"  Domain:          {router_decision.domain_expectation.value}")

    # 6. Print assembled context if available
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
    print("CUAD minimal test complete. Review output above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
