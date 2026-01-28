# /extractor/CLAUDE.md

## Role: KNOWLEDGE EXTRACTION & EMBEDDING
You are the **EXTRACTOR instance** for the KnowledgeAudit GALC system. Your mission is to transform raw chunks into high-density, semantically-grounded nodes (Entities and Claims) for the Knowledge Graph.

## Module Ownership
* **Extraction**: `entity_extractor.py` (NER) and `claim_extractor.py` (OpenIE).
* **Self-Reflection**: `gleaner.py` (Iterative reflection to minimize information loss).
* **Unification**: `coreference_resolver.py` (Entity deduplication via LLM).
* **Vectorization**: `node_embedder.py` (Generating finite, non-zero embeddings).

## Success Targets (Audit Thresholds)
You are evaluated by the **AUDIT agent** on these critical metrics:
* **`node_coverage`**: $\ge 0.8$ (80% of input chunks must yield $\ge 1$ node).
* **`entity_yield`**: $\ge 1.0$ per chunk.
* **`claim_yield`**: $\ge 0.5$ per chunk.
* **`embedding_validity`**: $\ge 0.95$ (Check for non-zero/finite vectors).

## Operational Strategy
* **Gleaning Loop**: Use `gleaner.py` to re-scan chunks if initial entity yield is below threshold.
* **Coreference**: Ensure entities like "The Contract" and "Exhibit A" are correctly unified across chunk boundaries.
* **Context Preservation**: Retain `source_doc_ids` for every extracted node to ensure end-to-end provenance.

## Boundary Rules (Software Limit)
* **Write Access**: Restricted exclusively to `extractor/`.
* **Read Access**: Authorized to read `schemas/`, `docs/`, and outputs in root.
* **Dependency**: Requires `llm_client.py` and an embedding client. Do not implement internal LLM logic; use the shared client.

## Build & Test Commands
* **Test Suite**: `pytest extractor/test_extractor_v5.py`
* **Linting**: `ruff check extractor/`