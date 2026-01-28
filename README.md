# KnowledgeAudit: Graph-Augmented Long-Context (GALC) Engine

## Project Structure

This project uses a multi-instance architecture with the following modules:

| Instance | Directory | Responsibility |
|----------|-----------|----------------|
| ROOT | `/` (root level) | Orchestration, evaluation, shared contracts, config |
| CHUNKER | `/chunker` | Pure-Python document chunking |
| EXTRACTOR | `/extractor` | LLM entity/claim extraction + embedding |
| LIBRARIAN | `/librarian` | Graph construction + community detection |
| PLANNER | `/planner` | Query scope + lane routing |
| REASONER | `/reasoner` | Epistemic reasoning + verdicts |
| EXPLORER | `/explorer` | Deprecated shim — delegates to CHUNKER + EXTRACTOR |

## v5 Pipeline

```
[Documents] → CHUNKER → EXTRACTOR → LIBRARIAN → [INDEX]
[Query]     → PLANNER → REASONER  → [Finding]
```

## Getting Started

1. ROOT instance owns `schemas/contracts.py` (shared type definitions)
2. CHUNKER splits documents into token-bounded chunks
3. EXTRACTOR extracts entities, claims, and embeddings from chunks
4. LIBRARIAN builds the knowledge graph with structural edges and communities
5. PLANNER resolves query scope via PPR and routes to a processing lane
6. REASONER performs epistemic reasoning and produces verdicts

See `docs/PLAN.md` for the full technical specification.
