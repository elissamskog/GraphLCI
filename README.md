# GraphLCI: graph-augmented claim verification

GraphLCI checks whether a claim is supported, contradicted or not covered by a set of documents, and returns a verdict
with the evidence and the source passages it relied on.

The hard part is that evidence is often spread across documents that use different words for the same thing. A
contract can cap liability at a fixed amount while the regulation that governs it requires unlimited liability. The two
texts share entities but almost no keywords, so plain vector search misses the connection. GraphLCI builds a knowledge
graph over the documents and follows those links to collect the evidence before an LLM reasons over it.

## How it works

```
Index time:  [Documents] → CHUNKER → EXTRACTOR → LIBRARIAN → knowledge graph + vector index
Query time:  [Claim]     → PLANNER → REASONER  → verdict + confidence + cited evidence
```

| Module | What it does |
|---|---|
| `chunker/` | Splits documents into overlapping, token-bounded passages |
| `extractor/` | LLM extraction of entities and claims, coreference resolution, embeddings for every node |
| `librarian/` | Builds the graph: co-occurrence, citation, shared-entity and synonym edges; community detection and community summaries |
| `planner/` | Finds seed nodes for the claim, expands scope with Personalized PageRank and routes the query to a reasoning lane |
| `reasoner/` | Identifies how the pieces of evidence relate (supports, contradicts, depends on), grounds every statement in a source passage and issues the verdict |
| `schemas/` | Shared typed contracts between modules, so each part can be developed and tested on its own |

**Three reasoning lanes.** The planner classifies the query and picks the evidence shape that fits it: full documents plus
the graph relationships between them for comparisons, community summaries for broad questions, and fast ranked
passages for simple look-ups. It falls back to vector-only retrieval when the graph scope is too thin.

**Verdicts.** Six levels (confirmed, partially confirmed, refuted, partially refuted, inconclusive, not found), each with a
confidence score and the passages it rests on.

## Evaluation

- **Benchmarks:** [SciFact](https://github.com/allenai/scifact) (scientific claims against ~5,000 PubMed abstracts),
  WikiFactClaims (multi-hop claims derived from HotpotQA), and a CUAD-based contract test.
- **Failure isolation:** a two-stage framework separates retrieval failures (the right documents were never surfaced)
  from reasoning failures (the right documents were there but the verdict was wrong).
- **Graph contribution delta:** measures how much accuracy the graph adds over vector-only retrieval, which is the core
  question the project tests.
- **Intrinsic checks:** structural health metrics (extraction coverage, graph connectivity, cross-document bridges)
  that need no ground truth.

Details: [`evaluation/EVALUATION.md`](evaluation/EVALUATION.md).

## Running it

Runs on local models through [Ollama](https://ollama.com) (default `qwen2.5:14b`, set with `OLLAMA_MODEL`).
Python 3, with `numpy`, `requests`, `datasets` and `pytest`.

```bash
# Full SciFact run on a small corpus
python -m evaluation.run_scifact_v5 --small-corpus --limit 50

# Same pipeline with a mock LLM, no model needed
python -m evaluation.run_scifact_v5 --mock --limit 10

# Multi-hop benchmark
python -m evaluation.run_wikifactclaims_v5
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): overview of the pipeline
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decisions and their trade-offs
- [`docs/PLAN.md`](docs/PLAN.md): full technical specification
- One document per module in `docs/`

Built by Elis Samskog.
