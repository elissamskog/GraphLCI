# GALC Architecture Overview

## What Problem Does This Solve?

GALC (Graph-Augmented LCI) verifies claims against a corpus of documents. The core challenge is that evidence is often scattered across documents using different vocabulary and framing. A contract may cap liability at a fixed amount while a governing regulation requires unlimited obligations — detecting that tension requires connecting information across document boundaries, not just finding similar words.

## How the Pipeline Works

The system operates in two phases:

- **Index time** — Documents are ingested, broken into passages, analyzed for entities and claims, and woven into a knowledge graph.
- **Query time** — A claim is matched to the graph, expanded via traversal to gather structurally connected evidence, and routed to a reasoning engine that produces a verdict — supported, contradicted, or uncertain — backed by evidence and provenance.

## Five Components

| Component | Role | Analogy |
|-----------|------|---------|
| Chunker | Split documents into overlapping passages | Scribe cutting a manuscript into labeled pages |
| Extractor | Extract entities and claims, embed as vectors | Research assistant highlighting key facts |
| Librarian | Build knowledge graph, connect entities, detect communities | Indexer building a cross-referencing card catalog |
| Planner | Find seed nodes, expand scope via graph traversal, route by intent | Reference librarian pulling the right sources |
| Reasoner | Weigh evidence, identify epistemic relationships, issue grounded verdict | Judge ruling with a written opinion |

- **Chunker** splits documents into overlapping passages for downstream analysis. Think of a scribe cutting a manuscript into labeled pages.

- **Extractor** reads each passage, pulls out entities and claims, and embeds every node as a vector. Think of a research assistant highlighting key facts.

- **Librarian** connects entities across documents and detects communities. Think of an indexer building a cross-referencing card catalog.

- **Planner** finds seed nodes, expands scope via graph traversal, and routes to a reasoning lane. Think of a reference librarian pulling the right sources.

- **Reasoner** identifies epistemic relationships and produces a grounded verdict with provenance. Think of a judge issuing a ruling with a written opinion.

## The Key Insight

- **Graph vs. vector** — Vector retrieval finds documents with similar words; graph-augmented retrieval finds documents that are *structurally connected*. A contract and its governing regulation may use entirely different language but share entities that link them in the graph.
- **Edge types** — Edges encode real relationships: co-occurrence, citation, shared entities, and semantic synonymy. Traversal discovers relevant context that keyword or vector similarity alone would miss.
- **Edge grammar** — The graph provides structural scaffolding that primes the reasoner to look for specific tensions and dependencies rather than reasoning from a flat bag of passages.

## How We Evaluate

- **Intrinsic metrics** — Fifteen metrics across all components verify structural health without requiring ground truth: are entities being extracted? Is the graph connected? Are cross-document bridges forming?
- **Failure isolation** — A two-stage framework separates retrieval failures (graph didn't surface the right documents) from reasoning failures (right documents were there but verdict was wrong), directing feedback to the responsible component.
- **Benchmarks** — Suites covering scientific claims, factual claims, and contract analysis measure end-to-end accuracy. A dedicated "graph contribution delta" metric quantifies the value the graph adds beyond vector-only retrieval.
