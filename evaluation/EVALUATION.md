# Evaluation System

This document describes the v5 evaluation framework for KnowledgeAudit. The framework evaluates the Graph-LCI pipeline against two external benchmarks (SciFact and WikiFactClaims) and provides intrinsic validation, structural feedback, and ground-truth fixture tests.

## Benchmarks

### SciFact

[SciFact](https://github.com/allenai/scifact) (AllenAI) is a benchmark of scientific claims paired with evidence abstracts. `SciFactLoader` loads from local JSONL files (with HuggingFace fallback) and yields:

- **Corpus:** ~5,000 PubMed abstracts (indexed once into a single knowledge graph)
- **Claims:** Validation-split claims, each labeled with a verdict and zero or more evidence documents (identified by doc ID + sentence indices)

Entry point: `python -m evaluation.run_evaluation_v5 --dataset scifact`

### WikiFactClaims

WikiFactClaims is a multi-hop reasoning benchmark derived from HotpotQA. Each row contains a question, 10 context passages, supporting-fact annotations, and 3 derived claims. `WikiFactClaimsLoader` loads from a local JSONL file and yields:

- **Per-row corpus:** 10 Wikipedia passages (indexed independently per row)
- **Claims:** 3 claims per row, each labeled SUPPORTS / REFUTES / NOT_ENOUGH_INFO
- **Metadata:** Question type (comparison / bridge) and difficulty level (medium / hard)

The key structural difference from SciFact: context passages are inline per-row, so the knowledge graph is built per-row rather than against a global corpus.

Entry point: `python -m evaluation.run_evaluation_v5 --dataset wikifactclaims`

### Shared Labels

Both benchmarks use the same three-way label scheme:

| Label | Meaning |
|---|---|
| `SUPPORTS` | Evidence confirms the claim |
| `REFUTES` | Evidence contradicts the claim |
| `NOT_ENOUGH_INFO` | No evidence in corpus supports or refutes |

## Pipeline Under Test

Evaluation exercises the full v5 pipeline:

### Phase 1: Index (once per corpus / once per row)

```
Documents --> CHUNKER --> EXTRACTOR --> LIBRARIAN --> Knowledge Graph
```

For SciFact, all ~5K abstracts (or a subset with `--small-corpus`) are indexed into a single knowledge graph. For WikiFactClaims, the 10 context passages per row are indexed into an independent graph per row.

### Phase 2: Query (per-claim)

```
Claim text --> PLANNER --> REASONER --> Verdict + Evidence
```

Each labeled claim is submitted as a query. The planner selects seed nodes, determines scope via PPR expansion, and routes to a lane. The reasoner produces a verdict, confidence score, grounded evidence, and epistemic relationships.

## Verdict Mapping

The v5 pipeline uses six fine-grained verdict types. These are collapsed to the three benchmark labels for scoring:

| v5 VerdictType | Benchmark Label |
|---|---|
| `CONFIRMED` | SUPPORTS |
| `PARTIALLY_CONFIRMED` | SUPPORTS |
| `REFUTED` | REFUTES |
| `PARTIALLY_REFUTED` | REFUTES |
| `INCONCLUSIVE` | NOT_ENOUGH_INFO |
| `NOT_FOUND` | NOT_ENOUGH_INFO |

A claim is **correct** when its mapped verdict matches the expected label.

## Metrics

### Core Metrics (both benchmarks)

| Metric | Definition |
|---|---|
| **Verdict Accuracy** | Fraction of completed claims where mapped verdict == expected label |
| **Mean Confidence** | Average of the reasoner's confidence score (0–1) across completed claims |

### SciFact-Specific

| Metric | Definition |
|---|---|
| **Evidence Doc Recall** | Per-claim: \|predicted evidence doc IDs ∩ expected doc IDs\| / \|expected doc IDs\|. Averaged across claims. Claims with no expected evidence get recall = 1.0 |

### WikiFactClaims-Specific

| Metric | Definition |
|---|---|
| **Planner Scope Recall** | Per-claim: \|supporting-fact titles in planner scope\| / \|total supporting-fact titles\|. Averaged across claims |
| **Per-Type Accuracy** | Verdict accuracy broken down by question type (comparison / bridge) |
| **Per-Level Accuracy** | Verdict accuracy broken down by difficulty level (medium / hard) |

### Per-Verdict Breakdown

For each of the three labels, both benchmarks report count, correct count, and accuracy among claims with that expected label.

### Confusion Matrix

A 3×3 matrix (rows = expected, columns = predicted) counting how predictions distribute across labels.

## Two-Stage Failure Analysis (SciFact)

For SciFact claims that have expected evidence documents, failures are isolated into two stages. This answers: *when the system gets a claim wrong, is it because retrieval failed or because reasoning failed?*

### Stage A: Retrieval Audit

For each expected evidence doc ID, check:

1. **In graph?** Does the librarian's knowledge graph contain nodes sourced from this document?
2. **In scope?** Did the planner's scope include this document?

Each expected doc gets an attribution:

| Status | Meaning |
|---|---|
| `NOT_APPLICABLE` | Doc is in scope (retrieval succeeded for this doc) |
| `LIBRARIAN_FAILURE` | Doc has no nodes in the graph at all |
| `PLANNER_FAILURE` | Doc has nodes in the graph but planner excluded it from scope |

**Graph Recall@K** = (gold docs in scope) / (total gold docs). If all gold docs are in scope, Stage A passes.

### Stage B: Reasoning Audit

Only evaluated when Stage A passes (all expected evidence was retrieved). Checks whether the reasoner produced the correct verdict given that it had access to the right documents.

### Failure Classification

Each claim with expected evidence is classified as:

| Category | Condition |
|---|---|
| `SUCCESS` | Stage A passed and verdict is correct |
| `RETRIEVAL_FAILURE` | At least one expected evidence doc was not in scope |
| `REASONING_FAILURE` | All evidence was in scope but verdict is wrong |

Aggregate rates (success rate, retrieval failure rate, reasoning failure rate) and sub-attributions (librarian vs. planner failure counts) are reported.

### Per-Evidence-Doc Attribution

Each expected evidence document across all claims also gets a finer status:

| Status | Meaning |
|---|---|
| `GROUNDED` | Doc is in scope and the reasoner cited it as evidence |
| `IN_SCOPE_NOT_GROUNDED` | Doc is in scope but the reasoner did not cite it |
| `IN_GRAPH_NOT_IN_SCOPE` | Doc has graph nodes but the planner excluded it |
| `ABSENT_FROM_GRAPH` | Doc has no nodes in the graph |

## Intrinsic Validation

`V5IntrinsicValidator` computes quantitative metrics from pipeline outputs without ground truth. Given a `V5PipelineSnapshot` (all intermediate module outputs from one pipeline run), it checks 13+ metrics across the modules:

**Chunker:** chunk count, document coverage, token statistics.

**Extractor:** node coverage (fraction of input docs with at least one node), entity yield (entities per chunk), claim yield (claims per chunk), embedding validity (fraction of embeddings that are non-zero and finite).

**Librarian:** graph connectivity (fraction of nodes reachable via BFS from any node), bridge edge density (bridge edges / total edges), document coverage (fraction of input docs with at least one graph node).

**Planner:** seed discovery (seed nodes found), scope coverage (fraction of input docs in scope), lane consistency (intent matches expected lane routing).

**Reasoner:** evidence grounding rate (fraction of evidence with non-empty exact quote), relationship completeness (relationships inferred), confidence validity (overall confidence > 0).

Each metric has a pass/fail threshold. The overall result passes only when all module metrics pass.

## Structural Feedback

`V5PipelineEvaluator` performs structural checks on each module's output and produces a list of `Feedback` items. Each item identifies severity (HIGH / MEDIUM / LOW), the responsible module, the relevant file, a description of the issue, and a suggested fix. This is used for development-time pipeline health checks, not benchmark scoring.

## Ground-Truth Fixtures

`v5_fixtures.py` defines labeled synthetic test cases (e.g., MSA liability cap vs. DORA unlimited liability) with expected verdicts, epistemic relationships, lane selections, and intents. These are used for regression testing: run the pipeline on known-correct inputs and verify the outputs match expectations.

## Per-Claim Trace Data (SciFact)

For every claim, `run_single_claim()` captures detailed trace data that feeds the trace JSON. This includes:

**Planner trace:** seed count, scope size (nodes and documents), selected lane, classified intent, router confidence, PPR mass, seed details (node ID, similarity, type, text, doc), scope node details, ranked chunks (Lane C).

**Reasoner trace:** verdict answer text, summary, reasoning trace, confidence factors (overall, evidence quality, source agreement, coverage), scope document IDs, bridge edges used, relationship details (source, target, type, confidence, reasoning text), evidence details (claim, source document, chunk, exact quote), timing (LLM time, total time).

**Two-stage results:** failure stage, retrieval attribution, graph recall@K, gold docs in scope/graph/total, planner scope doc IDs, planner scope recall, per-doc failure attribution.

The amount of detail captured depends on `--trace-level`:

| Level | What's included |
|---|---|
| `minimal` | Basic metrics only (verdict, confidence, evidence IDs) |
| `standard` | Rich detail: top-N seeds, scope nodes, relationships, evidence, confidence factors |
| `full` | Everything in standard plus raw LLM responses and uncapped lists |

## Output Files

### SciFact

| File | Content |
|---|---|
| `output.md` | Human-readable report: configuration, summary metrics, two-stage failure analysis, per-verdict breakdown, confusion matrix, per-claim results table, error log |
| `output.trace.json` | Machine-readable trace: config, index stats, aggregate metrics, confusion matrix, per-verdict accuracy, index diagnostics (chunker/extractor/librarian/coreference/cross-doc overlap), and per-claim traces |
| `output.md.checkpoint.json` | Partial results written every 10 claims for crash recovery. Deleted on successful completion |

### WikiFactClaims

| File | Content |
|---|---|
| `wiki_output.md` | Human-readable report: configuration, summary metrics, per-verdict breakdown, per-type accuracy, per-level accuracy, confusion matrix, per-row summary, per-claim results, error log |

## Corpus Modes (SciFact)

**Full corpus** (default): All ~5K SciFact abstracts are indexed. This tests the system's ability to find relevant documents in a large graph.

**Small corpus** (`--small-corpus`): Only `limit * 5` documents are indexed, always including each claim's evidence documents plus random distractors. This isolates reasoning quality from retrieval difficulty.

## Checkpointing (SciFact)

Every 10 claims, partial results are written to `output.md.checkpoint.json`. If the run is interrupted (Ctrl+C, network failure), the `finally` block still writes `output.md` and `output.trace.json` with whatever results were collected. The checkpoint file is deleted only after all claims complete.

## Index Diagnostics (SciFact)

After indexing, diagnostics are collected from each pipeline stage and logged to stdout. These metrics track pipeline health and are written to the trace JSON under `index_diagnostics`.

**Chunker:** total chunks, documents chunked, token stats (min/max/mean/median), section headers detected, chunks per doc.

**Extractor:** entities/claims extracted, gleaning iterations, LLM calls, extraction time, zero-entity/zero-claim chunk counts, mean entities/claims per chunk.

**Extractor (NodeSet):** chunk/entity/claim/node counts, source doc coverage, coreference groups, average claims per chunk.

**Librarian:** graph nodes/edges, document count, bridge edges (cross-document), community count, edge/node type distributions, document coverage (with/without nodes), coreference analysis (multi-doc groups, variant counts), cross-document entity overlap.

### Unified Runner

```
python -m evaluation.run_evaluation_v5 [options]

--dataset {scifact,wikifactclaims} (required)
--limit N              Maximum items to evaluate (default: all)
--output PATH          Output markdown path (default: {dataset}_output.md)
--model NAME           Ollama LLM model (default: OLLAMA_MODEL env)
--embed-model NAME     Ollama embedding model (default: OLLAMA_EMBED_MODEL env)
--mock                 Use mock clients for smoke testing
--trace-level LEVEL    minimal | standard | full (default: standard)

# SciFact only
--small-corpus         Index only limit*5 docs instead of full corpus
--trace-chunks N       Chunk details in trace JSON: 0=skip, -1=all (default: 50)

# WikiFactClaims only
--data-path PATH       Path to wikifactclaims.jsonl
```
