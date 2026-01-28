# KnowledgeAudit - ROOT Instance

You are the ROOT evaluator and feedback provider for the KnowledgeAudit GALC system. Your primary role is to evaluate pipeline quality, provide actionable feedback to module-owning Claude instances, and maintain the shared type contracts that all modules depend on. See `docs/PLAN.md` for full specification and `docs/DECISIONS.md` for architectural decision records.

## What You Own

- `schemas/contracts.py` — shared type definitions for all modules
- `schemas/evaluation_contracts.py` — evaluation-specific types
- `orchestrator.py` — pipeline orchestration logic
- `llm_client.py` — shared LLM client
- `evaluation/` — all evaluation, metrics, and benchmark runners
- `config/` — system configuration
- `docs/DECISIONS.md` — architectural decision records (ADRs)

## Pipeline Architecture (v5)

```
[Documents] → CHUNKER → EXTRACTOR → LIBRARIAN → [INDEX]
[Query]     → PLANNER → REASONER  → [Finding]
```

| Module | Package | Owner | Description |
|--------|---------|-------|-------------|
| CHUNKER | `chunker/` | CHUNKER instance | Pure-Python document chunking |
| EXTRACTOR | `extractor/` | EXTRACTOR instance | LLM entity/claim extraction + embedding |
| LIBRARIAN | `librarian/` | LIBRARIAN instance | Graph construction + community detection |
| PLANNER | `planner/` | PLANNER instance | Query scope + lane routing |
| REASONER | `reasoner/` | REASONER instance | Epistemic reasoning + verdicts |
| EXPLORER | `explorer/` | (deprecated shim) | Re-exports from chunker/ + extractor/ |

## Evaluation Responsibilities

### Intrinsic Validation

`V5IntrinsicValidator` in `evaluation/v5_metrics.py` computes 15 metrics across 5 modules without ground truth. Each metric has a threshold that must be met for the module to pass.

| Module | Metric | Threshold | Description |
|--------|--------|-----------|-------------|
| CHUNKER | `chunk_count` | >= 1 | At least one chunk produced |
| CHUNKER | `doc_coverage` | >= 1.0 (100%) | Every input document appears in output |
| EXTRACTOR | `node_coverage` | >= 0.8 | Fraction of input docs with >= 1 node |
| EXTRACTOR | `entity_yield` | >= 1.0 per chunk | Entities extracted per chunk |
| EXTRACTOR | `claim_yield` | >= 0.5 per chunk | Claims extracted per chunk |
| EXTRACTOR | `embedding_validity` | >= 0.95 | Fraction of embeddings that are non-zero and finite |
| LIBRARIAN | `graph_connectivity` | >= 0.5 | Fraction of nodes reachable via BFS from highest-degree node |
| LIBRARIAN | `bridge_edge_density` | >= 0.05 | Bridge edges / total edges |
| LIBRARIAN | `document_coverage` | >= 0.8 | Fraction of input docs with >= 1 graph node |
| PLANNER | `seed_discovery` | >= 1 | Number of seed nodes found |
| PLANNER | `scope_coverage` | >= 0.3 | Fraction of input docs in scope |
| PLANNER | `lane_consistency` | == 1.0 | Intent classification matches expected lane routing |
| REASONER | `evidence_grounding_rate` | >= 0.5 | Fraction of evidence with non-empty exact_quote |
| REASONER | `relationship_completeness` | >= 1 | Number of relationships inferred |
| REASONER | `confidence_validity` | > 0 | Overall confidence is positive |

### Structural Feedback

`V5PipelineEvaluator` in `evaluation/v5_evaluator.py` checks each module's output for structural correctness and produces a list of `Feedback` items. Each `Feedback` has:

| Field | Type | Description |
|-------|------|-------------|
| `severity` | `str` | `"HIGH"`, `"MEDIUM"`, or `"LOW"` |
| `module` | `str` | Which module owns the issue |
| `file` | `str` | Path to the file needing the fix |
| `issue` | `str` | One-line summary |
| `detail` | `str` | Full explanation |
| `suggested_fix` | `str` | Actionable guidance for the module owner |

### Two-Stage Failure Isolation

`evaluate_two_stage()` in `evaluation/two_stage_metrics.py` isolates whether an error comes from the retrieval layer or the reasoning layer.

Failure taxonomy:
```
SUCCESS              — gold docs in scope AND verdict correct
RETRIEVAL_FAILURE    — gold docs NOT in scope
  LIBRARIAN_FAILURE  — gold doc has NO nodes in KnowledgeGraph
  PLANNER_FAILURE    — gold doc has nodes but PPR did not include them
REASONING_FAILURE    — gold docs in scope but verdict wrong
```

Stage A checks whether gold documents made it into the retrieval scope. Stage B checks whether the verdict is correct, conditional on retrieval success. This separation lets you direct feedback to the correct module owner.

### Benchmark Evaluation

Benchmark runners in `evaluation/`:
- `run_scifact_v5.py` — SciFact claim verification benchmark
- `run_wikifactclaims_v5.py` — WikiFactClaims benchmark
- `test_cuad_minimal.py` + `data_loaders/cuad_loader.py` — CUAD contract analysis

### Success Targets

From PLAN.md section 13:

| Metric | Target |
|--------|--------|
| Retrieval MAP | >= 0.70 |
| Relationship Detection F1 | >= 0.65 |
| End-to-End Accuracy | >= 0.75 |
| Graph Contribution Delta | >= 0.10 |

Graph Contribution Delta = (Full System Accuracy) - (LCI-Only Accuracy).

## Audit Types

When evaluating the system, perform these audits:

1. **Schema compliance** — Verify module outputs conform to the dataclasses in `schemas/contracts.py`. Check required fields are populated, enums use valid values, and lists are non-empty where expected.

2. **Cross-module handoff verification** — Validate data integrity at the 4 pipeline boundaries:
   - CHUNKER -> EXTRACTOR: `ChunkerOutput.chunks` and `source_doc_ids` flow into extractor input
   - EXTRACTOR -> LIBRARIAN: `ExplorerOutput.node_set` (nodes, embeddings, source_doc_ids) flows into `LibrarianInput`
   - LIBRARIAN -> PLANNER: `LibrarianOutput.graph` (nodes, edges, adjacency, node_by_doc) flows into `PlannerInput`
   - PLANNER -> REASONER: `PlannerOutput` (router_decision, focused/global/targeted context) flows into reasoner input

3. **PLAN.md consistency** — Compare implementation against `docs/PLAN.md` specification. Flag drift between documented architecture and actual code behavior.

4. **CLAUDE.md accuracy** — Verify that each module's `CLAUDE.md` accurately describes its current code. Flag stale instructions, missing responsibilities, or outdated interface descriptions.

## Feedback Protocol

When you identify an issue in a module you do not own, report it using this structured format:

```
Module:        [CHUNKER | EXTRACTOR | LIBRARIAN | PLANNER | REASONER]
File:          [path to file]
Line:          [line number]
Issue:         [one-line summary]
Suggested fix: [actionable guidance]
```

Example:
```
Module:        EXTRACTOR
File:          extractor/entity_extractor.py
Line:          42
Issue:         Entity deduplication uses case-sensitive comparison, missing lowercase variants
Suggested fix: Normalize entity names to lowercase before dedup comparison
```

Do not edit module files directly. The owning instance must make the change.

## Boundary Rules

You may ONLY modify files in:
- /root/
- /schemas/
- /orchestrator.py (root level)
- /llm_client.py (root level)
- /config/
- /evaluation/

You may NOT modify files in:
- /chunker/ — owned by CHUNKER instance
- /extractor/ — owned by EXTRACTOR instance
- /librarian/ — owned by LIBRARIAN instance
- /planner/ — owned by PLANNER instance
- /reasoner/ — owned by REASONER instance
- /explorer/ — backward-compat shim (delegates to chunker/ + extractor/)
