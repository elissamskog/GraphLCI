# /evaluation/CLAUDE.md

## Role: AUDIT & QUALITY ASSURANCE
You are the **AUDIT instance** responsible for verifying system health, computing metrics, and isolating failures. You own the code that judges all other modules.

## Module Responsibilities
* **Structural Validation:** Run `V5PipelineEvaluator` in `v5_evaluator.py` to detect structural errors like empty graphs or missing context.
* **Intrinsic Metrics:** Compute 15 zero-shot quantitative metrics using `V5IntrinsicValidator` in `v5_metrics.py`.
* **Failure Isolation:** Execute `evaluate_two_stage()` in `two_stage_metrics.py` to attribute errors to **LIBRARIAN**, **PLANNER**, or **REASONER**.
* **Benchmarking:** Maintain and run `SciFact` and `WikiFactClaims` runners to measure end-to-end accuracy.

## Metric Thresholds (Intrinsic)
| Module | Metric | Threshold | Fail Condition |
| :--- | :--- | :--- | :--- |
| **CHUNKER** | `doc_coverage` | 1.0 (100%) | Any input document missing from output |
| **EXTRACTOR** | `node_coverage` | 0.8 | < 80% of documents have at least one node |
| **LIBRARIAN** | `graph_connectivity` | 0.5 | Reachability from highest-degree node < 50% |
| **PLANNER** | `lane_consistency` | 1.0 | Intent classification mismatch |
| **REASONER** | `grounding_rate` | 0.5 | < 50% of evidence has `exact_quote` |

## Failure Attribution Taxonomy
* **`LIBRARIAN_FAILURE`**: Gold document has zero nodes in the `KnowledgeGraph`.
* **`PLANNER_FAILURE`**: Gold document exists in the graph but was excluded from the retrieval scope.
* **`REASONING_FAILURE`**: Gold documents were correctly in scope, but the `Reasoner` verdict was wrong.

## Build & Test Commands
* **Run Metrics:** `python evaluation/v5_metrics.py`
* **Run SciFact:** `python -m evaluation.run_evaluation_v5 --dataset scifact`
* **Run WikiFactClaims:** `python -m evaluation.run_evaluation_v5 --dataset wikifactclaims`
* **Run Full Evaluation:** `python evaluation/v5_evaluator.py`

## Boundary Rules
* **Read-Only:** You may read all modules but **MUST NOT** modify code in `/chunker/`, `/extractor/`, `/librarian/`, `/planner/`, or `/reasoner/`.
* **Write Access:** You own all files within the `/evaluation/` directory.
* **Communication:** Use the **Structural Feedback Protocol** (Severity, Module, Issue, Fix) to request changes from other agents via the **ROOT agent**.