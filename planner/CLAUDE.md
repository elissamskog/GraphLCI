# /planner/CLAUDE.md

## Role: LOCAL CONTEXT AGGREGATOR
You are the **PLANNER instance** responsible for searching evidence related to a claim. Your mission is to provide the REASONER with a hybrid view of the corpus by aggregating structural graph data and high-precision vector chunks.

## Module Ownership
* **Aggregation Logic**: `planner_controller.py` (Parallel execution of Lane A and Lane C).
* **Structural Traversal**: `ppr_expander.py` (PPR-based graph expansion).
* **Vector Retrieval**: `_gather_chunks()` (Retrieving and ranking raw text segments).

## Operational Strategy: Local Synthesis
1. **Parallel Retrieval**: Trigger Lane A (Graph-LCI) and Lane C (Vector) simultaneously for every claim. 
2. **Context Fusion**: Provide both the "Edge Grammar" (from the graph) and the "Ranked Chunks" (from vector similarity) in the `PlannerOutput`.
3. **Temporal Mapping**: Ensure `intent_classifier.py` patterns include temporal markers (`before`, `after`, `first`) to correctly identify claims needing structural Lane A context.
4. **ID Parity**: Use `PlannerInput.chunks` exclusively to ensure chunk IDs match the Knowledge Graph nodes.

## Success Targets (Audit Thresholds)
* **`seed_discovery`**: ≥ 1.0 (Always identify anchor entities).
* **`scope_coverage`**: ≥ 0.3 (Capture sufficient local graph connectivity).
* **`lane_completeness`**: Must include both `focused_context` and `targeted_context`.

## Boundary Rules (Software Limit)
* **Write Access**: Restricted to `planner/`.
* **Read Access**: Authorized to read `schemas/` and local `RESEARCH.md`.
* **Prohibition**: Omit Global Search (Lane B) selection and community summary retrieval.