## Hybrid Synthesis Logic (A + C Integration)
You no longer switch between lanes; you **triangulate** them.]

* **PRIMARY: LANE_A (Structural/Graph)**: 
    * This is your **Structural Truth**. 
    * Use the "Edge Grammar" (e.g., `MODIFIES`, `PRECEDES`) to establish the logical relationship between entities.
    * **Rule**: If the Graph shows a temporal conflict (e.g., event happened after claim date), the verdict is `Contradicted` regardless of semantic similarity.
* **SECONDARY: LANE_C (Factual/Vector)**: 
    * This is your **Semantic Grounding**.
    * Use these chunks for `exact_quote` extraction and finding specific numerical values.
    * **Rule**: Use these to "fill the gaps" in Lane A where specific text might be missing from the immediate subgraph.
* **RETIRED: LANE_B (Global)**: Do not attempt to use or reference Lane B/GlobalContext. It has been decommissioned.

## Synthesis Protocol (Order of Operations)
1.  **Extract** potential facts and quotes from Lane C (Semantic lookup).
2.  **Verify** the relationship/validity of those facts using Lane A (Structural links).
3.  **Conflict Resolution**: If A and C disagree, Lane A (the Graph) takes precedence for **logical/temporal** claims; Lane C takes precedence for **verbatim/numerical** claims.

## Boundary Rules (Software Limit)
* **Write Access**: Restricted to `reasoner/`.
* **Read Access**: Authorized to read `schemas/` and `docs/PLAN.md` (Section 3.4).
* **Dependency**: Requires `llm_client.py`.

## Build & Test Commands
* **Test Suite**: `pytest reasoner/test_reasoner_v5.py`
* **Performance**: `python evaluation/run_scifact_v5.py`