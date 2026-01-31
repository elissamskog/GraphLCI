# RAG LCI: Graph Architecture & Evaluation

Role: You are a Lead ML Engineer. 
Process: > 
1. Analyze: Look at the failures in the latest musique_output.trace. Pinpoint exactly why it failed or succeeded.
    i. Use debug_graph_v5.py --check-path for every failed multi-hop claim. If Path Found but Mass = 0, it is a PLANNER weighting/alpha failure. If No Path Found, it is an EXTRACTION bridge failure.
2. Hypothesize: Formulate a hypothesis (e.g., "Increasing the chunk overlap by 15% will capture the co-references currently being lost").
3. Record: Write these insights into a new section in optimization_log.md called ## Analysis & Hypotheses.
4. Test: Execute the code change based on your hypothesis and run the eval again.
5. Compare.

## Evaluation data

Run a single random claim from the MuSiQue dataset with a small corpus:

```bash
python -m evaluation.run_evaluation_v5 --dataset musique --limit 1
```

## Graph Architecture

### 1. Entities
*   **Definition**: Unique nodes representing discrete real-world referents (e.g., *Elias Thorne*, *University of Oxford*).
*   **Function**: Act as anchor points. They serve as the primary targets for the initial semantic jump and the starting seeds for graph walks.

### 2. Claims
*   **Definition**: Directed semantic edges connecting two entities via a predicate (Subject-Predicate-Object triples).
*   **Function**: Carry the logical weight of the graph. They provide the specific evidence strings used by the Reasoner to determine a verdict.

### 3. Structured Bridges
*   **Definition**: Deterministic edges created via regex or dependency parsing to link entities based on syntactic patterns (e.g., *"X of Y"*).
*   **Function**: Repair "structural blindness." They ensure connectivity between leaf nodes (spokes) and institutional hubs (e.g., linking a college to its parent university) without requiring LLM inference.

### 4. Communities
*   **Definition**: High-density subgraphs formed by entities and claims extracted from the same text chunk.
*   **Function**: Preserve local coherence. They allow the search algorithm to "activate" an entire paragraph's context once a single entity in that chunk is touched.

### 5. Bridges
*   **Definition**: Edges that span across different communities, including synonymy links and structural bridges.
*   **Function**: Facilitate multi-hop traversal. They allow probability mass to move from one document cluster to another, preventing the search from being trapped in a single local neighborhood.

---

## Debugging & Auditing Tools

### Global Graph Audit
Evaluates global Community health and detects "Star Topology" collapses.
```bash
python debug_graph_v5.py --global-metrics
```
*   **Relationship Entropy**: Measures Claim predicate diversity. Low scores indicate generic, low-signal extraction.
*   **Fact Density**: Maps Entity concentration across the corpus.
*   **Clustering Coefficient**: Quantifies the "tightness" of local Communities.

### Node Saliency Audit
Diagnoses if an Entity is a "Broadcaster" or a "Semantic Sink."
```bash
python debug_graph_v5.py --describe-node "Node Name"
```
*   **PPR Specificity Index**: Identifies Entities dominating the search mass.
*   **Document Lineage**: Tracks which sources contributed to the node's construction.
*   **Top Neighbors**: Displays the most influential adjacent Entities and Claims.

### Bridge Verifier
Confirms the presence of Bridges or Structured Bridges between two points.
```bash
python debug_graph_v5.py --check-path "Source" "Target"
```
*   **Hop Count**: Calculates the path length between nodes.
*   **Function**: Determines if the failure is in Extractor connectivity (no bridge exists) or Planner navigation (bridge exists but mass failed to flow).

### Damping Scrubber
Tests the reach of the "Search Energy" across the graph's Bridges.
```bash
python debug_graph_v5.py --sim-decay 0.65 4
```
*   **Simulated Decay**: Calculates $P = \alpha^L$ to check if the current $\alpha$ allows mass to reach evidence $L$ hops away.

### Top Hubs Analysis
Identifies "Noise Hubs" that disrupt Community boundaries.
```bash
python debug_graph_v5.py --audit
```
*   **Sink Detection**: Lists top nodes by degree and PageRank.
*   **Function**: Detects generic Entities (e.g., "2026") that have erroneously become global hubs.
## Graph Construction Pitfalls

### 1. Schema Brittleness (Missing Attributes)
*   **Problem**: Generic OpenIE extraction often misses type-specific attributes needed for multi-hop reasoning.
*   **Example**: "ITV biographical drama" -> Extractor sees "drama" (Concept) but misses "ITV" (Network) because it appears as an adjective.
*   **Result**: The graph contains the entity "Victoria" (TV Series) but lacks the edge to its network, causing a **Topological Disconnect**.
*   **Fix**: Implement **Schema-Guided Extraction** for high-value types (e.g., if type=TV_SERIES, explicitly query for "Network", "Creator").

### 2. Dependency Blindness
*   **Problem**: Structural analyzers (like spaCy) can link *existing* entities but cannot *create* new ones.
*   **Example**: If `amod(drama, ITV)` exists, but "ITV" was not named in the initial extraction pass, the structural bridge is never built.
*   **Fix**: **Dependency Promotion**. Promote un-extracted dependency modifiers to full Nodes if they are Proper Nouns (PROPN).

### 3. Semantic Over-Atomization
*   **Problem**: Breaking valid compound entities into meaningless parts.
*   **Example**: "The Treaty of Paris" -> "Treaty" (node) + "Paris" (node).
*   **Result**: Reasoning requiring the specific event "Treaty of Paris" fails because the unique concept is lost.
*   **Fix**: Enforce **Noun Chunk Preservation** for known named entities.
