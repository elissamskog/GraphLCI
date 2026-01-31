# Graph Debugging for Agents

This document outlines the non-visual programmatic utilities available in `debug_graph_v5.py` for AI agent consumption.

## Global Metrics
Identify if the graph is structurally balanced or has collapsed into a "Star Topology" centered on a few hubs.

```bash
python debug_graph_v5.py --global-metrics
```
- **Relationship Entropy**: Shannon entropy of the predicate distribution. Low entropy suggests overuse of generic predicates.
- **Fact Density**: Average nodes extracted per document.
- **Clustering Coefficient**: Measures the "tightness" of local neighborhoods.

## Node Saliency Audit
Deep-dive into a specific node to identify if it is a "Broadcaster" or a "Semantic Sink."

```bash
python debug_graph_v5.py --describe-node "Node Name or ID" [--describe-node-claim "Claim ID"]
```
- **PPR Specificity Index**: Z-score comparing node mass to the graph average.
- **Document Lineage**: Lists all documents contributing to this node.
- **Top Neighbors**: Highest-weight adjacent entities.

## Bridge Verifier
Verify if the extraction phase successfully created a path between two entities.

```bash
python debug_graph_v5.py --check-path "Source Entity" "Target Entity"
```
- Returns the shortest path and hop count.
- **Use Case**: If a path exists but reasoning fails, the issue is likely search/retrieval (Stage 3).

## Damping Scrubber
Mathematically determine if the current PPR damping factor ($\alpha$) can reach necessary evidence.

```bash
python debug_graph_v5.py --sim-decay ALPHA LENGTH
```
- Calculates $P = \alpha^L$.
- **Use Case**: Determine if mass is theoretically capable of reaching a node $L$ hops away.

## Top Hubs Analysis
Find potential noise nodes that should have been filtered during extraction.

```bash
python debug_graph_v5.py --audit
```
- Lists top hubs by degree and PageRank.
- **Use Case**: If "2015" is a top hub, the Extractor is failing to prune generic time entities.
