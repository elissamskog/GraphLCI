# LIBRARIAN

## Purpose

Connects extracted entities and claims into a traversable knowledge graph. Determines which concepts across documents are equivalent, links related items, and optionally organizes everything into thematic communities with summaries.

## Inputs

Entities and claims, each with a numeric vector capturing its meaning, plus metadata about source documents and text passages.

## Outputs

A connected knowledge graph where nodes are concepts and edges represent relationships (semantic similarity, co-occurrence, shared document origin). Includes edges that bridge different documents. Optionally produces a hierarchy of thematic clusters with summaries.

## Key Decisions

- How similar two concepts must be before connecting them (similarity threshold)
- Maximum connections per concept (prevents any single node from dominating)
- Whether to prioritize cross-document connections over within-document ones
- How to partition the graph into meaningful thematic clusters
- How many levels of abstraction in the community hierarchy

## Failure Modes

- Connects unrelated concepts (threshold too low or embeddings too similar)
- Fails to connect equivalent concepts across documents (threshold too high)
- Produces a fragmented graph with unreachable isolated nodes
- Community detection creates clusters that don't correspond to meaningful themes
- Performance degrades on very large document sets
