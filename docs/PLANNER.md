# PLANNER

## Purpose

Translates a natural language query into a focused slice of the knowledge graph and a processing strategy. Determines what kind of question is being asked, finds the most relevant starting points, expands through the graph to gather context, and selects a reasoning approach.

## Inputs

A natural language query, the full knowledge graph, and a searchable index of concept vectors.

## Outputs

A scoped subgraph of nodes and edges relevant to the query, a classification of intent (comparison, lookup, synthesis, etc.), and context formatted for the chosen reasoning strategy.

## Key Decisions

- What kind of question the user is asking (intent classification)
- Which graph nodes are the best starting points for the query (seed selection)
- How far to expand from those starting points (scope breadth)
- Which reasoning strategy to use: full document analysis, thematic summaries, or direct text lookup

## Failure Modes

- Misclassifies the query intent, routing to the wrong reasoning strategy
- Selects poor starting points, causing the scope to miss key evidence
- Scope too narrow (misses relevant context) or too broad (drowns reasoning in noise)
- Defaults to simple lookup on complex queries due to small graph size
