# KnowledgeAudit - PLANNER Instance

You are the PLANNER instance managing intent classification and lane routing.

## Project Context
See `../docs/PLAN.md` section 3.2 for your full specification.

## Your Responsibilities
Query embedding, seed node identification, PPR traversal, intent classification, lane routing, scope assembly

## Directory Ownership
```
planner/
└── ...
```

## Key Requirements
- Routing accuracy target: ≥ 0.90
- Must handle adversarial paraphrasing (no keyword heuristics)
- Confidence scores must be calibrated

## Do Not Modify
- Files outside `planner/`
- `../schemas/` (owned by ROOT, but you import from it)


## INPUT/OUTPUT
Read ./schemas. Stick to the input output structures described here.

### Schema Contract

Your controller signature is `process(PlannerInput) -> PlannerOutput`.

**Input** (`schemas.contracts.PlannerInput`):
- `query: QueryRequest` — the user query to plan for
- `graph: KnowledgeGraph` — the knowledge graph from LIBRARIAN
- `similarity_index: Any` — SimilarityIndex instance for vector search
- `target_documents: Optional[List[str]]` — restrict scope to specific documents
- `community_structure: Optional[CommunityStructure]` — Leiden community hierarchy
- `document_store: Optional[Dict[str, RawDocument]]` — raw documents for LANE_A context
- `chunks: Optional[List[TextChunk]]` — original chunker output for LANE_C vector retrieval

**Output** (`schemas.contracts.PlannerOutput`):
- `router_decision: RouterDecisionV5` — intent, lane, seeds, scope
- `focused_context: Optional[FocusedContext]` — LANE_A: full docs + edge grammar
- `global_context: Optional[GlobalContext]` — LANE_B: community summaries
- `targeted_context: Optional[TargetedContext]` — LANE_C: ranked chunks

**Key shared types** (all from `schemas.contracts`):
`PlannerInput`, `PlannerOutput`, `RouterDecisionV5`, `Scope`, `ScopedNode`, `SeedNode`, `FocusedContext`, `GlobalContext`, `TargetedContext`, `IntentClassification`, `LaneType`, `IntentType`, `PlannerTrace`

## Implementation Notes

### Chunk gathering (`_gather_chunks`)
`_gather_chunks()` in `planner_controller.py` prefers real chunks from `PlannerInput.chunks` (produced by the CHUNKER) over building synthetic chunks from `document_store`. The synthetic fallback exists for backward compatibility but produces chunk IDs (e.g. `"chunk_Arthur's Magazine_0"`) that do not match the IDs stored in graph nodes (e.g. `"chunk_0029"`), causing `assemble_targeted_context()` to return zero ranked chunks.

Always ensure the orchestrator passes `chunks` into `PlannerInput` when available.

### Known issues: Lane routing
Three issues cause most claims to route to LANE_C regardless of intent:

1. **Missing temporal comparison patterns** — The intent classifier's COMPARISON patterns only cover structural/legal comparison terms. Temporal language (`before`, `after`, `earlier`, `later`, `first`, `last`) is not matched, so temporal comparisons fall through to LOOKUP. Needs new patterns in `intent_classifier.py`.
2. **High confidence threshold** — The 0.7 threshold in `_route_v5()` is too high for the pattern-based confidence formula. Even correctly identified COMPARISON intents often score ~0.4 and get routed to LANE_C. Consider lowering to 0.5 or making it configurable.
3. **Small-scope guard overrides intent** — The `node_count < 5` guard in `_route_v5()` forces LANE_C for small scopes, defeating LANE_A even for comparison queries that benefit most from rich context. Consider checking intent type before the node-count guard.