# Technical Plan

> **Statistical Stack & Algorithms:** Cosine similarity over dense embeddings |
> HNSW approximate nearest-neighbor index | Personalized PageRank (PPR) with
> damping factor for subgraph scope expansion | Leiden algorithm for hierarchical
> community detection (modularity optimization) | LLM-based Named Entity
> Recognition (NER) and Open Information Extraction (OpenIE) | Self-reflection
> gleaning loop for extraction recall | Coreference resolution | MAP / F1
> evaluation metrics | Intent-conditioned lane routing heuristic

## Version: 5.0
## Architecture: HippoRAG + GraphRAG Hybrid

---

## 1. Design Principles

### 1.1 Core Hypothesis

Graph traversal improves Long-Context Injection (LCI) by providing structurally-informed scope and relationship context.

### 1.2 Architectural Constraints

| Principle | Implementation |
|-----------|----------------|
| LLMs for semantics | Extraction and summarization only |
| Vectors for structure | Edge creation via embedding similarity |
| Algorithms for traversal | PPR for scope, Leiden for communities |
| Single-pass reasoning | Unified LLM determines epistemic relationships |

### 1.3 What LLMs Do / Don't Do

| LLM Responsibility | NOT LLM Responsibility |
|--------------------|------------------------|
| Extract entities from text | Create edges between entities |
| Extract claims from text | Determine document relationships |
| Summarize communities | Traverse the graph |
| Classify query intent | Resolve scope |
| Infer epistemic relationships | Build similarity index |

---

## 2. Pipeline Overview

```
[Documents]
     |
     v
+---------+
| CHUNKER | -> ChunkerOutput (TextChunks)
+---------+
     |
     v
+-----------+
| EXTRACTOR | -> ExplorerOutput (NodeSet: entities + claims + embeddings)
+-----------+
     |
     v
+-----------+
| LIBRARIAN | -> KnowledgeGraph (nodes + structural edges + communities)
+-----------+
     |
     v
[INDEX STORED]
     |
     | <- Query
     v
+---------+
| PLANNER | -> Scope + RouterDecision + AssembledContext
+---------+
     |
     v
+----------+
| REASONER | -> Finding (verdict + evidence + reasoning trace)
+----------+
```

---

## 3. Shared Types

### 3.1 Node Types

```python
class NodeType(Enum):
    DOCUMENT = "DOCUMENT"
    ENTITY = "ENTITY"
    DEFINITION = "DEFINITION"
    CLAUSE = "CLAUSE"
```

### 3.2 Structural Edge Types (Index Time)

```python
class StructuralEdgeType(Enum):
    SYNONYMY = "SYNONYMY"
    CITES = "CITES"
    REFERENCES = "REFERENCES"
    SAME_SOURCE = "SAME_SOURCE"
    SAME_CHUNK = "SAME_CHUNK"
    DEFINED_BY = "DEFINED_BY"
```

### 3.3 Epistemic Relationships (Reasoning Time)

```python
class EpistemicRelationship(Enum):
    SUPPORTS = "SUPPORTS"
    STRONGLY_SUPPORTS = "STRONGLY_SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    STRONGLY_CONTRADICTS = "STRONGLY_CONTRADICTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    PARTIALLY_CONTRADICTS = "PARTIALLY_CONTRADICTS"
    REFINES = "REFINES"
    EXTENDS = "EXTENDS"
    SUPERSEDES = "SUPERSEDES"
    PRECEDES = "PRECEDES"
    UNRELATED = "UNRELATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
```

### 3.4 Intent Types

```python
class IntentType(Enum):
    COMPARISON = "COMPARISON"
    SENSEMAKING = "SENSEMAKING"
    LOOKUP = "LOOKUP"
    CLAIM_VERIFICATION = "CLAIM_VERIFICATION"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"
    EVIDENCE_SYNTHESIS = "EVIDENCE_SYNTHESIS"
    METHODOLOGY_COMPARISON = "METHODOLOGY_COMPARISON"
```

### 3.5 Lane Types

```python
class LaneType(Enum):
    LANE_A_GRAPH_LCI = "LANE_A_GRAPH_LCI"       # Full docs + edge grammar
    LANE_B_GRAPH_ONLY = "LANE_B_GRAPH_ONLY"     # Community map-reduce
    LANE_C_VECTOR_ONLY = "LANE_C_VECTOR_ONLY"   # Direct chunk retrieval
```

### 3.6 Verdict Types

```python
class VerdictType(Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    PARTIALLY_REFUTED = "PARTIALLY_REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_FOUND = "NOT_FOUND"
```

---

## 4. CHUNKER

### 4.1 Purpose

Split documents into token-bounded text chunks. Pure Python — no LLM dependency.

### 4.2 Input

```python
@dataclass
class ChunkerInput:
    documents: List[RawDocument]
    config: Optional[Dict[str, Any]] = None
```

### 4.3 Output

```python
@dataclass
class ChunkerOutput:
    chunks: List[TextChunk]
    source_doc_ids: List[str]
    chunking_config: Optional[Dict[str, Any]] = None
    trace: Optional[ChunkerTrace] = None
```

### 4.4 Process

1. **Chunk** documents into ~1200 token segments with semantic boundaries

---

## 5. EXTRACTOR

### 5.1 Purpose

Transform text chunks into embedded nodes via LLM extraction.

### 5.2 Input

```python
@dataclass
class ExtractorInput:
    chunks: List[TextChunk]
    source_doc_ids: List[str]
    config: Optional[Dict[str, Any]] = None
```

### 5.3 Output

```python
@dataclass
class ExplorerOutput:
    node_set: NodeSet
    taxonomy: Optional[DomainTaxonomy] = None
    chunk_node_map: Optional[Dict[str, List[str]]] = None
    trace: Optional[ExtractorTrace] = None
```

Same contract as original EXPLORER — LIBRARIAN is unchanged.

### 5.4 Key Data Structures

```python
@dataclass
class TextChunk:
    chunk_id: str
    source_doc_id: str
    text: str
    token_count: int
    position: int
    page_number: Optional[int] = None
    section_header: Optional[str] = None


@dataclass
class ExtractedEntity:
    surface_form: str
    entity_type: str
    chunk_id: str
    char_offset: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ExtractedClaim:
    claim_id: str
    claim_text: str
    subject_entity: str
    object_entity: str
    predicate: str
    description: str
    chunk_id: str
    confidence: float


@dataclass
class EmbeddedNode:
    node_id: str
    node_type: str
    canonical_text: str
    embedding: List[float]
    source_doc_id: str
    source_chunks: List[str]
    metadata: Dict[str, Any]


@dataclass
class NodeSet:
    nodes: List[EmbeddedNode]
    source_doc_ids: List[str]
    chunk_count: int
    entity_count: int
    claim_count: int
    extraction_config: Optional[Dict[str, Any]] = None
    coreference_groups: Optional[List[CoreferenceGroup]] = None
    avg_claims_per_chunk: Optional[float] = None
```

### 5.5 Process

1. **Extract** entities via LLM (NER prompt)
2. **Extract** claims via LLM (OpenIE prompt with entities)
3. **Glean** missed items via reflection loop
4. **Resolve** coreferences via LLM
5. **Embed** canonical forms

---

## 6. LIBRARIAN

### 6.1 Purpose

Create structural edges via vector similarity and build the knowledge graph with optional community detection.

### 6.2 Input

```python
@dataclass
class LibrarianInput:
    node_set: NodeSet
    chunk_node_map: Optional[Dict[str, List[str]]] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    taxonomy: Optional[DomainTaxonomy] = None
```

### 6.3 Output

```python
@dataclass
class LibrarianOutput:
    graph: KnowledgeGraph
    community_structure: Optional[CommunityStructure] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    similarity_index_path: Optional[str] = None
```

### 6.4 Key Data Structures

```python
@dataclass
class StructuralEdge:
    edge_id: str
    source: str
    target: str
    edge_type: StructuralEdgeType
    weight: float
    metadata: Dict[str, Any]
    similarity: Optional[float] = None
    is_cross_document: bool = False


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, EmbeddedNode]
    edges: Dict[str, StructuralEdge]
    adjacency: Dict[str, List[str]]
    node_by_doc: Dict[str, List[str]]
    node_by_type: Optional[Dict[str, List[str]]] = None
    bridge_edges: Optional[List[str]] = None
    node_count: int = 0
    edge_count: int = 0
    document_count: int = 0
    bridge_count: int = 0


@dataclass
class Community:
    community_id: str
    level: int
    node_ids: List[str]
    parent_community_id: Optional[str] = None
    child_community_ids: Optional[List[str]] = None
    summary: Optional[str] = None
    key_themes: Optional[List[str]] = None


@dataclass
class CommunityStructure:
    communities: Dict[str, Community]
    levels: Dict[int, List[str]]
    node_to_community: Dict[str, str]
```

### 6.5 Process

1. **Extract** explicit edges (CITES, SAME_SOURCE, SAME_CHUNK)
2. **Build** similarity index (HNSW)
3. **Create** synonymy edges where `cosine(node_i, node_j) > τ` (default τ=0.8)
4. **Assemble** graph with all edges
5. **Detect** communities via Leiden algorithm (optional)
6. **Summarize** communities via LLM (optional)

---

## 7. PLANNER

### 7.1 Purpose

Resolve query scope via graph traversal and route to the appropriate processing lane.

### 7.2 Input

```python
@dataclass
class PlannerInput:
    query: QueryRequest
    graph: KnowledgeGraph
    similarity_index: Any  # SimilarityIndex instance
    target_documents: Optional[List[str]] = None
    community_structure: Optional[CommunityStructure] = None
    document_store: Optional[Dict[str, RawDocument]] = None
    chunks: Optional[List[TextChunk]] = None  # original chunker output for LANE_C
```

### 7.3 Output

```python
@dataclass
class PlannerOutput:
    router_decision: RouterDecisionV5
    focused_context: Optional[FocusedContext] = None
    global_context: Optional[GlobalContext] = None
    targeted_context: Optional[TargetedContext] = None
```

### 7.4 Key Data Structures

```python
@dataclass
class IntentClassification:
    classification: IntentType
    confidence: float
    reasoning: str


@dataclass
class SeedNode:
    node_id: str
    similarity: float
    node_type: str
    canonical_text: str
    source_doc_id: str


@dataclass
class ScopedNode:
    node: EmbeddedNode
    ppr_score: float
    rank: int


@dataclass
class Scope:
    nodes: Dict[str, ScopedNode]
    edges: Dict[str, StructuralEdge]
    document_ids: List[str]
    seed_node_ids: List[str]
    total_ppr_mass: float


@dataclass
class RouterDecisionV5:
    query_id: str
    intent: IntentClassification
    selected_lane: LaneType
    seed_nodes: List[SeedNode]
    scope: Optional[Scope] = None
    confidence: float = 0.0


@dataclass
class FocusedContext:
    documents: Dict[str, RawDocument]
    edge_grammar: str
    node_context: str
    scope: Scope


@dataclass
class GlobalContext:
    community_summaries: Dict[str, str]
    scope: Scope


@dataclass
class TargetedContext:
    chunks: List[RankedChunk]
    scope: Scope
```

### 7.5 Process

1. **Embed** query
2. **Classify** intent via LLM
3. **Find** seed nodes via similarity search
4. **Expand** scope via Personalized PageRank from seeds
5. **Route** to lane based on intent + scope size
6. **Assemble** context for selected lane

### 7.6 Routing Logic

| Intent | Default Lane | Override Condition |
|--------|--------------|-------------------|
| COMPARISON | LANE_A_GRAPH_LCI | — |
| CONTRADICTION_DETECTION | LANE_A_GRAPH_LCI | — |
| CLAIM_VERIFICATION | LANE_A_GRAPH_LCI | — |
| METHODOLOGY_COMPARISON | LANE_A_GRAPH_LCI | — |
| EVIDENCE_SYNTHESIS | LANE_B_GRAPH_ONLY | < 5 docs → LANE_A_GRAPH_LCI |
| SENSEMAKING | LANE_B_GRAPH_ONLY | < 5 docs → LANE_A_GRAPH_LCI |
| LOOKUP | LANE_C_VECTOR_ONLY | — |
| Any | LANE_C_VECTOR_ONLY | < 5 nodes in scope |

---

## 8. REASONER

### 8.1 Purpose

Perform unified LLM reasoning to infer epistemic relationships and produce a verdict with full provenance.

### 8.2 Input

```python
@dataclass
class ReasonerInput:
    query: QueryRequest
    query_id: str
    router_decision: RouterDecisionV5
    focused_context: Optional[FocusedContext] = None
    global_context: Optional[GlobalContext] = None
    targeted_context: Optional[TargetedContext] = None
```

### 8.3 Output

```python
@dataclass
class ReasonerOutput:
    finding: Finding
    trace: ReasonerTrace
    raw_response: str
```

### 8.4 Key Data Structures

```python
@dataclass
class InferredRelationship:
    source_id: str
    target_id: str
    relationship: EpistemicRelationship
    confidence: float
    reasoning: str
    evidence_quote: Optional[str] = None


@dataclass
class GroundedEvidence:
    claim: str
    source_document: str
    source_chunk: Optional[str] = None
    exact_quote: str = ""
    page_number: Optional[int] = None
    char_offset: Optional[int] = None


@dataclass
class Verdict:
    verdict_type: VerdictType
    answer: str
    confidence: float
    summary: str
    primary_relationship: Optional[EpistemicRelationship] = None


@dataclass
class ScopeSummary:
    document_ids: List[str]
    document_count: int
    node_count: int
    edge_count: int
    bridge_edges_used: int


@dataclass
class ConfidenceFactors:
    overall: float
    evidence_quality: float
    source_agreement: float
    coverage: float


@dataclass
class Finding:
    finding_id: str
    query_id: str
    timestamp: datetime
    verdict: Verdict
    relationships: List[InferredRelationship]
    evidence: List[GroundedEvidence]
    scope: ScopeSummary
    lane_used: str
    intent_classified: str
    confidence_factors: ConfidenceFactors
    reasoning_trace: str
```

### 8.5 Process

1. **Format** context into lane-specific prompt
2. **Execute** LLM reasoning (single pass)
3. **Extract** epistemic relationships from response
4. **Ground** evidence in source documents
5. **Determine** verdict type from relationships
6. **Assemble** Finding with full provenance

---

## 9. Reasoning Traces

Each component emits a trace for debugging and evaluation.

```python
@dataclass
class ChunkerTrace:
    trace_id: str
    input_doc_ids: List[str]
    total_chunks: int
    chunks: Optional[List[ChunkSummary]] = None


@dataclass
class ExtractorTrace:
    trace_id: str
    input_chunk_count: int
    entities_extracted: int
    claims_extracted: int
    gleaning_iterations: int
    llm_calls: int
    extraction_time_seconds: float
    chunk_details: Optional[List[ChunkExtractionDetail]] = None


@dataclass
class ExplorerTrace:
    trace_id: str
    input_doc_ids: List[str]
    total_chunks: int
    entities_extracted: int
    claims_extracted: int
    gleaning_iterations: int
    llm_calls: int
    extraction_time_seconds: float


@dataclass
class LibrarianTrace:
    trace_id: str
    input_node_count: int
    explicit_edges_created: int
    synonymy_edges_created: int
    bridge_edges_identified: int
    similarity_threshold_used: float
    communities_detected: Optional[int] = None
    total_time_seconds: float = 0.0
    llm_calls: int = 0


@dataclass
class PlannerTrace:
    trace_id: str
    query_id: str
    intent_classification: str
    seeds_found: int
    ppr_iterations: int
    scope_nodes: int
    scope_documents: int
    selected_lane: str
    total_time_ms: float


@dataclass
class ReasonerTrace:
    trace_id: str
    query_id: str
    lane_used: str
    intent: str
    documents_in_scope: int
    relationships_inferred: int
    evidence_grounded: int
    verdict_type: str
    verdict_confidence: float
    llm_time_ms: float
    total_time_ms: float
```

---

## 10. Configuration

```yaml
chunker:
  chunking:
    target_size: 1200
    overlap: 100
    strategy: "semantic"

extractor:
  extraction:
    model: "gpt-4o-mini"
    temperature: 0.0
    phases: { ner: true, openIE: true }
  gleaning:
    enabled: true
    max_iterations: 2
    skip_below_tokens: 800
  postprocessing:
    coreference_resolution: true
    embedding_model: "text-embedding-3-small"
  schema_path: null
  tracing: { enabled: true, verbosity: "summary" }

librarian:
  synonymy:
    threshold: 0.8
    max_edges_per_node: 10
  similarity_index:
    type: "hnsw"
  community_detection:
    enabled: true
    algorithm: "leiden"
    levels: 3
  summarization:
    enabled: true
    model: "gpt-4o-mini"

planner:
  seeding:
    top_k: 10
    similarity_threshold: 0.5
  ppr:
    damping: 0.85
    max_iterations: 100
  scope:
    max_nodes: 50
    score_threshold: 0.01
  routing:
    global_threshold_docs: 5

reasoner:
  llm:
    model: "gpt-4o"
    temperature: 0.1
    max_tokens: 4000
  grounding:
    fuzzy_match_threshold: 0.85
```

---

## 11. Key Algorithmic Details

### 11.1 Synonymy Edge Creation (HippoRAG Protocol)

```
For each node pair (i, j):
    similarity = cosine(embedding_i, embedding_j)
    if similarity > threshold τ:
        create edge (i, j, SYNONYMY, weight=similarity)
```

**Threshold τ = 0.8** balances precision and recall.

### 11.2 Personalized PageRank

```
Initialize: scores = seed_weights (normalized)
Repeat until convergence:
    scores = (1 - damping) * seed_weights + damping * (transition_matrix @ scores)
```

**Damping = 0.85** (standard). Seeds weighted by query similarity.

### 11.3 Lane Routing

```
if intent in [COMPARISON, CONTRADICTION_DETECTION, CLAIM_VERIFICATION, METHODOLOGY_COMPARISON]:
    return LANE_A_GRAPH_LCI
elif intent in [EVIDENCE_SYNTHESIS, SENSEMAKING]:
    return LANE_B_GRAPH_ONLY if docs > 5 else LANE_A_GRAPH_LCI
elif intent == LOOKUP:
    return LANE_C_VECTOR_ONLY
if scope.nodes < 5:
    return LANE_C_VECTOR_ONLY  # override
```

---

## 12. Example Flow

Demonstrates a CONTRADICTION query ("Does the MSA liability cap contradict DORA?") flowing through CHUNKER → EXTRACTOR → LIBRARIAN → PLANNER → REASONER to a REFUTED verdict.

---

## 13. Success Metrics

| Metric | Target |
|--------|--------|
| Retrieval MAP | ≥ 0.70 |
| Relationship Detection F1 | ≥ 0.65 |
| End-to-End Accuracy | ≥ 0.75 |
| Graph Contribution Δ | ≥ 0.10 |