## EXPLORER: Node Extraction Specification

> **Statistical Stack & Algorithms:** *(Deprecated — delegates to CHUNKER +
> EXTRACTOR.)* LLM-based Named Entity Recognition (NER) and Open Information
> Extraction (OpenIE) | Self-reflection gleaning loop for recall recovery |
> LLM-driven coreference resolution | Dense vector embeddings via transformer
> encoder (text-embedding-3-small / BGE) | Token-bounded semantic chunking with
> sliding-window overlap

### Overview

The EXPLORER component transforms unstructured text into structured graph nodes. It uses LLMs as extraction engines—not for bridging or relationship inference, but for semantic parsing of text into discrete, embeddable units.

---

### 1. Chunking Strategy

#### 1.1 Chunk Size Selection

| Chunk Size | Extraction Recall | Token Cost | Recommendation |
|------------|-------------------|------------|----------------|
| 600 tokens | High | High | Use for high-precision domains |
| 1200 tokens | Medium | Medium | Default |
| 2400 tokens | Low (without gleaning) | Low | Use only with gleaning loop |

**Default Configuration:**
```yaml
chunking:
  target_size: 1200  # tokens
  overlap: 100       # token overlap between chunks
  strategy: "semantic"  # prefer sentence/paragraph boundaries
```

#### 1.2 Chunk Metadata

Each chunk retains provenance for downstream assembly:

```python
@dataclass
class TextChunk:
    chunk_id: str
    source_doc_id: str
    text: str
    token_count: int
    position: int              # ordinal position in source
    page_number: Optional[int]
    section_header: Optional[str]
```

---

### 2. Extraction Pipeline

`TextChunk → NER → OpenIE → Gleaning Loop → Coreference + Embedding → ExtractedNodes`

---

### 3. Phase 1: Named Entity Recognition

**Prompt Intent:** Given a text chunk, the LLM extracts all named entities (concepts, actors, artifacts, quantities) exactly as they appear. Returns a structured list with entity types. No inference of unstated entities.

#### 3.1 Output Schema

```python
@dataclass
class ExtractedEntity:
    surface_form: str
    entity_type: str
    chunk_id: str
    char_offset: Optional[int] = None
    description: Optional[str] = None
```

---

### 4. Phase 2: Open Information Extraction

Using the entities from Phase 1, extract structured claims and relationships.

**Prompt Intent:** Given a text chunk and the Phase 1 entity list, the LLM extracts atomic, self-contained factual claims with subject/object/predicate structure. Each claim includes a contextual description and confidence score.

#### 4.1 Output Schema

```python
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
```

---

### 5. Phase 3: Self-Reflection (Gleaning Loop)

To compensate for extraction recall degradation on larger chunks, implement a reflection loop.

**Prompt Intent:** Given the original text, already-extracted entities, and already-extracted claims, the LLM re-examines the text for any missed items. Responds with new entities/claims or "EXTRACTION COMPLETE".

```python
def extract_with_gleaning(chunk: TextChunk, max_iterations: int = 2) -> ExtractionResult:
    """Run NER + OpenIE, then iteratively glean missed entities/claims up to max_iterations.
    Returns merged ExtractionResult with all entities and claims found across passes."""
```

#### 5.1 Configuration

```yaml
gleaning:
  enabled: true
  max_iterations: 2
  # Skip gleaning for small chunks where recall is already high
  skip_below_tokens: 800
```

---

### 6. Phase 4: Post-Processing

#### 6.1 Coreference Resolution

Unify entities that refer to the same concept within and across chunks.

**Prompt Intent:** Given a list of extracted entities, the LLM identifies co-referent entities (same real-world concept) and groups them under a canonical form with variants.

```python
@dataclass
class CoreferenceGroup:
    canonical_form: str
    variants: List[str]
```

#### 6.2 Node Embedding

After coreference resolution, embed each canonical entity and claim for downstream vector bridging.

```python
@dataclass
class EmbeddedNode:
    node_id: str
    node_type: str
    canonical_text: str
    embedding: List[float]
    source_doc_id: str
    source_chunks: List[str]
    metadata: Dict[str, Any]


class NodeType(Enum):
    DOCUMENT = "DOCUMENT"
    ENTITY = "ENTITY"
    DEFINITION = "DEFINITION"
    CLAUSE = "CLAUSE"
```

**Embedding Model Selection:**

| Model | Dimension | Use Case |
|-------|-----------|----------|
| `text-embedding-3-small` | 1536 | General purpose |
| `BAAI/bge-large-en-v1.5` | 1024 | Open source alternative |
| Domain-specific encoder | Varies | If available for domain |

---

### 7. Schema Enforcement (Optional)

For known domains, constrain extraction with a schema:

```yaml
# Example: Legal domain schema
extraction_schema:
  entity_types:
    - name: "Clause"
      description: "A specific provision in a contract or regulation"
      indicators: ["Section", "Article", "§", "Clause"]
    - name: "Obligation"
      description: "A required action or prohibition"
      indicators: ["shall", "must", "required to", "prohibited from"]
    - name: "Party"
      description: "A person or organization bound by the document"
      indicators: ["Vendor", "Client", "Licensor", "the Company"]
    - name: "Definition"
      description: "A term explicitly defined in the document"
      indicators: ["means", "defined as", "refers to"]

  claim_types:
    - name: "Requirement"
      description: "Something that must be done"
    - name: "Prohibition"
      description: "Something that must not be done"
    - name: "Condition"
      description: "A prerequisite or trigger"
    - name: "Limitation"
      description: "A cap, boundary, or constraint"
```

---

### 8. EXPLORER Output Contract

The EXPLORER produces a `NodeSet` ready for the LIBRARIAN's edge-building phase:

```python
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


@dataclass
class ExplorerOutput:
    node_set: NodeSet
    taxonomy: Optional[DomainTaxonomy] = None
    chunk_node_map: Optional[Dict[str, List[str]]] = None
    trace: Optional[ExtractorTrace] = None
```

---

### 9. Reasoning Trace

For debugging and evaluation, EXPLORER emits a trace:

```python
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
```

---

### 10. Configuration Reference

```yaml
explorer:
  # Chunking
  chunking:
    target_size: 1200
    overlap: 100
    strategy: "semantic"  # or "fixed"

  # Extraction
  extraction:
    model: "gpt-4o-mini"  # or local: "qwen2.5:14b"
    temperature: 0.0      # deterministic extraction
    phases:
      ner: true
      openIE: true

  # Gleaning
  gleaning:
    enabled: true
    max_iterations: 2
    skip_below_tokens: 800

  # Post-processing
  postprocessing:
    coreference_resolution: true
    embedding_model: "text-embedding-3-small"

  # Optional schema
  schema_path: null  # or "schemas/legal.yaml"

  # Tracing
  tracing:
    enabled: true
    verbosity: "summary"  # or "full"
```
