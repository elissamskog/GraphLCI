## CHUNKER: Document Chunking Specification

> **Statistical Stack & Algorithms:** Token-bounded segmentation with sliding
> window overlap | Semantic boundary detection (sentence and paragraph splitting)
> | Configurable chunk-size / recall tradeoff (600–2400 tokens) | No stochastic
> or learned components — deterministic, rule-based splitting only

### Overview

Pure-Python module that splits raw documents into token-bounded text chunks.
No LLM or embedding client dependency.

---

### 1. Chunking Strategy

#### 1.1 Chunk Size Selection

| Chunk Size | Extraction Recall | Token Cost | Recommendation |
|------------|-------------------|------------|----------------|
| 600 tokens | High | High | Use for high-precision domains |
| 1200 tokens | Medium | Medium | Default |
| 2400 tokens | Low (without gleaning) | Low | Use only with gleaning loop |

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

#### 1.3 Default Configuration

```yaml
chunking:
  target_size: 1200  # tokens
  overlap: 100       # token overlap between chunks
  strategy: "semantic"  # prefer sentence/paragraph boundaries
```

---

### 2. Input Contract

```python
@dataclass
class ChunkerInput:
    documents: List[RawDocument]
    config: Optional[Dict[str, Any]] = None
```

---

### 3. Output Contract

```python
@dataclass
class ChunkerOutput:
    chunks: List[TextChunk]
    source_doc_ids: List[str]
    chunking_config: Optional[Dict[str, Any]] = None
    trace: Optional[ChunkerTrace] = None
```

---

### 4. Configuration Reference

```yaml
chunker:
  chunking:
    target_size: 1200
    overlap: 100
    strategy: "semantic"
```

---

### 5. Reasoning Trace

```python
@dataclass
class ChunkSummary:
    chunk_id: str
    source_doc_id: str
    token_count: int
    position: int
    section_header: Optional[str]
    text_preview: str               # first 300 chars of chunk text

@dataclass
class ChunkerTrace:
    trace_id: str
    input_doc_ids: List[str]
    total_chunks: int
    chunks: Optional[List[ChunkSummary]] = None
```

The `chunks` field contains a `ChunkSummary` per produced chunk with a 300-char
text preview, allowing inspection of chunking results without accessing the full
`ChunkerOutput.chunks` list.
