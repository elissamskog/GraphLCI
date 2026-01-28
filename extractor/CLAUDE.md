# KnowledgeAudit - EXTRACTOR Instance

You are the EXTRACTOR instance for the KnowledgeAudit GALC system.

## Module Overview
LLM-based entity/claim extraction, gleaning, coreference resolution, and node embedding.
Receives pre-chunked text from the CHUNKER and produces embedded nodes for the LIBRARIAN.
See `docs/EXTRACTOR.md` for full specification.

## Your Responsibilities
- Entity extraction in `entity_extractor.py`
- Claim extraction in `claim_extractor.py`
- Gleaning (self-reflection loop) in `gleaner.py`
- Coreference resolution in `coreference_resolver.py`
- Node embedding in `node_embedder.py`
- JSON parsing utilities in `json_utils.py`
- Controller orchestration in `extractor_controller.py`
- Legacy v4 support in `summarizer.py` and `domain_reasoner.py`
- Unit tests in `test_extractor_v5.py`

## Directory Ownership
```
extractor/
├── __init__.py               # Exports ExtractorController
├── CLAUDE.md                 # This file
├── extractor_controller.py   # ExtractorController.process(ExtractorInput) → ExplorerOutput
├── entity_extractor.py       # extract_entities() — NER via LLM
├── claim_extractor.py        # extract_claims() — OpenIE via LLM
├── gleaner.py                # glean_missing() — iterative reflection loop
├── coreference_resolver.py   # resolve_coreferences() — entity unification via LLM
├── node_embedder.py          # embed_nodes() — vector embedding
├── json_utils.py             # parse_json(), parse_json_list() — robust LLM response parsing
├── summarizer.py             # Summarizer class (legacy v4)
├── domain_reasoner.py        # DomainReasoner class (legacy v4)
└── test_extractor_v5.py      # Unit tests
```

## Data Flow
```
ExtractorInput(chunks, source_doc_ids, config)
    → Per chunk:
        extract_entities() → ExtractedEntity[]
        extract_claims()   → ExtractedClaim[]
        glean_missing()    → GleaningResult (additional entities/claims)
    → resolve_coreferences() → CoreferenceGroup[]
    → Build NodeDescriptors (entity + claim descriptors)
    → embed_nodes() → EmbeddedNode[]
    → ExplorerOutput(node_set, chunk_node_map)
```

## Key Contracts
- **Input**: `ExtractorInput` (from `schemas.contracts`)
- **Output**: `ExplorerOutput` (from `schemas.contracts`) — same contract LIBRARIAN consumes
- **Intermediate**: `ExtractedEntity`, `ExtractedClaim`, `CoreferenceGroup`,
  `EmbeddedNode`, `NodeSet` (all from `schemas.contracts`)

## Boundary Rules

You may ONLY modify files in:
- `extractor/`

You may NOT modify files in:
- `schemas/` — owned by ROOT instance
- `orchestrator.py` — owned by ROOT instance
- `llm_client.py` — owned by ROOT instance
- `config/` — owned by ROOT instance
- `evaluation/` — owned by ROOT instance
- `chunker/` — owned by CHUNKER instance
- `librarian/` — owned by LIBRARIAN instance
- `planner/` — owned by PLANNER instance
- `reasoner/` — owned by REASONER instance
- `explorer/` — backward-compat shim (no direct edits needed)

If you identify a bug in another module's code, report it to the user
with the specific file, line, and fix needed. Do not edit their files directly.

## Design Constraints
- **Requires LLM client**: `entity_extractor`, `claim_extractor`, `gleaner`, and
  `coreference_resolver` all depend on an LLM client implementing `generate(prompt) → str`.
- **Requires embedding client**: `node_embedder` depends on an embedding client
  implementing `embed(texts) → List[List[float]]`.
- **No chunking**: Document splitting is handled by the CHUNKER module. This module
  receives pre-chunked `TextChunk` objects only.
- Internal imports use `from extractor.*` (not `from explorer.*`).
