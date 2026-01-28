# KnowledgeAudit - CHUNKER Instance

You are the CHUNKER instance for the KnowledgeAudit GALC system.

## Module Overview
Pure-Python document chunking — no LLM or embedding dependency.
Splits raw documents into token-bounded text chunks for downstream extraction.
See `docs/CHUNKER.md` for full specification.

## Your Responsibilities
- Document splitting logic in `chunker.py`
- Controller orchestration in `chunker_controller.py`
- Unit tests in `test_chunker.py`

## Directory Ownership
```
chunker/
├── __init__.py               # Exports ChunkerController, chunk_document
├── CLAUDE.md                 # This file
├── chunker.py                # chunk_document() — core splitting logic
├── chunker_controller.py     # ChunkerController.process(ChunkerInput) → ChunkerOutput
└── test_chunker.py           # Unit tests
```

## Data Flow
```
ChunkerInput(documents, config)
    → chunk_document() per doc
    → ChunkerOutput(chunks, source_doc_ids, chunking_config)
```

## Key Contracts
- **Input**: `ChunkerInput` (from `schemas.contracts`)
- **Output**: `ChunkerOutput` (from `schemas.contracts`)
- **Intermediate**: `TextChunk` (from `schemas.contracts`)

## Boundary Rules

You may ONLY modify files in:
- `chunker/`

You may NOT modify files in:
- `schemas/` — owned by ROOT instance
- `orchestrator.py` — owned by ROOT instance
- `llm_client.py` — owned by ROOT instance
- `config/` — owned by ROOT instance
- `evaluation/` — owned by ROOT instance
- `extractor/` — owned by EXTRACTOR instance
- `librarian/` — owned by LIBRARIAN instance
- `planner/` — owned by PLANNER instance
- `reasoner/` — owned by REASONER instance
- `explorer/` — backward-compat shim (no direct edits needed)

If you identify a bug in another module's code, report it to the user
with the specific file, line, and fix needed. Do not edit their files directly.

## Design Constraints
- **No LLM dependency**: This module must never import or call an LLM client.
- **No embedding dependency**: This module must never import or call an embedding client.
- **Pure Python**: Only standard library + `schemas.contracts` imports.
- The chunking algorithm uses greedy segment accumulation with paragraph/sentence
  boundary detection and configurable overlap.
