# KnowledgeAudit - EXPLORER Instance (Deprecated Shim)

## Status: Backward-Compatibility Shim Only

The `explorer/` package has been refactored. All implementation code now lives in
two sibling packages:
- `chunker/` — Pure-Python document chunking (see `docs/CHUNKER.md`)
- `extractor/` — LLM extraction pipeline (see `docs/EXTRACTOR.md`)

## What Remains Here
```
explorer/
├── __init__.py               # Re-exports from chunker/ and extractor/
├── CLAUDE.md                 # This file
└── explorer_controller.py    # ExplorerControllerV5 delegates to ChunkerController + ExtractorController
```

## How the Shim Works
- `explorer/__init__.py` re-exports all public symbols from `chunker` and `extractor`
  so that existing `from explorer.X import Y` statements continue to work.
- `ExplorerControllerV5` in `explorer_controller.py` delegates internally to
  `ChunkerController.process()` then `ExtractorController.process()`.
- `ExplorerController` (legacy v4) imports `Summarizer` and `DomainReasoner`
  from `extractor/`.

## Boundary Rules

**Do not add new code here.** New features belong in `chunker/` or `extractor/`.

If a backward-compat re-export is missing, add it to `explorer/__init__.py` — but
the actual implementation must live in `chunker/` or `extractor/`.

You may NOT modify files in:
- `schemas/` — owned by ROOT instance
- `orchestrator.py` — owned by ROOT instance
- `chunker/` — owned by CHUNKER instance
- `extractor/` — owned by EXTRACTOR instance
- `librarian/` — owned by LIBRARIAN instance
- `planner/` — owned by PLANNER instance
- `reasoner/` — owned by REASONER instance
