# KnowledgeAudit - REASONER Instance

You are the REASONER instance responsible for epistemic inference and verdict generation.

## Project Context
See `../docs/REASONER.md` for your full specification.
See `../docs/PLAN.md` for overall architecture.

## Your Responsibilities
- Build prompts from assembled context (focused, global, targeted)
- Extract epistemic relationships from LLM responses
- Ground evidence to source documents and chunks
- Determine verdict type

## Directory Ownership
```
reasoner/
├── __init__.py
├── CLAUDE.md                    # This file
├── reasoner_controller.py       # Main controller with process()
├── prompt_builder.py            # Build prompts per lane
├── epistemic_extractor.py       # Extract relationships from LLM output
├── evidence_grounder.py         # Ground evidence to sources
└── verdict_resolver.py          # Determine verdict and confidence
```

## Shared Types
Import from the project-level schemas:
```python
from schemas.contracts import (
    ReasonerInput,
    ReasonerOutput,
    ReasonerTrace,
    Finding,
    Verdict,
    VerdictType,
    InferredRelationship,
    EpistemicRelationship,
    GroundedEvidence,
    ConfidenceFactors,
    ScopeSummary,
    FocusedContext,
    GlobalContext,
    TargetedContext,
    LLMClient,
    # ... other types
)
```

## Do Not Modify
- Files outside `reasoner/`
- `../schemas/` (owned by ROOT, but you import from it)
