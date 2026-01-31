# Three-Stage Extraction Architecture

## Overview

The v5 extractor implements a **three-stage pipeline** that separates structural analysis from semantic claim extraction:

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Deterministic Pre-processing                      │
│  (structural_analyzer.py)                                    │
│  - NER via spaCy/Stanza dependency parsing                  │
│  - Inject structural edges (compounds, appositives)         │
│  - No LLM calls, high confidence (0.95-0.99)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: LLM Claim Extraction                              │
│  (claim_extractor.py)                                        │
│  - Simplified prompt (no hierarchical decomposition)        │
│  - Focus on high-fidelity temporal & causal claims          │
│  - LLM (Qwen-14B) extracts factual assertions               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Parsing/Decomposition                             │
│  (extractor_controller.py)                                   │
│  - Parse LLM JSON output                                    │
│  - Apply deduplication across structural + LLM claims       │
│  - Coreference resolution & embedding                       │
└─────────────────────────────────────────────────────────────┘
```

## Design Rationale

### Why Three Stages?

1. **Separation of Concerns**: Structural relationships (e.g., "Exeter College, University of Oxford") are deterministic and don't require expensive LLM calls.

2. **LLM Efficiency**: By offloading hierarchical decomposition to Stage 1, we free up the LLM's context window to focus on complex temporal and causal reasoning.

3. **Robustness**: Dependency parsing via spaCy provides linguistically-grounded structural analysis, while falling back to heuristics when NLP libraries are unavailable.

4. **Auditability**: Structural bridges are tagged with `_struct_` IDs and logged separately, making it easy to verify the "Oxford Bridge" fix.

## Stage 1: Structural Analyzer

### When spaCy is Available

The `StructuralAnalyzer` uses dependency parsing to identify:

- **Appositives**: "Jane Doe, CEO of Acme" → `is_a("Jane Doe", "CEO")`
- **Prepositional Phrases**: "Exeter College of Oxford" → `part_of("Exeter College", "Oxford")`
- **Compounds**: Future support for compound nouns

### Fallback Heuristics

When spaCy is unavailable, the analyzer uses pattern matching:

- **Comma Rule**: "Entity A, Entity B" → `part_of` or `located_in` based on entity type

### Configuration

```yaml
structural_analysis:
  use_spacy: true  # Enable spaCy dependency parsing
  spacy_model: "en_core_web_sm"  # Model to load
```

## Stage 2: LLM Claim Extraction

### Simplified Prompt

The claim extraction prompt has been **simplified** to remove:

- ❌ HIERARCHICAL DECOMPOSITION rule (moved to Stage 1)
- ❌ SHARED CONTEXT rule (moved to Stage 1)

The LLM now focuses on:

- ✅ TEMPORAL: Dates, durations, sequences
- ✅ RELATIONAL: High-level relationships (married, works at)
- ✅ ATTRIBUTIVE: Properties and characteristics
- ✅ QUANTITATIVE: Numbers, amounts, thresholds
- ✅ CAUSAL: Causes, effects, consequences

### Example

**Input Text:**
> "She attended Exeter College, University of Oxford from 1990-1994."

**Stage 1 Output (Structural):**
```json
{
  "claim": "Exeter College is part of University of Oxford",
  "subject": "Exeter College",
  "object": "University of Oxford",
  "predicate": "part_of",
  "confidence": 0.95
}
```

**Stage 2 Output (LLM):**
```json
{
  "claim": "She attended Exeter College from 1990-1994",
  "subject": "She",
  "object": "Exeter College",
  "predicate": "attended",
  "confidence": 0.98
}
```

## Stage 3: Parsing & Deduplication

The `ExtractorController` merges structural and LLM claims, then:

1. Applies coreference resolution across all entities
2. Deduplicates claims (exact text match, case-insensitive)
3. Embeds nodes and builds chunk-node mappings

### Logging

Structural bridges are logged with the prefix:

```
Structural Bridges Injected: 3
  BRIDGE: Exeter College is part of University of Oxford
  BRIDGE: Jane Doe is CEO
```

## Installation

### With spaCy (Recommended)

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Without spaCy (Fallback)

The extractor will automatically fall back to heuristic-based analysis if spaCy is not installed.

## Testing the "Oxford Bridge" Fix

To verify that the hierarchical decomposition is working correctly, check the trace output:

```bash
grep "Structural Bridges Injected" multihop_output.trace.json
```

You should see bridge claims for all comma-separated entities with high confidence (0.95+).

## Performance Notes

- **Structural Analysis**: ~10-50ms per chunk (with spaCy)
- **LLM Claim Extraction**: ~500-2000ms per chunk (Qwen-14B)
- **Total Overhead**: <5% increase due to dependency parsing

## Future Enhancements

- [ ] Support for compound nouns via spaCy
- [ ] Genitive relationships ("Oxford's library")
- [ ] Cross-chunk entity linking
- [ ] Multilingual dependency parsing (Stanza)
