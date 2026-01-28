# REASONER

## Purpose

Produces the final answer by reading assembled evidence and judging what it means. Determines how evidence relates to the query (supports, contradicts, refines, supersedes), grounds every claim in exact source text, and delivers a verdict with calibrated confidence.

## Inputs

The user query, scoped evidence from the planner (full documents, thematic summaries, or ranked passages depending on strategy), and the query's classified intent.

## Outputs

A verdict (confirmed, refuted, partially confirmed, inconclusive, or not found), epistemic relationships describing how evidence relates to the query, grounded citations with exact source quotes, a confidence score, and a reasoning trace.

## Key Decisions

- How each evidence piece relates to the query (epistemic relationship classification)
- How to weigh conflicting evidence when sources disagree
- Whether enough evidence exists to issue a strong verdict or only an inconclusive one
- When to say "not found" instead of speculating from thin evidence

## Failure Modes

- Infers relationships not supported by the text (hallucinated reasoning)
- Paraphrases instead of quoting exact source text, breaking the grounding chain
- Overconfident verdicts from thin evidence
- Cannot detect when the scope is missing critical evidence
- Struggles with nuanced partial support/contradiction distinctions
