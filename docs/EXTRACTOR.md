# EXTRACTOR

## Purpose

Reads chunks of text and identifies the concrete concepts (entities) and factual assertions (claims) within them. Produces structured semantic building blocks — each tagged with where it came from and represented as a numeric vector for similarity comparison.

## Inputs

Text passages (chunks) split from source documents, with metadata about which document each chunk came from.

## Outputs

A collection of named entities (people, organizations, clauses, quantities), factual claims (who did what to whom), and numeric vector representations of each, with full traceability to source text.

## Key Decisions

- Which mentions are real entities vs. noise
- How to split complex sentences into atomic, self-contained claims
- When multiple mentions refer to the same thing (coreference)
- How many re-reads to perform to catch missed content
- Confidence rating for each extracted claim

## Failure Modes

- Invents entities or claims not present in the text (hallucination)
- Misses subtle or implicit information (under-extraction)
- Groups distinct concepts as the same entity, or fails to recognize duplicates
- Assigns inflated confidence to uncertain extractions
- Struggles with domain-specific terminology not seen in training
