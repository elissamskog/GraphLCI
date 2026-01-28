"""
Hybrid Synthesis Prompt Builder (v5.1)
Location: reasoner/prompt_builder.py
Owner: REASONER Instance (Epistemic Judge)
"""

from __future__ import annotations
from typing import Optional
from schemas.contracts import (
    FocusedContext,
    TargetedContext,
    QueryRequest,
)

_JSON_SCHEMA = """\
You MUST respond with valid JSON matching this exact schema:
{
  "relationships": [
    {
      "source_id": "<doc/node ID>",
      "target_id": "<doc/node ID>",
      "relationship": "<one of: SUPPORTS, STRONGLY_SUPPORTS, CONTRADICTS, STRONGLY_CONTRADICTS, PARTIALLY_SUPPORTS, PARTIALLY_CONTRADICTS, REFINES, EXTENDS, SUPERSEDES, PRECEDES, UNRELATED, INSUFFICIENT_EVIDENCE>",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<explanation>",
      "evidence_quote": "<exact quote from source>"
    }
  ],
  "evidence": [
    {
      "claim": "<claim text>",
      "source_document": "<doc_id>",
      "exact_quote": "<quote from source>"
    }
  ],
  "verdict": {
    "verdict_type": "<one of: CONFIRMED, REFUTED, PARTIALLY_CONFIRMED, PARTIALLY_REFUTED, INCONCLUSIVE, NOT_FOUND>",
    "answer": "<answer text>",
    "confidence": <float 0.0-1.0>,
    "summary": "<2-3 sentence summary>"
  },
  "reasoning_trace": "<step-by-step reasoning INCLUDING the Monologue requirement>"
}"""


def build_focused_prompt(query: QueryRequest, context: FocusedContext) -> str:
    """Build a prompt for Lane A (Graph-LCI) using focused context."""
    return build_hybrid_prompt(query, focused=context, targeted=None)


def build_targeted_prompt(query: QueryRequest, context: TargetedContext) -> str:
    """Build a prompt for Lane C (Vector-only) using ranked chunks."""
    return build_hybrid_prompt(query, focused=None, targeted=context)

def build_hybrid_prompt(
    query: QueryRequest,
    focused: Optional[FocusedContext] = None,
    targeted: Optional[TargetedContext] = None,
) -> str:
    """Build a hybrid prompt synthesizing structural (A) and semantic (C) context."""
    
    parts = [
        "You are an epistemic synthesis engine for auditing. "
        "Your task is to synthesize structural evidence from a knowledge graph "
        "with semantic facts from text chunks.",

        "\n### THE PRECEDENCE RULE\n"
        "If a conflict occurs, LANE A (Graph) defines the logical and temporal validity of a document, "
        "while LANE C (Vector) defines the literal wording.",

        "\n### ANALYTICAL DEDUCTION PROTOCOL\n"
        "1. SYNTHESIS AUTHORITY: You are authorized to synthesize information across multiple documents and nodes. "
        "A conclusion derived from the logical combination of multiple grounded facts is considered grounded evidence.\n"
        "2. METADATA PRIMACY: Treat structural metadata (timestamps, sequence numbers, and defined relationships in Lane A) as immutable. "
        "If metadata establishes a clear logical or temporal sequence, use it to resolve comparisons even if no single text chunk explicitly describes the relationship.\n"
        "3. DEDUCTIVE VERDICTS: Do not return 'INCONCLUSIVE' if the provided structural and semantic evidence contains the necessary data to perform a mathematical or logical deduction.",

        "\n### REASONING MONOLOGUE\n"
        "Before issuing a verdict, you must explicitly state if the structural path in the graph (A) "
        "supports or refutes the relevance of the semantic chunk (C). This must be included in your reasoning_trace.",

        "\nIMPORTANT: Base your analysis ONLY on the evidence provided below. "
        "If the provided context does not address the claim, return verdict_type 'NOT_FOUND' with confidence 0.0."
    ]

    # Stream 1: Structural (Lane A)
    if focused:
        parts.append("\n\n## STREAM 1: STRUCTURAL (LANE A)")
        parts.append(f"Edge Grammar:\n{focused.edge_grammar}")
        parts.append(f"Node Context:\n{focused.node_context}")
        parts.append("\nDocuments:")
        for doc_id, doc in focused.documents.items():
            parts.append(f"\n[Document: {doc_id}]\n{doc.content}")

    # Stream 2: Semantic (Lane C)
    if targeted:
        parts.append("\n\n## STREAM 2: SEMANTIC (LANE C)")
        for ranked in targeted.chunks:
            chunk = ranked.chunk
            parts.append(
                f"\n[Chunk: {chunk.chunk_id} | Source: {chunk.source_doc_id} | "
                f"Relevance: {ranked.relevance_score:.3f}]\n{chunk.text}"
            )

    # Output format
    parts.append(f"\n\n--- OUTPUT FORMAT ---\n{_JSON_SCHEMA}")

    # Query
    parts.append(f"\n\n--- QUERY ---\n{query.query_text}")

    return "\n".join(parts)