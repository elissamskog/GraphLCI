"""
Prompt Builder (v5)
Location: reasoner/prompt_builder.py
Owner: REASONER Instance

Builds LLM prompts from assembled context for each lane type.
"""

from __future__ import annotations

from schemas.contracts import (
    FocusedContext,
    GlobalContext,
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
  "reasoning_trace": "<step-by-step reasoning>"
}"""


def build_focused_prompt(query: QueryRequest, context: FocusedContext) -> str:
    """Build a prompt for Lane A (Graph-LCI) using focused context.

    Includes edge grammar, node context, and full documents.
    """
    parts = [
        "You are an epistemic analysis engine for legal document auditing. "
        "Your task is to analyze relationships between documents using the "
        "provided graph structure and full document content. Identify "
        "epistemic relationships (support, contradiction, extension, etc.) "
        "between source documents, ground your findings in exact quotes, "
        "and produce a verdict."
        "\n\nIMPORTANT: Base your analysis ONLY on the documents and structural "
        "context provided above. Do not use information from the query itself as "
        "evidence. If the provided documents do not address the claim, return "
        'verdict_type "NOT_FOUND" with confidence 0.0.',
    ]

    # Structural context
    parts.append("\n\n--- STRUCTURAL CONTEXT ---")
    parts.append(f"\nEdge Grammar:\n{context.edge_grammar}")
    parts.append(f"\nNode Context:\n{context.node_context}")

    # Documents
    parts.append("\n\n--- DOCUMENTS ---")
    for doc_id, doc in context.documents.items():
        parts.append(f"\n[Document: {doc_id}]\n{doc.content}")

    # Output format
    parts.append(f"\n\n--- OUTPUT FORMAT ---\n{_JSON_SCHEMA}")

    # Query
    parts.append(f"\n\n--- QUERY ---\n{query.query_text}")

    return "\n".join(parts)


def build_global_prompt(query: QueryRequest, context: GlobalContext) -> str:
    """Build a prompt for Lane B (Graph-only) using community summaries."""
    parts = [
        "You are an epistemic synthesis engine for legal document auditing. "
        "Your task is to synthesize knowledge from community-level summaries "
        "of a document graph. Identify overarching themes, tensions between "
        "communities, and produce a synthesized verdict. Focus on high-level "
        "patterns and cross-community relationships."
        "\n\nIMPORTANT: Base your analysis ONLY on the community summaries "
        "provided above. Do not use information from the query itself as "
        "evidence. If the provided summaries do not address the claim, return "
        'verdict_type "NOT_FOUND" with confidence 0.0.',
    ]

    # Community summaries
    parts.append("\n\n--- COMMUNITY SUMMARIES ---")
    for community_id, summary in context.community_summaries.items():
        parts.append(f"\n[Community: {community_id}]\n{summary}")

    # Output format
    parts.append(f"\n\n--- OUTPUT FORMAT ---\n{_JSON_SCHEMA}")

    # Query
    parts.append(f"\n\n--- QUERY ---\n{query.query_text}")

    return "\n".join(parts)


def build_targeted_prompt(query: QueryRequest, context: TargetedContext) -> str:
    """Build a prompt for Lane C (Vector-only) using ranked chunks."""
    parts = [
        "You are an evidence lookup engine for legal document auditing. "
        "Your task is to find specific information in the provided text chunks, "
        "ranked by relevance. Cite your sources precisely with chunk IDs and "
        "exact quotes. Ground every claim in the provided text."
        "\n\nIMPORTANT: Base your analysis ONLY on the text chunks provided "
        "above. Do not use information from the query itself as evidence. If no "
        "chunks are provided or the chunks do not address the claim, return "
        'verdict_type "NOT_FOUND" with confidence 0.0.',
    ]

    # Ranked chunks
    parts.append("\n\n--- RANKED CHUNKS ---")
    for ranked in context.chunks:
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
