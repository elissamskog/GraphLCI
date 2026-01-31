"""
Hybrid Synthesis Prompt Builder (v5.1)
Location: reasoner/prompt_builder.py
Owner: REASONER Instance (Epistemic Judge)
"""

from __future__ import annotations
from typing import Dict, Optional
from schemas.contracts import (
    AliasMap,
    FocusedContext,
    TargetedContext,
    QueryRequest,
)

_JSON_SCHEMA = """\
You MUST respond with valid JSON matching this exact schema:
{
  "relationships": [
    {
      "source_id": "<alias like [D1] or doc_id>",
      "target_id": "<alias like [D2] or doc_id>",
      "relationship": "<one of: SUPPORTS, STRONGLY_SUPPORTS, CONTRADICTS, STRONGLY_CONTRADICTS, PARTIALLY_SUPPORTS, PARTIALLY_CONTRADICTS, REFINES, EXTENDS, SUPERSEDES, PRECEDES, UNRELATED, INSUFFICIENT_EVIDENCE>",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<explanation>",
      "evidence_quote": "<exact quote from source>"
    }
  ],
  "evidence": [
    {
      "claim": "<claim text>",
      "source_document": "<REQUIRED: alias like [D1] — DO NOT leave empty>",
      "exact_quote": "<quote from source INCLUDING the [D#] tag at the end>"
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


def build_focused_prompt(
    query: QueryRequest,
    context: FocusedContext,
    alias_map: Optional[AliasMap] = None,
) -> str:
    """Build a prompt for Lane A (Graph-LCI) using focused context."""
    return build_hybrid_prompt(query, focused=context, targeted=None, alias_map=alias_map)


def build_targeted_prompt(
    query: QueryRequest,
    context: TargetedContext,
    alias_map: Optional[AliasMap] = None,
) -> str:
    """Build a prompt for Lane C (Vector-only) using ranked chunks."""
    return build_hybrid_prompt(query, focused=None, targeted=context, alias_map=alias_map)


def build_hybrid_prompt(
    query: QueryRequest,
    focused: Optional[FocusedContext] = None,
    targeted: Optional[TargetedContext] = None,
    alias_map: Optional[AliasMap] = None,
) -> str:
    """Build a hybrid prompt synthesizing structural (A) and semantic (C) context.
    
    If alias_map is provided, documents and chunks are labeled with short aliases
    like [D1], [D2] for easier LLM citation.
    """
    
    # Build reverse lookup: doc_id -> alias
    doc_to_alias: Dict[str, str] = {}
    if alias_map:
        for alias, aliased_doc in alias_map.mappings.items():
            doc_to_alias[aliased_doc.doc_id] = alias
    
    parts = [
        "You are an epistemic synthesis engine for auditing. "
        "Your task is to synthesize evidence from a knowledge graph "
        "with semantic facts from text chunks.",

        "\n### CITATION INSTRUCTIONS\n"
        "Each document and chunk is labeled with a short alias (e.g., [D1], [D2], [C1], [C2]).\n"
        "When citing evidence, use these short aliases.\n"
        "Example: \"The magazine started in 1844 [D1].\"\n",

        "\n### THE PRECEDENCE RULE\n"
        "If a conflict occurs, LANE A (Graph) defines the logical and temporal validity of a document, "
        "while LANE C (Vector) defines the literal wording.",

        "\n### ANALYTICAL DEDUCTION PROTOCOL\n"
        "1. SYNTHESIS AUTHORITY: You are authorized to synthesize information across multiple documents and nodes. "
        "A conclusion derived from the logical combination of multiple facts is considered evidence.\n"
        "2. METADATA PRIMACY: Treat structural metadata (timestamps, sequence numbers, and defined relationships in Lane A) as immutable. "
        "If metadata establishes a clear logical or temporal sequence, use it to resolve comparisons even if no single text chunk explicitly describes the relationship.\n"
        "3. DEDUCTIVE VERDICTS: Do not return 'INCONCLUSIVE' if the provided evidence contains the necessary data to perform a mathematical or logical deduction.\n"
        "4. TRANSITIVE INFERENCE: You MUST perform multi-step transitive deduction. (Example: If A is on B, and B flows into C, then A is on a tributary of C). Do not demand a single sentence stating the full chain.\n"
        "5. ENTITY TYPE RELAXATION: Prioritize logical relationships over strict entity type matching. If the question asks for a 'river' but the answer is a 'bay' or 'lake' that fulfills the 'tributary' relationship, accept it. Explain the discrepancy in the reasoning trace.",

        "\n### REASONING MONOLOGUE\n"
        "Before issuing a verdict, you must explicitly state if the structural path in the graph (A) "
        "supports or refutes the relevance of the semantic chunk (C). This must be included in your reasoning_trace.",

        "\n### VERDICT DEFINITIONS\n"
        "- CONFIRMED: The claim is fully supported by the evidence (direct or logically deduced).\n"
        "- REFUTED: The claim is contradicted by the evidence (e.g. dates do not match, events didn't happen as stated).\n"
        "- PARTIALLY_CONFIRMED: The core claim is true or logically inferred, but some details are missing or minor inaccuracies exist.\n"
        "- PARTIALLY_REFUTED: The core claim is false, but some parts are true.\n"
        "- INCONCLUSIVE: Evidence is conflicting or insufficient to make a determination.\n"
        "- NOT_FOUND: The entities or events mentioned in the claim are not present in the provided evidence.",

        "\nIMPORTANT: Base your analysis ONLY on the evidence provided below. "
        "If the provided evidence does not address the claim, return verdict_type 'NOT_FOUND' with confidence 0.0."
    ]

    # --- Stream 1: Structural (Lane A) ---
    if focused and focused.documents:
        parts.append("\n\n## STREAM 1: STRUCTURAL (LANE A)")
        
        # Truncate edge grammar and node context to prevent overflow
        edge_lines = focused.edge_grammar.split('\n')
        if len(edge_lines) > 50:
            edge_grammar_safe = "\n".join(edge_lines[:50]) + "\n... (truncated)"
        else:
            edge_grammar_safe = focused.edge_grammar
            
        node_lines = focused.node_context.split('\n')
        if len(node_lines) > 50:
            node_context_safe = "\n".join(node_lines[:50]) + "\n... (truncated)"
        else:
            node_context_safe = focused.node_context

        parts.append(f"Edge Grammar:\n{edge_grammar_safe}")
        parts.append(f"Node Context:\n{node_context_safe}")
        parts.append("\nDocuments:")
        
        for doc_id, doc in focused.documents.items():
            alias = doc_to_alias.get(doc_id, doc_id)
            # Truncate document content to ~2000 chars (approx 500 tokens)
            content_preview = doc.content[:2000] + "..." if len(doc.content) > 2000 else doc.content
            parts.append(f"\n{alias} (doc_id: {doc_id})\n{content_preview}")

    # --- Stream 2: Semantic (Lane C) ---
    if targeted and targeted.chunks:
        parts.append("\n\n## STREAM 2: SEMANTIC (LANE C)")
        chunk_counter = 1
        # Limit to top 5 chunks to save context
        for ranked in targeted.chunks[:5]:
            chunk = ranked.chunk
            chunk_alias = f"[C{chunk_counter}]"
            chunk_counter += 1
            parts.append(
                f"\n{chunk_alias} (chunk_id: {chunk.chunk_id} | source: {chunk.source_doc_id} | "
                f"relevance: {ranked.relevance_score:.3f})\n{chunk.text}"
            )

        
    # Output format
    parts.append(f"\n\n--- OUTPUT FORMAT ---\n{_JSON_SCHEMA}")

    # Query
    parts.append(f"\n\n--- QUERY ---\n{query.query_text}")

    return "\n".join(parts)