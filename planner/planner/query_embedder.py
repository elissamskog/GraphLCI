"""
Query Embedder (v5)
Location: planner/planner/query_embedder.py
Owner: PLANNER Instance

Embeds user queries into vector space for seed node lookup.
"""

from __future__ import annotations

from typing import List

from schemas.contracts import EmbeddingClient


def embed_query(
    query_text: str,
    client: EmbeddingClient,
) -> List[float]:
    """Embed a query string into a vector.

    Parameters
    ----------
    query_text:
        The user's query text.
    client:
        Embedding client.

    Returns
    -------
    List[float]
        Query embedding vector.
    """
    results = client.embed([query_text])
    return results[0]


def embed_entities(
    entity_texts: List[str],
    client: EmbeddingClient,
) -> List[List[float]]:
    """Batch-embed entity surface forms into vectors.

    Parameters
    ----------
    entity_texts:
        List of entity surface form strings to embed.
    client:
        Embedding client.

    Returns
    -------
    List[List[float]]
        One embedding vector per entity text.
    """
    if not entity_texts:
        return []
    return client.embed(entity_texts)
