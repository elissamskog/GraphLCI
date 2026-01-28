"""
Node Embedder (v5)
Location: extractor/node_embedder.py
Owner: EXTRACTOR Instance

Embeds canonical entity/claim forms into vector space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from schemas.contracts import EmbeddedNode, EmbeddingClient


@dataclass
class NodeDescriptor:
    """Rich descriptor for a node to be embedded."""
    node_id: str
    node_type: str        # "ENTITY" or "CLAIM"
    canonical_text: str
    source_doc_id: str
    source_chunks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _embedding_text(descriptor: NodeDescriptor) -> str:
    """Build the text to embed for a descriptor.

    For ENTITY nodes with a non-empty description in metadata, produces
    ``"canonical_text: description"`` for richer embeddings.  All other
    nodes use the bare canonical_text.
    """
    if descriptor.node_type == "ENTITY":
        desc = descriptor.metadata.get("description")
        if desc:
            return f"{descriptor.canonical_text}: {desc}"
    return descriptor.canonical_text


def embed_nodes(
    descriptors: List[NodeDescriptor],
    client: EmbeddingClient,
) -> List[EmbeddedNode]:
    """Embed node descriptors into vector space.

    Parameters
    ----------
    descriptors:
        List of NodeDescriptor objects with metadata for embedding.
    client:
        Embedding client that implements the EmbeddingClient protocol.

    Returns
    -------
    List[EmbeddedNode]
        Nodes with embedding vectors and metadata.
    """
    if not descriptors:
        return []

    texts = [_embedding_text(d) for d in descriptors]
    embeddings = client.embed(texts)

    nodes: List[EmbeddedNode] = []
    for desc, embedding in zip(descriptors, embeddings):
        nodes.append(
            EmbeddedNode(
                node_id=desc.node_id,
                node_type=desc.node_type,
                canonical_text=desc.canonical_text,
                embedding=embedding,
                source_doc_id=desc.source_doc_id,
                source_chunks=desc.source_chunks,
                metadata=desc.metadata,
            )
        )

    return nodes
