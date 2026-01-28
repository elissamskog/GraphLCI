"""
Similarity Index (v5)
Location: librarian/similarity_index.py
Owner: LIBRARIAN Instance

Vector similarity index for fast nearest-neighbor lookups.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from schemas.contracts import EmbeddedNode


class SimilarityIndex:
    """In-memory similarity index over embedded nodes.

    Supports adding nodes and querying for nearest neighbors
    by cosine similarity.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def add_nodes(self, nodes: Dict[str, EmbeddedNode]) -> None:
        """Add embedded nodes to the index.

        Parameters
        ----------
        nodes:
            Nodes keyed by node_id with embedding vectors.
        """
        if not nodes:
            return

        embeddings: List[List[float]] = []
        for node_id, node in nodes.items():
            if not node.embedding:
                continue
            self._id_to_idx[node_id] = len(self._ids)
            self._ids.append(node_id)
            embeddings.append(node.embedding)

        if not embeddings:
            return

        mat = np.array(embeddings, dtype=np.float32)
        # L2-normalize each row
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms

        self._matrix = mat

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Find nearest neighbors by cosine similarity.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        top_k:
            Maximum number of results.
        threshold:
            Minimum similarity threshold.

        Returns
        -------
        List[Tuple[str, float]]
            List of (node_id, similarity) pairs, sorted descending.
        """
        if self._matrix is None or len(self._ids) == 0:
            return []

        query_vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        # Cosine similarity = dot product for unit vectors
        similarities = self._matrix @ query_vec

        # Sort descending
        sorted_indices = np.argsort(-similarities)

        results: List[Tuple[str, float]] = []
        for idx in sorted_indices:
            sim = float(similarities[idx])
            if threshold is not None and sim < threshold:
                continue
            results.append((self._ids[idx], sim))
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        """Persist the index to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.savez(str(p / "matrix.npz"), matrix=self._matrix)
        with open(p / "ids.json", "w") as f:
            json.dump(self._ids, f)

    @classmethod
    def load(cls, path: str) -> SimilarityIndex:
        """Load an index from disk."""
        p = Path(path)
        instance = cls()
        with open(p / "ids.json") as f:
            instance._ids = json.load(f)
        instance._id_to_idx = {nid: i for i, nid in enumerate(instance._ids)}
        npz_path = p / "matrix.npz"
        if npz_path.exists():
            data = np.load(str(npz_path))
            instance._matrix = data["matrix"]
        return instance
