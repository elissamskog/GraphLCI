"""
Chunker Controller
Location: chunker/chunker_controller.py
Owner: CHUNKER Instance

Orchestrates document chunking. Pure Python — no LLM dependency.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from schemas.contracts import ChunkerInput, ChunkerOutput, ChunkerTrace, ChunkSummary, TextChunk
from chunker.chunker import chunk_document


class ChunkerController:
    """Split documents into text chunks.

    Parameters
    ----------
    config:
        Chunker configuration dictionary (e.g. chunking target_size, overlap).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def process(self, chunker_input: ChunkerInput) -> ChunkerOutput:
        """Chunk all documents in the input.

        Parameters
        ----------
        chunker_input:
            Documents and optional config to chunk.

        Returns
        -------
        ChunkerOutput
            Chunks with source doc IDs and chunking config.
        """
        chunk_config = self._config.get("chunking", None)
        # Allow per-request config override
        if chunker_input.config and "chunking" in chunker_input.config:
            chunk_config = chunker_input.config["chunking"]

        all_chunks: List[TextChunk] = []
        source_doc_ids: List[str] = []

        for doc in chunker_input.documents:
            chunks = chunk_document(doc, chunk_config)
            all_chunks.extend(chunks)
            if doc.doc_id not in source_doc_ids:
                source_doc_ids.append(doc.doc_id)

        chunk_summaries = [
            ChunkSummary(
                chunk_id=c.chunk_id,
                source_doc_id=c.source_doc_id,
                token_count=c.token_count,
                position=c.position,
                section_header=c.section_header,
                text_preview=c.text[:300],
            )
            for c in all_chunks
        ]

        trace = ChunkerTrace(
            trace_id=str(uuid.uuid4()),
            input_doc_ids=list(source_doc_ids),
            total_chunks=len(all_chunks),
            chunks=chunk_summaries,
        )

        return ChunkerOutput(
            chunks=all_chunks,
            source_doc_ids=source_doc_ids,
            chunking_config=chunk_config,
            trace=trace,
        )
