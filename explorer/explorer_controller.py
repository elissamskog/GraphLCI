"""
Explorer Controller — Backward Compatibility Shim
Location: explorer/explorer_controller.py

Delegates to ChunkerController + ExtractorController.
Kept for backward compatibility — new code should use
chunker.chunker_controller.ChunkerController and
extractor.extractor_controller.ExtractorController directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.contracts import (
    ChunkerInput,
    ExplorationRequest,
    ExplorationResponse,
    ExplorationResult,
    ExplorerInput,
    ExplorerOutput,
    ExtractorInput,
)
from extractor.summarizer import Summarizer
from extractor.domain_reasoner import DomainReasoner
from chunker.chunker_controller import ChunkerController
from extractor.extractor_controller import ExtractorController


class ExplorerController:
    """Coordinate document summarization and domain taxonomy discovery.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    config:
        Explorer-specific configuration dictionary.
    """

    def __init__(self, llm_client: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self._llm_client = llm_client
        self._config = config or {}
        self._summarizer = Summarizer(llm_client)
        self._domain_reasoner = DomainReasoner(llm_client)

    def process(self, request: ExplorationRequest) -> ExplorationResponse:
        """Run the exploration pipeline over a corpus.

        Stage 1 - Summarize every document in the corpus.
        Stage 2 - Discover a domain taxonomy from the summaries.
        """
        try:
            # Stage 1: summarize each document
            summaries = [
                self._summarizer.summarize(doc)
                for doc in request.corpus.documents
            ]

            # Stage 2: discover taxonomy
            taxonomy = self._domain_reasoner.discover_taxonomy(summaries)

            result = ExplorationResult(
                corpus_id=request.corpus.corpus_id,
                summaries=summaries,
                taxonomy=taxonomy,
                exploration_metadata={
                    "document_count": len(summaries),
                    "domain_count": len(taxonomy.domains),
                    "relationship_count": len(taxonomy.cross_domain_relationships),
                },
            )

            return ExplorationResponse(success=True, result=result, errors=[])

        except Exception as e:
            return ExplorationResponse(success=False, result=None, errors=[str(e)])


class ExplorerControllerV5:
    """v5 EXPLORER pipeline — delegates to ChunkerController + ExtractorController.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    embedding_client:
        Object satisfying the ``EmbeddingClient`` protocol.
    config:
        Explorer-specific configuration dictionary.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_client: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._chunker = ChunkerController(config)
        self._extractor = ExtractorController(llm_client, embedding_client, config)

    def process(self, explorer_input: ExplorerInput) -> ExplorerOutput:
        """Run the v5 exploration pipeline by delegating to CHUNKER + EXTRACTOR."""
        chunker_output = self._chunker.process(
            ChunkerInput(
                documents=explorer_input.documents,
                config=explorer_input.config,
            )
        )
        extractor_input = ExtractorInput(
            chunks=chunker_output.chunks,
            source_doc_ids=chunker_output.source_doc_ids,
            config=explorer_input.config,
        )
        return self._extractor.process(extractor_input)
