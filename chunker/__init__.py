"""
CHUNKER Package
Location: chunker/__init__.py

Pure-Python document chunking — no LLM dependency.
"""

from chunker.chunker import chunk_document
from chunker.chunker_controller import ChunkerController

__all__ = ["chunk_document", "ChunkerController"]
