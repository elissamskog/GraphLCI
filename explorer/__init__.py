"""
Explorer Package — Backward Compatibility Shim
Location: explorer/__init__.py

Re-exports from chunker/ and extractor/ packages.
The explorer/ package is kept for backward compatibility only.
New code should import from chunker/ and extractor/ directly.
"""

# Re-export chunker functions
from chunker.chunker import chunk_document

# Re-export extractor functions and classes
from extractor.entity_extractor import extract_entities
from extractor.claim_extractor import extract_claims
from extractor.gleaner import glean_missing, GleaningResult
from extractor.coreference_resolver import resolve_coreferences
from extractor.node_embedder import embed_nodes, NodeDescriptor
from extractor.json_utils import parse_json, parse_json_list
from extractor.summarizer import Summarizer
from extractor.domain_reasoner import DomainReasoner

__all__ = [
    "chunk_document",
    "extract_entities",
    "extract_claims",
    "glean_missing",
    "GleaningResult",
    "resolve_coreferences",
    "embed_nodes",
    "NodeDescriptor",
    "parse_json",
    "parse_json_list",
    "Summarizer",
    "DomainReasoner",
]
