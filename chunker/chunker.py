"""
Document Chunker (v5)
Location: chunker/chunker.py
Owner: CHUNKER Instance

Splits raw documents into token-bounded text chunks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from schemas.contracts import RawDocument, TextChunk

# Regex for detecting section headers
_SECTION_RE = re.compile(
    r"^(?:Section|Article|Part|Chapter|Clause|Schedule|Appendix|Annex)\s+\d+",
    re.IGNORECASE,
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using word splitting (word ~ token)."""
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at period/question/exclamation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _detect_section_header(text: str) -> Optional[str]:
    """Return the section header if the text starts with one, else None."""
    first_line = text.strip().split("\n", 1)[0]
    match = _SECTION_RE.match(first_line.strip())
    return match.group(0) if match else None


def chunk_document(
    doc: RawDocument,
    config: Optional[Dict[str, Any]] = None,
) -> List[TextChunk]:
    """Split a document into text chunks.

    Parameters
    ----------
    doc:
        The raw document to chunk.
    config:
        Optional chunking configuration (e.g. target_size, overlap).

    Returns
    -------
    List[TextChunk]
        Ordered list of text chunks from the document.
    """
    config = config or {}
    target_size = config.get("target_size", 1200)
    overlap = config.get("overlap", 100)
    strategy = config.get("strategy", "semantic")  # forward-compat; only "semantic" implemented

    text = doc.content
    if not text or not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Break paragraphs that exceed target into sentences
    segments: List[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) > target_size:
            sentences = _split_sentences(para)
            segments.extend(sentences)
        else:
            segments.append(para)

    # Greedily accumulate segments into chunks
    chunks: List[TextChunk] = []
    current_segments: List[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _estimate_tokens(seg)

        if current_tokens + seg_tokens > target_size and current_segments:
            # Emit current chunk
            chunk_text = "\n\n".join(current_segments)
            _emit_chunk(chunks, doc, chunk_text, len(chunks))

            # Build overlap prefix from tail of current segments
            overlap_text = _build_overlap(current_segments, overlap)
            current_segments = []
            current_tokens = 0
            if overlap_text:
                current_segments.append(overlap_text)
                current_tokens = _estimate_tokens(overlap_text)

        current_segments.append(seg)
        current_tokens += seg_tokens

    # Emit final chunk
    if current_segments:
        chunk_text = "\n\n".join(current_segments)
        _emit_chunk(chunks, doc, chunk_text, len(chunks))

    return chunks


def _build_overlap(segments: List[str], overlap_tokens: int) -> str:
    """Build overlap text from the tail of segments."""
    if overlap_tokens <= 0:
        return ""

    # Walk backwards through segments collecting overlap words
    words: List[str] = []
    for seg in reversed(segments):
        seg_words = seg.split()
        words = seg_words + words
        if len(words) >= overlap_tokens:
            break

    # Take last overlap_tokens words
    overlap_words = words[-overlap_tokens:]
    return " ".join(overlap_words) if overlap_words else ""


def _emit_chunk(
    chunks: List[TextChunk],
    doc: RawDocument,
    text: str,
    position: int,
) -> None:
    """Create a TextChunk and append it to the list."""
    chunk_id = f"{doc.doc_id}_chunk_{position:04d}"
    section = _detect_section_header(text)
    chunks.append(
        TextChunk(
            chunk_id=chunk_id,
            source_doc_id=doc.doc_id,
            text=text,
            token_count=_estimate_tokens(text),
            position=position,
            section_header=section,
        )
    )
