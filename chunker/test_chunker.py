"""
Chunker Tests
Location: chunker/test_chunker.py
Owner: CHUNKER Instance

Unit tests for the CHUNKER module.
"""

from __future__ import annotations

import sys
import os
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.contracts import ChunkerInput, RawDocument
from chunker.chunker import chunk_document
from chunker.chunker_controller import ChunkerController


# ============================================================================
# TEST CASES
# ============================================================================

class TestChunkDocumentBasic(unittest.TestCase):
    """Short document produces a single chunk."""

    def test_chunk_document_basic(self):
        doc = RawDocument(
            doc_id="doc_001",
            content="This is a short document with a few words.",
            metadata={},
        )
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "doc_001_chunk_0000")
        self.assertEqual(chunks[0].source_doc_id, "doc_001")
        self.assertIn("short document", chunks[0].text)
        self.assertGreater(chunks[0].token_count, 0)


class TestChunkDocumentSplits(unittest.TestCase):
    """Long document produces multiple chunks with overlap."""

    def test_chunk_document_splits(self):
        paragraph = "This is a test sentence with multiple words. " * 200
        content = paragraph + "\n\n" + paragraph
        doc = RawDocument(doc_id="doc_002", content=content, metadata={})

        chunks = chunk_document(doc, config={"target_size": 300, "overlap": 50})
        self.assertGreater(len(chunks), 1)

        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"doc_002_chunk_{i:04d}")
            self.assertEqual(chunk.position, i)

        # Verify overlap: later chunks should share words with earlier ones
        if len(chunks) >= 2:
            first_words = set(chunks[0].text.split()[-50:])
            second_words = set(chunks[1].text.split()[:50])
            overlap = first_words & second_words
            self.assertGreater(len(overlap), 0, "Expected overlap between chunks")


class TestChunkDocumentEmpty(unittest.TestCase):
    """Empty document produces no chunks."""

    def test_empty_content(self):
        doc = RawDocument(doc_id="doc_empty", content="", metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 0)

    def test_whitespace_content(self):
        doc = RawDocument(doc_id="doc_ws", content="   \n\n  ", metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 0)


class TestChunkerController(unittest.TestCase):
    """ChunkerController processes multiple documents."""

    def test_process_multiple_docs(self):
        docs = [
            RawDocument(
                doc_id="doc_a",
                content="Document A has some content here.",
                metadata={},
            ),
            RawDocument(
                doc_id="doc_b",
                content="Document B has different content.",
                metadata={},
            ),
        ]
        controller = ChunkerController()
        result = controller.process(ChunkerInput(documents=docs))

        self.assertGreater(len(result.chunks), 0)
        self.assertIn("doc_a", result.source_doc_ids)
        self.assertIn("doc_b", result.source_doc_ids)
        self.assertEqual(len(result.source_doc_ids), 2)

    def test_process_with_config(self):
        docs = [
            RawDocument(
                doc_id="doc_c",
                content="A short document.",
                metadata={},
            ),
        ]
        controller = ChunkerController(config={"chunking": {"target_size": 500}})
        result = controller.process(ChunkerInput(documents=docs))

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.source_doc_ids, ["doc_c"])


class TestSectionHeaderDetection(unittest.TestCase):
    """Section header detection in chunk text."""

    def test_section_header_detected(self):
        content = "Section 1 Introduction\n\nThis section introduces the topic."
        doc = RawDocument(doc_id="doc_hdr", content=content, metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section_header, "Section 1")

    def test_article_header_detected(self):
        content = "Article 42 Definitions\n\nThis article defines key terms."
        doc = RawDocument(doc_id="doc_art", content=content, metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(chunks[0].section_header, "Article 42")

    def test_chapter_header_detected(self):
        content = "Chapter 3 Scope\n\nThe scope of this regulation."
        doc = RawDocument(doc_id="doc_ch", content=content, metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(chunks[0].section_header, "Chapter 3")

    def test_no_header_when_absent(self):
        content = "This document has no section header at the start."
        doc = RawDocument(doc_id="doc_nohdr", content=content, metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].section_header)

    def test_header_case_insensitive(self):
        content = "SCHEDULE 5 Fees\n\nFee schedule details."
        doc = RawDocument(doc_id="doc_case", content=content, metadata={})
        chunks = chunk_document(doc)
        self.assertEqual(chunks[0].section_header, "SCHEDULE 5")


class TestTraceGeneration(unittest.TestCase):
    """ChunkerController generates correct trace and ChunkSummary."""

    def test_trace_fields(self):
        docs = [
            RawDocument(doc_id="doc_t1", content="Some content here.", metadata={}),
            RawDocument(doc_id="doc_t2", content="More content here.", metadata={}),
        ]
        controller = ChunkerController()
        result = controller.process(ChunkerInput(documents=docs))

        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertIsInstance(trace.trace_id, str)
        self.assertGreater(len(trace.trace_id), 0)
        self.assertEqual(set(trace.input_doc_ids), {"doc_t1", "doc_t2"})
        self.assertEqual(trace.total_chunks, len(result.chunks))

    def test_chunk_summary_fields(self):
        content = "A" * 500
        docs = [RawDocument(doc_id="doc_sum", content=content, metadata={})]
        controller = ChunkerController()
        result = controller.process(ChunkerInput(documents=docs))

        trace = result.trace
        self.assertIsNotNone(trace.chunks)
        self.assertEqual(len(trace.chunks), len(result.chunks))

        summary = trace.chunks[0]
        self.assertEqual(summary.chunk_id, result.chunks[0].chunk_id)
        self.assertEqual(summary.source_doc_id, "doc_sum")
        self.assertEqual(summary.token_count, result.chunks[0].token_count)
        self.assertEqual(summary.position, 0)

    def test_text_preview_truncated_at_300(self):
        content = "word " * 200  # 1000 chars
        docs = [RawDocument(doc_id="doc_prev", content=content, metadata={})]
        controller = ChunkerController()
        result = controller.process(ChunkerInput(documents=docs))

        summary = result.trace.chunks[0]
        self.assertLessEqual(len(summary.text_preview), 300)
        self.assertEqual(summary.text_preview, result.chunks[0].text[:300])


class TestConfigOverride(unittest.TestCase):
    """Per-request config overrides instance-level config."""

    def test_per_request_overrides_instance(self):
        long_content = "This is a test sentence. " * 500
        docs = [RawDocument(doc_id="doc_cfg", content=long_content, metadata={})]

        # Instance config: large target → fewer chunks
        controller = ChunkerController(config={"chunking": {"target_size": 2400, "overlap": 50}})
        result_default = controller.process(ChunkerInput(documents=docs))

        # Per-request config: small target → more chunks
        result_override = controller.process(
            ChunkerInput(
                documents=docs,
                config={"chunking": {"target_size": 300, "overlap": 20}},
            )
        )

        self.assertGreater(
            len(result_override.chunks),
            len(result_default.chunks),
            "Per-request smaller target_size should produce more chunks",
        )

    def test_instance_config_used_when_no_override(self):
        long_content = "Sentence for testing. " * 300
        docs = [RawDocument(doc_id="doc_inst", content=long_content, metadata={})]

        ctrl_small = ChunkerController(config={"chunking": {"target_size": 200, "overlap": 10}})
        ctrl_large = ChunkerController(config={"chunking": {"target_size": 2400, "overlap": 10}})

        result_small = ctrl_small.process(ChunkerInput(documents=docs))
        result_large = ctrl_large.process(ChunkerInput(documents=docs))

        self.assertGreater(
            len(result_small.chunks),
            len(result_large.chunks),
            "Smaller instance target_size should produce more chunks",
        )


class TestSingleGiantParagraph(unittest.TestCase):
    """Edge case: one paragraph that exceeds target_size with no blank lines."""

    def test_giant_paragraph_splits_on_sentences(self):
        # A single paragraph (no double-newlines) much larger than target
        giant = "This is sentence number one. " * 500
        doc = RawDocument(doc_id="doc_giant", content=giant.strip(), metadata={})
        chunks = chunk_document(doc, config={"target_size": 200, "overlap": 20})

        self.assertGreater(len(chunks), 1, "Giant paragraph should be split into multiple chunks")
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.position, i)
            self.assertEqual(chunk.source_doc_id, "doc_giant")


if __name__ == "__main__":
    unittest.main()
