"""
Unit tests for reasoner/prompt_builder.py
Verifies context truncation and prompt construction logic.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reasoner.prompt_builder import build_hybrid_prompt
from schemas.contracts import (
    QueryRequest,
    FocusedContext,
    TargetedContext,
    RankedChunk,
    TextChunk,
    RawDocument,
    Scope
)

class TestPromptBuilder(unittest.TestCase):
    
    def test_truncation_of_edge_grammar_and_node_context(self):
        """Verify that edge_grammar and node_context are truncated if > 50 lines."""
        
        # Create long grammar and context (55 lines each)
        long_grammar = "\n".join([f"edge {i}" for i in range(55)])
        long_node_ctx = "\n".join([f"node {i}" for i in range(55)])
        
        focused = FocusedContext(
            documents={"d1": RawDocument(doc_id="d1", content="content", metadata={})},
            edge_grammar=long_grammar,
            node_context=long_node_ctx,
            scope=MagicMock(spec=Scope)
        )
        
        query = QueryRequest(query_id="q1", query_text="test query")
        
        prompt = build_hybrid_prompt(query, focused=focused)
        
        # Check for truncation markers
        self.assertIn("... (truncated)", prompt)
        # Check that we don't have the last line
        self.assertNotIn("edge 54", prompt)
        self.assertNotIn("node 54", prompt)
        # Check that we have the 50th line (index 49)
        self.assertIn("edge 49", prompt)
        self.assertIn("node 49", prompt)

    def test_truncation_of_document_content(self):
        """Verify that document content is truncated if > 2000 chars."""
        
        # Create long content (2500 chars)
        long_content = "a" * 2500
        
        doc = RawDocument(
            doc_id="doc1",
            content=long_content,
            metadata={}
        )
        
        focused = FocusedContext(
            documents={"doc1": doc},
            edge_grammar="edges",
            node_context="nodes",
            scope=MagicMock(spec=Scope)
        )
        
        query = QueryRequest(query_id="q1", query_text="test query")
        
        prompt = build_hybrid_prompt(query, focused=focused)
        
        # Check for truncation
        self.assertTrue(len(long_content) > 2000)
        expected_preview = long_content[:2000] + "..."
        self.assertIn(expected_preview, prompt)
        self.assertNotIn(long_content, prompt) # Full content shouldn't be there

    def test_targeted_chunks_limit(self):
        """Verify that only top 5 targeted chunks are included."""
        
        chunks = []
        for i in range(10):
            chunk = TextChunk(
                chunk_id=f"c{i}",
                source_doc_id="d1",
                text=f"chunk text {i}",
                token_count=10,
                position=i
            )
            ranked = RankedChunk(chunk=chunk, relevance_score=0.9 - (i * 0.01))
            chunks.append(ranked)
        
        targeted = TargetedContext(
            chunks=chunks,
            scope=MagicMock(spec=Scope)
        )
        
        query = QueryRequest(query_id="q1", query_text="test query")
        
        prompt = build_hybrid_prompt(query, targeted=targeted)
        
        # Verify top 5 are present
        for i in range(5):
            self.assertIn(f"[C{i+1}]", prompt)
            self.assertIn(f"chunk text {i}", prompt)
            
        # Verify 6th is NOT present
        self.assertNotIn("[C6]", prompt)
        self.assertNotIn("chunk text 5", prompt) # c5 is the 6th chunk

    def test_chunk_aliasing_and_counter(self):
        """Verify chunk counter increments correctly (no UnboundLocalError)."""
        
        chunk1 = TextChunk(chunk_id="c1", source_doc_id="d1", text="text1", token_count=10, position=0)
        chunk2 = TextChunk(chunk_id="c2", source_doc_id="d1", text="text2", token_count=10, position=1)
        
        ranked1 = RankedChunk(chunk=chunk1, relevance_score=0.9)
        ranked2 = RankedChunk(chunk=chunk2, relevance_score=0.8)
        
        targeted = TargetedContext(
            chunks=[ranked1, ranked2],
            scope=MagicMock(spec=Scope)
        )
        
        query = QueryRequest(query_id="q1", query_text="test query")
        
        prompt = build_hybrid_prompt(query, targeted=targeted)
        
        self.assertIn("[C1]", prompt)
        self.assertIn("text1", prompt)
        self.assertIn("[C2]", prompt)
        self.assertIn("text2", prompt)

if __name__ == "__main__":
    unittest.main()
