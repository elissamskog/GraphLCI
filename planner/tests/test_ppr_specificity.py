
import sys
import os
import math
from typing import Dict, List

# Ensure project root and planner dir are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas.contracts import (
    EmbeddedNode,
    KnowledgeGraph,
    StructuralEdge,
)
from planner.ppr_expander import personalized_pagerank

def _make_node(node_id: str, chunk_count: int) -> EmbeddedNode:
    """Create a dummy node with a specific number of source chunks."""
    # Create distinct chunk IDs
    chunks = [f"chk_{node_id}_{i}" for i in range(chunk_count)]
    return EmbeddedNode(
        node_id=node_id,
        node_type="ENTITY",
        canonical_text=f"Node {node_id}",
        embedding=[], 
        source_doc_id="DOC_1",
        source_chunks=chunks,
        metadata={}
    )

def test_specificity_weighting():
    """
    Verify that nodes with fewer source chunks (rarer) get higher impact 
    in the personalization vector.
    """
    # Create two nodes
    # Rare: 1 chunk
    rare_node = _make_node("rare", 1)
    # Common: 10 chunks
    common_node = _make_node("common", 100)
    
    nodes = {
        "rare": rare_node,
        "common": common_node
    }
    
    # Graph with just these nodes, no edges needed for this specific test 
    # as we are testing the initialization (personalization vector) impact.
    # But to see propagation, let's just checking the scores after 0 iterations 
    # (which should be exactly the personalization vector) or 1 iteration.
    
    # Actually, we can check 0 iterations to see the raw personalization distribution 
    # OR we can run normal PPR. If there are no edges, PPR converges to personalization vector.
    graph = KnowledgeGraph(
        nodes=nodes,
        edges={},
        adjacency={},
        node_by_doc={},
        node_count=2
    )
    
    seeds = ["rare", "common"]
    
    # Run PPR
    # With specificity, we expect weight ~ 1/1 vs 1/100
    # So Rare should be ~100x Common.
    scores = personalized_pagerank(graph, seeds, max_iterations=1)
    
    print(f"Rare Score: {scores['rare']}")
    print(f"Common Score: {scores['common']}")
    
    # Without specificity, they would be equal (0.5 each).
    # With specificity:
    # w_rare = 1/1 = 1.0
    # w_common = 1/100 = 0.01
    # total = 1.01
    # rare_norm = 1.0 / 1.01 ~= 0.99
    # common_norm = 0.01 / 1.01 ~= 0.0099
    
    assert scores['rare'] > scores['common'], "Rare node should have higher score"
    ratio = scores['rare'] / scores['common']
    print(f"Ratio: {ratio}")
    
    assert ratio > 50.0, f"Expected ratio > 50 (approx 100), got {ratio}"

if __name__ == "__main__":
    try:
        test_specificity_weighting()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
