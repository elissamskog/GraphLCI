import os
import sys
import logging
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from llm_client import get_client
from planner.planner.claim_decomposer import decompose_claim

logging.basicConfig(level=logging.INFO)

def test_decomposition():
    llm_client = get_client()
    query = "Who is the spouse of the Green performer?"
    config = {"enable_decomposition": True, "min_entities_for_decomposition": 2}
    
    print(f"\nDecomposing: {query}")
    decomposition = decompose_claim(query, llm_client, config)
    
    print("\nResult:")
    print(f"Should Decompose: {decomposition.should_decompose}")
    print(f"Primary Subject: {decomposition.primary_subject}")
    print(f"Confidence: {decomposition.confidence}")
    print(f"Reasoning: {decomposition.decomposition_reasoning}")
    print(f"Subclaims: {len(decomposition.subclaims)}")
    for sc in decomposition.subclaims:
        print(f"  - [{sc.priority}] {sc.objective} (Entities: {sc.entities})")

if __name__ == "__main__":
    test_decomposition()
