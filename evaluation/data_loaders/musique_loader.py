"""
MuSiQue Dataset Loader
Location: evaluation/data_loaders/musique_loader.py
Implements loading logic for MuSiQue multi-hop dataset.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Dict, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import random
from schemas.contracts import (
    RawCorpus,
    RawDocument,
)
from schemas.evaluation_contracts import (
    SciFactTestCase,
    SciFactVerdict,
)

_logger = logging.getLogger(__name__)

# Path to local MuSiQue file
_MUSIQUE_LOCAL_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "musique", "musique_ans_v1.0_dev.json")

def load_musique_data(
    num_distractors: int = 5,
    limit: int = 0
) -> Tuple[Dict[str, SciFactTestCase], Dict[str, RawDocument]]:
    """
    Load MuSiQue dataset from local JSON and format as SciFactTestCase.
    
    Args:
        num_distractors: Number of non-supporting paragraphs to include per case.
        limit: Max number of cases to load (0 for all). If > 0, selects random examples.
        
    Returns:
        tuple: (test_cases, corpus_docs)
    """
    _logger.info(f"Loading MuSiQue from {_MUSIQUE_LOCAL_PATH}...")
    
    if not os.path.exists(_MUSIQUE_LOCAL_PATH):
        error_msg = f"MuSiQue data not found at {_MUSIQUE_LOCAL_PATH}. Run scripts/download_musique.py first."
        _logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(_MUSIQUE_LOCAL_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        _logger.error(f"Failed to load dataset: {e}")
        raise e
    
    # Filter for answerable questions first
    valid_data = [row for row in data if row.get('answerable', True)]
    
    # Random sampling if limit is set
    if limit > 0 and limit < len(valid_data):
        _logger.info(f"Randomly sampling {limit} cases from {len(valid_data)} total...")
        selected_rows = random.sample(valid_data, limit)
    else:
        selected_rows = valid_data
        if limit > 0:
             selected_rows = selected_rows[:limit]
    
    test_cases: Dict[str, SciFactTestCase] = {}
    corpus_docs: Dict[str, RawDocument] = {}
    
    for row in selected_rows:
        case_id = str(row['id'])
        question = row['question']
        answer = row['answer']
        paragraphs = row['paragraphs']
        
        # Identify Gold and Distractors
        # MuSiQue 'is_supporting' bool flag
        gold_paras = [p for p in paragraphs if p.get('is_supporting')]
        distractor_paras = [p for p in paragraphs if not p.get('is_supporting')]
        
        # Truncate distractors
        if len(distractor_paras) > num_distractors:
            distractor_paras = distractor_paras[:num_distractors]
            
        selected_paragraphs = gold_paras + distractor_paras
        
        # Add to corpus
        claim_corpus_ids = []
        for p in selected_paragraphs:
            # Create a unique doc_id: {case_id}_{idx}
            idx = p['idx']
            doc_id = f"{case_id}_{idx}"
            claim_corpus_ids.append(doc_id)
            
            # Avoid overwriting if already exists (shouldn't happen with unique ID strategy)
            if doc_id not in corpus_docs:
                doc = RawDocument(
                    doc_id=doc_id,
                    content=p['paragraph_text'],
                    metadata={
                        "title": p['title'],
                        "is_supporting": p['is_supporting'],
                        "case_id": case_id
                    }
                )
                corpus_docs[doc_id] = doc
        
        # Construct TestCase
        # We assume Question -> Claim
        # Verdict -> SUPPORTS (since it's answerable and provided)
        # Store answer in metadata
        
        test_case = SciFactTestCase(
            claim_id=case_id, # MuSiQue IDs are strings, but SciFactTestCase might expect int?
                              # Looking at contracts.py: claim_id: int. 
                              # MuSiQue IDs seem to be '2hop_...' strings in some versions, but 'id' field in schema dump was '460946' (int looking).
                              # I'll try to convert to int or keep as str if contract allows (Python runtime ignore type hints).
                              # run_evaluation_v5 heavily relies on claim_id being string in EvalClaim, but SciFactLoader uses int.
                              # I'll check row['id'] type.
            claim_text=question,
            expected_verdict=SciFactVerdict.SUPPORTS,
            expected_evidence={}, # We don't have sentence-level labels easily, just paragraph level.
            cited_doc_ids=[f"{case_id}_{p['idx']}" for p in gold_paras],
            metadata={"answer": answer}
        )
        # Handle string IDs if necessary. MuSiQue IDs like '2hop_123' won't cast to int.
        # But EvalClaim expects string. SciFactTestCase type hint is int. 
        # I'll just pass the string. It's Python.
        
        
        test_cases[str(case_id)] = test_case
        # count += 1 (Removed count)
        
    return test_cases, corpus_docs
