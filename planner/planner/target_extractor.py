"""
Target Extractor Module
Responsibility: Identify documents and entities referenced in a query.

Dual-mode: LLM-based extraction with an offline rule-based fallback
for testing (same pattern as intent_classifier.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from schemas.contracts import IdentifiedTargets


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ExtractionInput:
    """Input to the target extractor."""
    query_text: str
    available_documents: List[str]
    known_entities: List[str]


@dataclass
class ExtractionOutput:
    """Output from the target extractor."""
    identified_targets: IdentifiedTargets
    ambiguities: List[str]


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

TARGET_EXTRACTION_PROMPT = """\
You are a target extractor for a legal document audit system.

TASK: Identify which documents and entities the user is asking about.

AVAILABLE DOCUMENTS IN CORPUS:
{available_documents}

KNOWN ENTITIES (parties, terms, clauses):
{known_entities}

QUERY: {query_text}

Extract:
1. EXPLICIT DOCUMENTS: Documents directly named or clearly referenced
2. IMPLICIT ENTITIES: Concepts, parties, or terms that imply specific documents

Respond in JSON format:
{{
  "explicit_docs": ["doc_id_1", "doc_id_2"],
  "implicit_entities": ["Liability Cap", "Vendor obligations"],
  "ambiguities": ["'the contract' could refer to MSA or SOW"],
  "confidence": 0.0-1.0
}}"""


# ============================================================================
# EXTRACTOR
# ============================================================================

# Common ambiguous references to detect in offline mode
_AMBIGUOUS_REFERENCES = [
    (r'\bthe contract\b', "'the contract' is ambiguous"),
    (r'\bthe agreement\b', "'the agreement' is ambiguous"),
    (r'\bthe document\b', "'the document' is ambiguous"),
    (r'\bthe policy\b', "'the policy' is ambiguous"),
    (r'\bthe addendum\b', "'the addendum' is ambiguous"),
    (r'\bthe exhibit\b', "'the exhibit' is ambiguous"),
    (r'\bthe schedule\b', "'the schedule' is ambiguous"),
]


class TargetExtractor:
    """
    Extracts document and entity targets from queries.

    In production, calls a Claude model with the prompt template.
    For testing/offline use, includes a rule-based fallback that
    performs word-boundary document matching and entity containment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def build_prompt(self, input_data: ExtractionInput) -> str:
        """Build the extraction prompt from the template."""
        return TARGET_EXTRACTION_PROMPT.format(
            query_text=input_data.query_text,
            available_documents=json.dumps(input_data.available_documents),
            known_entities=json.dumps(input_data.known_entities),
        )

    def extract(self, input_data: ExtractionInput) -> ExtractionOutput:
        """
        Extract targets from a query.

        Dispatches to LLM if configured, otherwise uses offline fallback.
        """
        llm_client = self.config.get('llm_client')
        if llm_client is not None:
            prompt = self.build_prompt(input_data)
            return self._extract_with_llm(llm_client, prompt)

        return self._extract_offline(input_data)

    def _extract_with_llm(
        self, llm_client: Any, prompt: str
    ) -> ExtractionOutput:
        """Send prompt to LLM and parse structured response."""
        raw_response = llm_client.generate(prompt)
        return self._parse_llm_response(raw_response)

    def _parse_llm_response(self, raw_response: str) -> ExtractionOutput:
        """Parse JSON response from LLM into ExtractionOutput."""
        json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if not json_match:
            return ExtractionOutput(
                identified_targets=IdentifiedTargets(
                    explicit_docs=[],
                    implicit_entities=[],
                ),
                ambiguities=["Failed to parse LLM response"],
            )

        data = json.loads(json_match.group())
        return ExtractionOutput(
            identified_targets=IdentifiedTargets(
                explicit_docs=data.get("explicit_docs", []),
                implicit_entities=data.get("implicit_entities", []),
            ),
            ambiguities=data.get("ambiguities", []),
        )

    def _extract_offline(self, input_data: ExtractionInput) -> ExtractionOutput:
        """
        Rule-based offline extraction for testing.

        - Match available_documents against query using word-boundary regex
        - Match known_entities against query using case-insensitive containment
        - Detect ambiguous references
        """
        query = input_data.query_text

        # Match documents using word-boundary regex
        matched_docs = []
        for doc_id in input_data.available_documents:
            pattern = r'\b' + re.escape(doc_id) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                matched_docs.append(doc_id)

        # Match entities using case-insensitive containment
        matched_entities = []
        query_lower = query.lower()
        for entity in input_data.known_entities:
            if entity.lower() in query_lower:
                matched_entities.append(entity)

        # Detect ambiguous references
        ambiguities = []
        for pattern, message in _AMBIGUOUS_REFERENCES:
            if re.search(pattern, query, re.IGNORECASE):
                ambiguities.append(message)

        return ExtractionOutput(
            identified_targets=IdentifiedTargets(
                explicit_docs=matched_docs,
                implicit_entities=matched_entities,
            ),
            ambiguities=ambiguities,
        )


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_target_extractor() -> None:
    """
    Test the target extractor with document matches, entity matches,
    no matches, ambiguity detection, multiple matches, and empty inputs.
    """
    extractor = TargetExtractor()

    available_docs = ["MSA", "DORA", "SOW-2024", "NDA", "Exhibit-A"]
    known_entities = ["Liability Cap", "Vendor", "Customer", "termination fee"]

    test_cases = [
        # 1. Exact document match
        {
            "name": "Exact document match",
            "query": "Does the MSA contradict DORA?",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": ["MSA", "DORA"],
            "expected_entities": [],
            "min_ambiguities": 0,
        },
        # 2. Entity match
        {
            "name": "Entity match",
            "query": "What is the Liability Cap?",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": [],
            "expected_entities": ["Liability Cap"],
            "min_ambiguities": 0,
        },
        # 3. No matches
        {
            "name": "No matches at all",
            "query": "Tell me about climate policy.",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": [],
            "expected_entities": [],
            "min_ambiguities": 0,
        },
        # 4. Ambiguity detection
        {
            "name": "Ambiguity detection",
            "query": "What does the contract say about termination fee?",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": [],
            "expected_entities": ["termination fee"],
            "min_ambiguities": 1,
        },
        # 5. Multiple document matches
        {
            "name": "Multiple document matches",
            "query": "Compare MSA, DORA, and SOW-2024 provisions.",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": ["MSA", "DORA", "SOW-2024"],
            "expected_entities": [],
            "min_ambiguities": 0,
        },
        # 6. Empty inputs
        {
            "name": "Empty available docs and entities",
            "query": "Does the MSA contradict DORA?",
            "available_documents": [],
            "known_entities": [],
            "expected_docs": [],
            "expected_entities": [],
            "min_ambiguities": 0,
        },
        # 7. Word-boundary prevents substring match
        {
            "name": "Word-boundary prevents substring match (NDA not in 'redundancy')",
            "query": "Check redundancy in Exhibit-A.",
            "available_documents": available_docs,
            "known_entities": known_entities,
            "expected_docs": ["Exhibit-A"],
            "expected_entities": [],
            "min_ambiguities": 0,
        },
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Target Extractor Test Suite")
    print("=" * 70)

    for case in test_cases:
        input_data = ExtractionInput(
            query_text=case["query"],
            available_documents=case["available_documents"],
            known_entities=case["known_entities"],
        )
        output = extractor.extract(input_data)

        docs_ok = sorted(output.identified_targets.explicit_docs) == sorted(case["expected_docs"])
        entities_ok = sorted(output.identified_targets.implicit_entities) == sorted(case["expected_entities"])
        ambiguity_ok = len(output.ambiguities) >= case["min_ambiguities"]

        all_ok = docs_ok and entities_ok and ambiguity_ok
        status = "PASS" if all_ok else "FAIL"

        if all_ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {case['name']}")
        print(f"  Query: \"{case['query']}\"")
        print(f"  Expected docs:     {case['expected_docs']}")
        print(f"  Got docs:          {output.identified_targets.explicit_docs}")
        print(f"  Expected entities: {case['expected_entities']}")
        print(f"  Got entities:      {output.identified_targets.implicit_entities}")
        print(f"  Ambiguities:       {output.ambiguities}")
        if not docs_ok:
            print("  ** DOCS MISMATCH")
        if not entities_ok:
            print("  ** ENTITIES MISMATCH")
        if not ambiguity_ok:
            print(f"  ** EXPECTED >= {case['min_ambiguities']} ambiguities")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    # Also test prompt generation
    print("\n--- Sample Prompt ---")
    sample = ExtractionInput(
        query_text="Does the MSA contradict DORA?",
        available_documents=["MSA", "DORA", "SOW"],
        known_entities=["Vendor", "Customer"],
    )
    prompt = extractor.build_prompt(sample)
    print(prompt[:500] + "...")

    assert failed == 0, f"{failed} test(s) failed"
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_target_extractor()
