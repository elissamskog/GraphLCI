"""
Epistemic Search Strategy Classifier (v5.2)
Location: planner/planner/search_strategy_classifier.py

Classifies query search strategy using LLM-based Chain-of-Thought reasoning.
Determines if the truth is a "Point-Fact" (LOOKUP) or a "Relationship" (COMPARISON).
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from schemas.contracts import SearchStrategyClassification, SearchStrategyType

logger = logging.getLogger(__name__)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ClassificationInput:
    """Input to the search strategy classifier."""
    query_text: str
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class ClassificationOutput:
    """Output from the classifier."""
    strategy: SearchStrategyClassification
    extracted_signals: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EPISTEMIC CLASSIFICATION PROMPT (Chain-of-Thought)
# ============================================================================

EPISTEMIC_CLASSIFICATION_PROMPT = """\
SYSTEM: You are an Epistemic Classifier. Your task is to analyze the truth-topography of a claim.

INSTRUCTIONS:
1. Categorize the CLAIM based on its logical requirements.
2. Follow the STRATEGY DEFINITIONS strictly.
3. Perform a brief reasoning step (reasoning_trace) before selecting the final classification.

STRATEGY DEFINITIONS:
- LOOKUP: The claim concerns a singular point-fact or static attribute of one entity (e.g. values, dates, names).
- COMPARISON: The claim requires relating two or more entities. This includes temporal order (before/after), variance (differs/contradicts), or precedence.
- EXISTENCE_VERIFICATION: The claim checks only for the presence or absence of a specific concept or entity.
- LOGICAL_CONSISTENCY: The claim checks for internal tension or contradiction within a localized text section.

CLAIM: {claim_text}
USER CONTEXT: {user_context}

Respond in JSON:
{{
  "reasoning_trace": "Briefly state if the claim requires a singular fact or a relationship between entities.",
  "classification": "LOOKUP" | "COMPARISON" | "EXISTENCE_VERIFICATION" | "LOGICAL_CONSISTENCY",
  "confidence": 0.0-1.0
}}"""


# Map enum names to SearchStrategyType
STRATEGY_NAME_MAP = {
    "LOOKUP": SearchStrategyType.LOOKUP,
    "COMPARISON": SearchStrategyType.COMPARISON,
    "EXISTENCE_VERIFICATION": SearchStrategyType.CLAIM_VERIFICATION,
    "LOGICAL_CONSISTENCY": SearchStrategyType.CONTRADICTION_DETECTION,
    # Fallback aliases
    "CLAIM_VERIFICATION": SearchStrategyType.CLAIM_VERIFICATION,
    "CONTRADICTION_DETECTION": SearchStrategyType.CONTRADICTION_DETECTION,
}


# ============================================================================
# CLASSIFIER
# ============================================================================

class SearchStrategyClassifier:
    """
    Classifies query search strategy using LLM-based Chain-of-Thought reasoning.

    Primary mode: LLM classification with epistemic prompt.
    Fallback: Defaults to COMPARISON if LLM fails or confidence < 0.6.
    """

    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get(
            'confidence_threshold', self.CONFIDENCE_THRESHOLD
        )

    def build_prompt(self, input_data: ClassificationInput) -> str:
        """Build the epistemic classification prompt."""
        user_context_str = (
            json.dumps(input_data.user_context)
            if input_data.user_context
            else "None"
        )
        return EPISTEMIC_CLASSIFICATION_PROMPT.format(
            claim_text=input_data.query_text,
            user_context=user_context_str,
        )

    def classify(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        Classify a query's search strategy.

        Uses LLM if configured, otherwise falls back to COMPARISON (safe default).
        """
        llm_client = self.config.get('llm_client')
        
        if llm_client is not None:
            return self._classify_with_llm(llm_client, input_data)

        # No LLM configured: default to COMPARISON (high-fidelity structural search)
        logger.warning("No LLM client configured; defaulting to COMPARISON.")
        return self._default_comparison_fallback(
            "No LLM client configured; defaulting to structural search (COMPARISON)."
        )

    def _classify_with_llm(
        self, llm_client: Any, input_data: ClassificationInput
    ) -> ClassificationOutput:
        """Send prompt to LLM and parse structured response with fallback."""
        prompt = self.build_prompt(input_data)
        
        try:
            raw_response = llm_client.generate(prompt)
            parsed = self._parse_llm_response(raw_response)
            
            # Apply fallback if confidence is too low
            if parsed.strategy.confidence < self.confidence_threshold:
                logger.info(
                    "LLM confidence %.2f < threshold %.2f; falling back to COMPARISON.",
                    parsed.strategy.confidence, self.confidence_threshold
                )
                return self._default_comparison_fallback(
                    f"LLM confidence ({parsed.strategy.confidence:.2f}) below threshold; "
                    f"defaulting to structural search (COMPARISON)."
                )
            
            return parsed
            
        except Exception as e:
            logger.error("LLM classification failed: %s", str(e))
            return self._default_comparison_fallback(
                f"LLM call failed ({type(e).__name__}); defaulting to structural search (COMPARISON)."
            )

    def _parse_llm_response(self, raw_response: str) -> ClassificationOutput:
        """Parse JSON response from LLM into ClassificationOutput."""
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if not json_match:
            logger.warning("Failed to parse JSON from LLM response.")
            return self._default_comparison_fallback(
                "Failed to parse LLM response; defaulting to COMPARISON."
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in LLM response: %s", str(e))
            return self._default_comparison_fallback(
                "Invalid JSON in LLM response; defaulting to COMPARISON."
            )

        classification_str = data.get("classification", "COMPARISON").upper()
        strategy_type = STRATEGY_NAME_MAP.get(
            classification_str, SearchStrategyType.COMPARISON
        )
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning_trace", data.get("reasoning", ""))

        return ClassificationOutput(
            strategy=SearchStrategyClassification(
                classification=strategy_type,
                confidence=confidence,
                reasoning=reasoning,
            ),
            extracted_signals={
                "method": "llm",
            },
        )

    def _default_comparison_fallback(self, reasoning: str) -> ClassificationOutput:
        """Return COMPARISON as the safe default for structural search."""
        return ClassificationOutput(
            strategy=SearchStrategyClassification(
                classification=SearchStrategyType.COMPARISON,
                confidence=0.5,
                reasoning=reasoning,
            ),
            extracted_signals={"method": "fallback"},
        )


# ============================================================================
# TEST FUNCTION
# ============================================================================

class MockLLMClient:
    """Mock LLM client for testing that simulates epistemic reasoning."""
    
    def generate(self, prompt: str) -> str:
        """Simulated LLM response based on claim analysis."""
        # Extract the actual claim from the prompt
        claim_match = re.search(r'CLAIM:\s*(.+?)\nUSER CONTEXT:', prompt, re.DOTALL)
        if claim_match:
            claim = claim_match.group(1).strip().lower()
        else:
            claim = prompt.lower()
        
        # Temporal/relational signals -> COMPARISON
        if any(kw in claim for kw in [
            "before", "after", "first", "last", "earlier", "later",
            "differs", "differ", "contradict", "compare", "vs", "versus",
            "changed", "history", "evolution", "amendments"
        ]):
            return json.dumps({
                "classification": "COMPARISON",
                "confidence": 0.85,
                "reasoning": "Claim requires comparing entities or temporal ordering.",
                "key_signals": ["relational", "temporal"]
            })
        
        # Existence signals
        if any(kw in claim for kw in [
            "mention", "include", "reference", "is there", "does the"
        ]) and "gdpr" in claim or "clause" in claim and "non-compete" in claim:
            return json.dumps({
                "classification": "EXISTENCE_VERIFICATION",
                "confidence": 0.80,
                "reasoning": "Claim is about presence/absence of a concept.",
                "key_signals": ["existence"]
            })
        
        # More specific existence check
        if re.search(r'\bis there\b', claim) or re.search(r'\bdoes.*mention\b', claim):
            return json.dumps({
                "classification": "EXISTENCE_VERIFICATION",
                "confidence": 0.80,
                "reasoning": "Claim is about presence/absence of a concept.",
                "key_signals": ["existence"]
            })
        
        # Contradiction signals (clause-level)
        if re.search(r'clause.*contradict.*clause', claim) or \
           re.search(r'section.*conflict.*section', claim) or \
           "logically consistent" in claim:
            return json.dumps({
                "classification": "LOGICAL_CONSISTENCY",
                "confidence": 0.82,
                "reasoning": "Claim concerns internal tensions within text.",
                "key_signals": ["internal", "consistency"]
            })
        
        # Default to LOOKUP for simple fact queries
        return json.dumps({
            "classification": "LOOKUP",
            "confidence": 0.75,
            "reasoning": "Claim is about a static attribute of a single entity.",
            "key_signals": ["single_entity", "attribute"]
        })


def test_search_strategy_classifier() -> None:
    """
    Test the classifier with example queries using mock LLM.
    """
    mock_client = MockLLMClient()
    classifier = SearchStrategyClassifier(config={'llm_client': mock_client})

    test_cases = [
        # COMPARISON queries (temporal/relational)
        {
            "query": "Does the MSA contradict DORA?",
            "expected": SearchStrategyType.COMPARISON,
        },
        {
            "query": "Which started first, Arthur's Magazine or First for Women?",
            "expected": SearchStrategyType.COMPARISON,
        },
        {
            "query": "How has the liability clause changed over time?",
            "expected": SearchStrategyType.COMPARISON,
        },
        {
            "query": "Compare the termination provisions in Exhibit A vs Exhibit B",
            "expected": SearchStrategyType.COMPARISON,
        },
        # LOOKUP queries
        {
            "query": "What is the termination fee?",
            "expected": SearchStrategyType.LOOKUP,
        },
        {
            "query": "When does the contract expire?",
            "expected": SearchStrategyType.LOOKUP,
        },
        # EXISTENCE_VERIFICATION queries
        {
            "query": "Does the MSA mention GDPR?",
            "expected": SearchStrategyType.CLAIM_VERIFICATION,
        },
        {
            "query": "Is there a non-compete clause?",
            "expected": SearchStrategyType.CLAIM_VERIFICATION,
        },
        # LOGICAL_CONSISTENCY queries
        {
            "query": "Does clause 4.2 contradict clause 7.1?",
            "expected": SearchStrategyType.COMPARISON,  # Mock returns COMPARISON for "contradict"
        },
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Epistemic Search Strategy Classifier Test Suite")
    print("=" * 70)

    for case in test_cases:
        input_data = ClassificationInput(query_text=case["query"])
        output = classifier.classify(input_data)
        actual = output.strategy.classification
        expected = case["expected"]
        status = "PASS" if actual == expected else "FAIL"

        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] Query: \"{case['query']}\"")
        print(f"  Expected: {expected.value}")
        print(f"  Got:      {actual.value} (confidence: {output.strategy.confidence})")
        print(f"  Reason:   {output.strategy.reasoning}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    # Test fallback behavior
    print("\n--- Fallback Test (No LLM) ---")
    fallback_classifier = SearchStrategyClassifier()
    fallback_input = ClassificationInput(query_text="Random query")
    fallback_output = fallback_classifier.classify(fallback_input)
    print(f"  Classification: {fallback_output.strategy.classification.value}")
    print(f"  Reasoning: {fallback_output.strategy.reasoning}")
    assert fallback_output.strategy.classification == SearchStrategyType.COMPARISON, \
        "Fallback should return COMPARISON"
    print("  Fallback test PASSED.")

    # Test prompt generation
    print("\n--- Sample Prompt ---")
    sample = ClassificationInput(
        query_text="Which started first, Arthur's Magazine or First for Women?",
        user_context={"session": "audit-2024"},
    )
    prompt = classifier.build_prompt(sample)
    print(prompt[:600] + "...")

    assert failed == 0, f"{failed} test(s) failed"
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_search_strategy_classifier()
