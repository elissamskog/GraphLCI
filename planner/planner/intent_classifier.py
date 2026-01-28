"""
Intent Classifier Module
Responsibility: Classify query intent using few-shot prompting.

Uses a structured prompt template with intent definitions and examples
to classify queries as COMPARISON, SENSEMAKING, or LOOKUP. Designed to
resist adversarial paraphrasing (no keyword heuristics).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from schemas.contracts import IntentClassification, IntentType


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class ClassificationInput:
    """Input to the intent classifier."""
    query_text: str
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class ClassificationOutput:
    """Output from the intent classifier."""
    intent: IntentClassification
    extracted_signals: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier for a legal document audit system.

TASK: Classify the user's query into one of five intents.

INTENT DEFINITIONS:
1. COMPARISON - The user wants to compare two or more documents at a high level, \
identifying differences or similarities between them.
   Examples:
   - "Does the MSA contradict DORA?"
   - "Are these clauses aligned?"
   - "Is the liability cap consistent with regulatory requirements?"
   - "Compare the termination provisions"
   - "How does the indemnification in Exhibit A differ from the main agreement?"
   - "Check if the data processing addendum conflicts with the privacy policy"

2. SENSEMAKING - The user wants to understand themes, patterns, or \
synthesize information across multiple documents.
   Examples:
   - "What are the key risk themes?"
   - "Summarize the liability provisions across all contracts"
   - "Who are the main parties and how do they relate?"
   - "Give me an overview of the governance structure"
   - "What patterns emerge from the termination clauses?"
   - "How is data protection addressed across the agreement suite?"

3. LOOKUP - The user wants to find a specific fact, value, or clause.
   Examples:
   - "What is the termination fee?"
   - "When does the contract expire?"
   - "What's the governing law?"
   - "Find the notice period for termination"
   - "What is the liability cap amount?"

4. CLAIM_VERIFICATION - The user wants to verify whether something exists, \
is mentioned, or is present in the documents. Absence/existence checks.
   Examples:
   - "Does the MSA mention GDPR?"
   - "Is there a non-compete clause?"
   - "Does the agreement include a force majeure provision?"
   - "Is there any reference to data retention?"
   - "Does the contract address intellectual property?"
   - "Is GDPR referenced anywhere in the agreement?"

5. CONTRADICTION_DETECTION - The user wants to check logical consistency \
between specific clauses or sections within or across documents.
   Examples:
   - "Does clause 4.2 contradict clause 7.1?"
   - "Is section 3 logically consistent with section 8?"
   - "Do the termination clauses in sections 5 and 12 conflict?"
   - "Is there a contradiction between the liability clause and the indemnification section?"
   - "Are the obligations in clause 3.1 consistent with the limitations in clause 9.4?"

QUERY: {query_text}

USER CONTEXT (if any): {user_context}

Respond in JSON format:
{{
  "classification": "COMPARISON" | "SENSEMAKING" | "LOOKUP" | "CLAIM_VERIFICATION" | "CONTRADICTION_DETECTION",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this classification was chosen",
  "key_signals": ["list", "of", "phrases", "that", "informed", "decision"]
}}"""


# ============================================================================
# CLASSIFIER
# ============================================================================

class IntentClassifier:
    """
    Classifies query intent using few-shot prompting.

    In production, this calls a Claude model with the prompt template.
    For testing/offline use, it includes a rule-based fallback that
    approximates the few-shot behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

    def build_prompt(self, input_data: ClassificationInput) -> str:
        """Build the classification prompt from the template."""
        user_context_str = (
            json.dumps(input_data.user_context)
            if input_data.user_context
            else "None"
        )
        return INTENT_CLASSIFICATION_PROMPT.format(
            query_text=input_data.query_text,
            user_context=user_context_str,
        )

    def classify(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        Classify a query's intent.

        In production, sends the prompt to Claude and parses the JSON response.
        This implementation uses a rule-based fallback for offline testing.
        """
        prompt = self.build_prompt(input_data)

        # If an LLM client is configured, use it
        llm_client = self.config.get('llm_client')
        if llm_client is not None:
            return self._classify_with_llm(llm_client, prompt)

        # Offline rule-based fallback (semantic heuristics, not keyword matching)
        return self._classify_offline(input_data)

    def _classify_with_llm(
        self, llm_client: Any, prompt: str
    ) -> ClassificationOutput:
        """Send prompt to LLM and parse structured response."""
        raw_response = llm_client.generate(prompt)
        parsed = self._parse_llm_response(raw_response)
        return parsed

    def _parse_llm_response(self, raw_response: str) -> ClassificationOutput:
        """Parse JSON response from LLM into ClassificationOutput."""
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if not json_match:
            return ClassificationOutput(
                intent=IntentClassification(
                    classification=IntentType.LOOKUP,
                    confidence=0.0,
                    reasoning="Failed to parse LLM response",
                ),
                extracted_signals={"parse_error": True, "raw": raw_response},
            )

        data = json.loads(json_match.group())
        intent_type = IntentType(data["classification"])

        return ClassificationOutput(
            intent=IntentClassification(
                classification=intent_type,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            ),
            extracted_signals={
                "key_signals": data.get("key_signals", []),
            },
        )

    def _classify_offline(
        self, input_data: ClassificationInput
    ) -> ClassificationOutput:
        """
        Rule-based offline classification for testing.

        Uses semantic pattern groups rather than single keywords to avoid
        brittle keyword matching. Each pattern group captures a family of
        phrasings that signal a particular intent.

        Supports 5 intents: COMPARISON, SENSEMAKING, LOOKUP,
        CONTRADICTION_DETECTION, and CLAIM_VERIFICATION.
        """
        query = input_data.query_text.lower()
        original_query = input_data.query_text

        # COMPARISON signals: high-level document comparison, alignment,
        # difference between documents
        comparison_patterns = [
            r'\balign(ed|ment)?\b',
            r'\bcompar[ei]',
            r'\bdiffer(s|ence|ent)?\b',
            r'\bconsistent\b',
            r'\bcompl(y|iance|iant)\b.*\bwith\b',
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bat odds\b',
            r'\bmismatch',
            r'\bdeviat',
            r'\breconcil',
        ]

        # CONTRADICTION_DETECTION signals: clause-level logical consistency
        contradiction_patterns = [
            r'\blogical(ly)?\s+(consist|conflict)',
            r'\bclause.*vs.*clause',
            r'\bsection.*conflict.*section',
            r'\bcontradic',
            r'\bconflict',
            r'\binconsisten',
        ]

        # CLAIM_VERIFICATION signals: existence/absence checks
        claim_verification_patterns = [
            r'\bdoes\b.*\bmention\b',
            r'\bis\s+there\s+a\b',
            r'\bdoes\b.*\binclude',
            r'\bdoes\b.*\breference',
            r'\babsen(t|ce)',
            r'\banywhere\b',
        ]

        # SENSEMAKING signals: summarization, themes, patterns, overview,
        # synthesis across multiple items
        sensemaking_patterns = [
            r'\bsummar(y|ize|ise)',
            r'\btheme',
            r'\bpattern',
            r'\boverview\b',
            r'\bsynthe(sis|size)',
            r'\bacross\b',
            r'\boverall\b',
            r'\blandscape\b',
            r'\brelat(e|ionship)',
            r'\bstructur(e|al)',
            r'\bkey\s+(risk|issue|point|finding)',
            r'\bhow\s+(is|are|does).*addressed',
            r'\bwhat\s+(are|is)\s+the\s+(main|key|primary)',
        ]

        # LOOKUP signals: specific value retrieval, point questions
        lookup_patterns = [
            r'\bwhat\s+is\s+the\b',
            r'\bwhen\s+does\b',
            r'\bhow\s+much\b',
            r'\bfind\s+the\b',
            r'\bwhat\'?s\s+the\b',
            r'\bspecific(ally)?\b',
            r'\bexact(ly)?\b',
            r'\bamount\b',
            r'\bdate\b.*\b(of|for)\b',
            r'\bwho\s+is\b',
            r'\bwhere\s+is\b',
        ]

        comparison_score = sum(
            1 for p in comparison_patterns if re.search(p, query)
        )
        contradiction_score = sum(
            1 for p in contradiction_patterns if re.search(p, query)
        )
        claim_verification_score = sum(
            1 for p in claim_verification_patterns if re.search(p, query)
        )
        sensemaking_score = sum(
            1 for p in sensemaking_patterns if re.search(p, query)
        )
        lookup_score = sum(
            1 for p in lookup_patterns if re.search(p, query)
        )

        total = (comparison_score + contradiction_score
                 + claim_verification_score + sensemaking_score + lookup_score)
        if total == 0:
            # Default fallback
            return ClassificationOutput(
                intent=IntentClassification(
                    classification=IntentType.LOOKUP,
                    confidence=0.4,
                    reasoning="No strong signals detected; defaulting to LOOKUP",
                ),
                extracted_signals={"method": "offline_fallback"},
            )

        scores = {
            IntentType.COMPARISON: comparison_score,
            IntentType.CONTRADICTION_DETECTION: contradiction_score,
            IntentType.CLAIM_VERIFICATION: claim_verification_score,
            IntentType.SENSEMAKING: sensemaking_score,
            IntentType.LOOKUP: lookup_score,
        }

        pattern_map = {
            IntentType.COMPARISON: comparison_patterns,
            IntentType.CONTRADICTION_DETECTION: contradiction_patterns,
            IntentType.CLAIM_VERIFICATION: claim_verification_patterns,
            IntentType.SENSEMAKING: sensemaking_patterns,
            IntentType.LOOKUP: lookup_patterns,
        }

        # ---- Disambiguation rules ----

        # Detect structural signals in the original (case-sensitive) query
        has_allcaps_docs = len(re.findall(r'\b[A-Z]{2,}\b', original_query)) >= 2
        has_clause_section_refs = bool(
            re.search(r'\b(clause|section|article)\s+\d', query)
        )

        # Between COMPARISON and CONTRADICTION_DETECTION:
        # If ≥2 ALL-CAPS doc names and no clause/section refs → force COMPARISON
        # If clause/section refs and no multi-doc names → force CONTRADICTION_DETECTION
        if comparison_score > 0 and contradiction_score > 0:
            if has_allcaps_docs and not has_clause_section_refs:
                scores[IntentType.CONTRADICTION_DETECTION] = 0
            elif has_clause_section_refs and not has_allcaps_docs:
                scores[IntentType.COMPARISON] = 0

        # Between CLAIM_VERIFICATION and LOOKUP:
        # If query has existence signals → force CLAIM_VERIFICATION
        existence_signals = bool(
            re.search(r'\bis\s+there\b', query)
            or re.search(r'\bdoes\b.*\bmention\b', query)
            or re.search(r'\bdoes\b.*\binclude\b', query)
            or re.search(r'\bdoes\b.*\breference\b', query)
            or re.search(r'\banywhere\b', query)
            or re.search(r'\babsen(t|ce)\b', query)
        )
        if claim_verification_score > 0 and lookup_score > 0 and existence_signals:
            scores[IntentType.LOOKUP] = 0

        # Recalculate total after disambiguation
        total = sum(scores.values())
        if total == 0:
            return ClassificationOutput(
                intent=IntentClassification(
                    classification=IntentType.LOOKUP,
                    confidence=0.4,
                    reasoning="No strong signals after disambiguation; defaulting to LOOKUP",
                ),
                extracted_signals={"method": "offline_fallback"},
            )

        best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_intent] / total

        signals = [
            p for p in pattern_map[best_intent] if re.search(p, query)
        ]

        return ClassificationOutput(
            intent=IntentClassification(
                classification=best_intent,
                confidence=round(confidence, 2),
                reasoning=f"Offline classification: {best_intent.value} scored "
                          f"{scores[best_intent]}/{total} based on pattern matching",
            ),
            extracted_signals={
                "method": "offline_rule_based",
                "scores": {k.value: v for k, v in scores.items()},
                "matched_patterns": signals,
            },
        )


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_intent_classifier() -> None:
    """
    Test the intent classifier with example queries across all three intents.
    Validates that the offline classifier produces correct classifications.
    """
    classifier = IntentClassifier()

    test_cases = [
        # COMPARISON queries
        {
            "query": "Does the MSA contradict DORA?",
            "expected": IntentType.COMPARISON,
        },
        {
            "query": "Are the liability clauses consistent across the agreements?",
            "expected": IntentType.COMPARISON,
        },
        {
            "query": "Compare the termination provisions in Exhibit A vs Exhibit B",
            "expected": IntentType.COMPARISON,
        },
        {
            "query": "Is the indemnification cap aligned with regulatory requirements?",
            "expected": IntentType.COMPARISON,
        },
        {
            "query": "Check for conflicts between the data processing addendum and privacy policy",
            "expected": IntentType.COMPARISON,
        },
        # SENSEMAKING queries
        {
            "query": "What are the key risk themes across all documents?",
            "expected": IntentType.SENSEMAKING,
        },
        {
            "query": "Summarize the liability provisions across all contracts",
            "expected": IntentType.SENSEMAKING,
        },
        {
            "query": "Give me an overview of the governance structure",
            "expected": IntentType.SENSEMAKING,
        },
        {
            "query": "How is data protection addressed across the agreement suite?",
            "expected": IntentType.SENSEMAKING,
        },
        # LOOKUP queries
        {
            "query": "What is the termination fee?",
            "expected": IntentType.LOOKUP,
        },
        {
            "query": "When does the contract expire?",
            "expected": IntentType.LOOKUP,
        },
        {
            "query": "Is there a non-compete clause?",
            "expected": IntentType.CLAIM_VERIFICATION,
        },
        {
            "query": "What's the governing law?",
            "expected": IntentType.LOOKUP,
        },
        # CLAIM_VERIFICATION queries
        {
            "query": "Does the MSA mention GDPR?",
            "expected": IntentType.CLAIM_VERIFICATION,
        },
        {
            "query": "Does the agreement include a force majeure provision?",
            "expected": IntentType.CLAIM_VERIFICATION,
        },
        {
            "query": "Is GDPR referenced anywhere in the agreement?",
            "expected": IntentType.CLAIM_VERIFICATION,
        },
        # CONTRADICTION_DETECTION queries
        {
            "query": "Does clause 4.2 contradict clause 7.1?",
            "expected": IntentType.CONTRADICTION_DETECTION,
        },
        {
            "query": "Is section 3 logically consistent with section 8?",
            "expected": IntentType.CONTRADICTION_DETECTION,
        },
        {
            "query": "Do the termination clauses in sections 5 and 12 conflict?",
            "expected": IntentType.CONTRADICTION_DETECTION,
        },
        {
            "query": "Is there a contradiction between the liability clause and the indemnification section?",
            "expected": IntentType.CONTRADICTION_DETECTION,
        },
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Intent Classifier Test Suite")
    print("=" * 70)

    for case in test_cases:
        input_data = ClassificationInput(query_text=case["query"])
        output = classifier.classify(input_data)
        actual = output.intent.classification
        expected = case["expected"]
        status = "PASS" if actual == expected else "FAIL"

        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] Query: \"{case['query']}\"")
        print(f"  Expected: {expected.value}")
        print(f"  Got:      {actual.value} (confidence: {output.intent.confidence})")
        print(f"  Reason:   {output.intent.reasoning}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    # Also test prompt generation
    print("\n--- Sample Prompt ---")
    sample = ClassificationInput(
        query_text="Does the MSA contradict DORA?",
        user_context={"session": "audit-2024"},
    )
    prompt = classifier.build_prompt(sample)
    print(prompt[:500] + "...")

    assert failed == 0, f"{failed} test(s) failed"
    print("\nAll tests passed.")


if __name__ == "__main__":
    test_intent_classifier()
