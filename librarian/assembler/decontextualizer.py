"""
Decontextualizer Module
Location: librarian/assembler/decontextualizer.py
Owner: LIBRARIAN Instance

Responsibility: Substitute defined terms with their definitions so that
each document chunk is self-contained ("decontextualized").  Uses an
LLM-driven pass for high-quality substitution, with a deterministic
regex-based fallback when the LLM call fails or is disabled.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from schemas.contracts import (
    DecontextualizationReport,
    SubstitutionFailure,
    SubstitutionFailureReason,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LOCAL I/O CONTRACTS
# ============================================================================

@dataclass
class DecontextInput:
    """Input for decontextualization of a single text unit."""
    content: str
    definitions: Dict[str, str]   # term -> definition text
    max_tokens: int


@dataclass
class DecontextOutput:
    """Output of decontextualization."""
    decontextualized_content: str
    report: DecontextualizationReport


# ============================================================================
# LLM CLIENT PROTOCOL
# ============================================================================

class LLMClient(Protocol):
    """Protocol for the underlying LLM call."""
    def generate(self, prompt: str) -> str: ...


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

DECONTEXTUALIZATION_PROMPT = """\
You are an expert text processor performing definition substitution for an audit.

ORIGINAL TEXT:
{content}

DEFINITIONS AVAILABLE:
{definitions}

TASK: Rewrite the text by substituting defined terms with their definitions to make the text self-contained.

RULES:
1. Replace each defined term with: [term (defined as: definition)]
2. Only substitute on first occurrence per paragraph
3. If substitution would exceed {max_tokens} tokens, note as overflow
4. If a term references another defined term (circular), note and skip
5. Preserve original meaning exactly—do not paraphrase

EXAMPLE:
Original: "The Publication shall notify the Subscriber of all Changes."
Definitions: {{"Publication": "The Morning Star", "Changes": "updates to the delivery schedule"}}
Output: "The Publication [defined as: The Morning Star] shall notify the Subscriber of all Changes [defined as: updates to the delivery schedule]."

Respond in JSON:
{{
  "decontextualized_content": "...",
  "substitutions_made": [
    {{"term": "...", "definition": "...", "location": "..."}}
  ],
  "failures": [
    {{"term": "...", "reason": "NOT_FOUND | CIRCULAR | OVERFLOW"}}
  ],
  "final_token_count": 123
}}"""


# ============================================================================
# PARSING HELPERS
# ============================================================================

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from an LLM response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    return json.loads(text)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


def _safe_failure_reason(raw: str) -> SubstitutionFailureReason:
    """Map a raw string to SubstitutionFailureReason."""
    try:
        return SubstitutionFailureReason(raw.upper())
    except (ValueError, AttributeError):
        return SubstitutionFailureReason.NOT_FOUND


# ============================================================================
# CIRCULAR REFERENCE DETECTION
# ============================================================================

def _detect_circular_terms(definitions: Dict[str, str]) -> set[str]:
    """Return the set of terms involved in circular definition chains.

    A term is circular if its definition text contains another defined
    term whose own definition (transitively) refers back to the original.
    """
    circular: set[str] = set()

    def _references(term: str) -> set[str]:
        """Terms referenced in the definition of *term*."""
        defn = definitions.get(term, "")
        return {
            t for t in definitions
            if t != term and re.search(r'\b' + re.escape(t) + r'\b', defn, re.IGNORECASE)
        }

    for term in definitions:
        visited: set[str] = set()
        stack = list(_references(term))
        while stack:
            current = stack.pop()
            if current == term:
                circular.add(term)
                break
            if current in visited:
                continue
            visited.add(current)
            stack.extend(_references(current))

    return circular


# ============================================================================
# DECONTEXTUALIZER
# ============================================================================

class Decontextualizer:
    """Substitute defined terms within document text.

    The primary path uses an LLM to produce context-aware substitution.
    If the LLM call fails (or ``use_llm=False`` in config), a
    deterministic regex-based fallback is used instead.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.
    config:
        Optional dictionary with keys:
        - ``use_llm`` (bool, default True)
        - ``max_tokens_per_unit`` (int, default 512)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or {}
        self._use_llm = self._config.get("use_llm", True)
        self._default_max_tokens = self._config.get("max_tokens_per_unit", 512)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, inp: DecontextInput) -> DecontextOutput:
        """Decontextualize a single text unit.

        Parameters
        ----------
        inp:
            ``DecontextInput`` with content, definitions, and token limit.

        Returns
        -------
        DecontextOutput
            The rewritten content and a substitution report.
        """
        if not inp.definitions:
            return DecontextOutput(
                decontextualized_content=inp.content,
                report=DecontextualizationReport(
                    terms_resolved=0,
                    terms_failed=[],
                    substitution_success_rate=1.0,
                ),
            )

        if self._use_llm:
            result = self._llm_substitute(inp)
            if result is not None:
                return result

        # Fallback: deterministic substitution
        return self._deterministic_substitute(inp)

    # ------------------------------------------------------------------
    # LLM-driven substitution
    # ------------------------------------------------------------------

    def _llm_substitute(self, inp: DecontextInput) -> Optional[DecontextOutput]:
        """Attempt LLM-driven substitution. Returns None on failure."""
        definitions_str = json.dumps(inp.definitions, indent=2)
        prompt = DECONTEXTUALIZATION_PROMPT.format(
            content=inp.content,
            definitions=definitions_str,
            max_tokens=inp.max_tokens,
        )

        try:
            raw_response = self._llm.generate(prompt)
            data = _parse_json_response(raw_response)
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            logger.warning("LLM decontextualization failed: %s", exc)
            return None

        decontext_content = data.get("decontextualized_content", inp.content)
        substitutions = data.get("substitutions_made", [])
        failures_raw = data.get("failures", [])

        terms_resolved = len(substitutions)
        terms_failed: List[SubstitutionFailure] = [
            SubstitutionFailure(
                term=f.get("term", "unknown"),
                failure_reason=_safe_failure_reason(f.get("reason", "NOT_FOUND")),
            )
            for f in failures_raw
        ]

        total = terms_resolved + len(terms_failed)
        success_rate = terms_resolved / total if total > 0 else 1.0

        return DecontextOutput(
            decontextualized_content=decontext_content,
            report=DecontextualizationReport(
                terms_resolved=terms_resolved,
                terms_failed=terms_failed,
                substitution_success_rate=success_rate,
            ),
        )

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    def _deterministic_substitute(self, inp: DecontextInput) -> DecontextOutput:
        """Regex-based first-occurrence-per-paragraph substitution.

        Rules applied:
        - Circular terms are skipped.
        - If the running token estimate exceeds ``max_tokens``, remaining
          terms are marked as OVERFLOW.
        - Terms not found in the text are marked NOT_FOUND.
        """
        circular_terms = _detect_circular_terms(inp.definitions)
        paragraphs = inp.content.split("\n\n")

        terms_resolved = 0
        terms_failed: List[SubstitutionFailure] = []
        substituted_paragraphs: List[str] = []

        # Sort terms longest-first to avoid partial matches
        sorted_terms = sorted(inp.definitions.keys(), key=len, reverse=True)

        for para in paragraphs:
            current = para
            for term in sorted_terms:
                # Skip circular terms
                if term in circular_terms:
                    if para == paragraphs[0]:  # record failure once
                        terms_failed.append(
                            SubstitutionFailure(
                                term=term,
                                failure_reason=SubstitutionFailureReason.CIRCULAR,
                            )
                        )
                    continue

                definition = inp.definitions[term]
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                match = pattern.search(current)
                if match is None:
                    continue

                replacement = f"{term} [defined as: {definition}]"

                # Check overflow before applying
                candidate = pattern.sub(replacement, current, count=1)
                running_tokens = _estimate_tokens(
                    "\n\n".join(substituted_paragraphs + [candidate])
                )
                if running_tokens > inp.max_tokens:
                    terms_failed.append(
                        SubstitutionFailure(
                            term=term,
                            failure_reason=SubstitutionFailureReason.OVERFLOW,
                        )
                    )
                    continue

                current = candidate
                terms_resolved += 1

            substituted_paragraphs.append(current)

        # Record terms not found anywhere in the content
        for term in sorted_terms:
            if term in circular_terms:
                continue
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            if not pattern.search(inp.content):
                already_recorded = any(f.term == term for f in terms_failed)
                if not already_recorded:
                    terms_failed.append(
                        SubstitutionFailure(
                            term=term,
                            failure_reason=SubstitutionFailureReason.NOT_FOUND,
                        )
                    )

        total = terms_resolved + len(terms_failed)
        success_rate = terms_resolved / total if total > 0 else 1.0

        return DecontextOutput(
            decontextualized_content="\n\n".join(substituted_paragraphs),
            report=DecontextualizationReport(
                terms_resolved=terms_resolved,
                terms_failed=terms_failed,
                substitution_success_rate=success_rate,
            ),
        )
