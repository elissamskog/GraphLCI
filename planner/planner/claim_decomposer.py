"""
Claim Decomposer for Two-Stage Seeding Architecture.
Location: planner/planner/claim_decomposer.py

Decomposes complex claims into atomic subclaims to enable multi-source
PPR seeding and resolve the Locality Trap.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from schemas.contracts import Subclaim, ClaimDecomposition

logger = logging.getLogger("knowledgeaudit.planner.claim_decomposer")


# LLM Prompt Template for Decomposition
# LLM Prompt Template for Decomposition
DECOMPOSITION_PROMPT = """You are a retrieval-augmented generation (RAG) assistant specializing in multi-hop query decomposition.

Analyze the user query to see if it requires multiple search steps to resolve.

DECOMPOSITION LOGIC:
1. FUNCTIONAL DESCRIPTORS: If an entity is described by a property or role (e.g., "The author of X", "The winner of Y") rather than a name, you MUST decompose.
   - Step 1: Resolve the descriptor to a named entity. (Priority: bridge)
   - Step 2: Query the property of that resolved entity. (Priority: primary)

2. RELATIONSHIP CHAINS: If the query connects multiple entities through a chain of facts (A -> B -> C), you MUST decompose.
   - Generate a subclaim for each link in the chain.

3. KNOWLEDGE LOCALITY: If the query requires information that is unlikely to be contained in a single subject-specific document, decompose to enable parallel retrieval.

OPERATIONAL RULES:
- `should_decompose`: Set to `true` if any of the above apply.
- `priority`: Use "bridge" for intermediate discovery steps and "primary" for the final objective step.
- `bridge_entities`: Always include the descriptive terms that need to be resolved in the first step (e.g., "author of report").

OUTPUT FORMAT (JSON):
{{
  "primary_subject": "the main entity or role being investigated",
  "subclaims": [
    {{
      "subclaim_id": "sc_1",
      "objective": "atomic search objective",
      "entities": ["list of search terms or entities"],
      "relationship_type": "identity|relationship|attribute",
      "priority": "bridge|primary"
    }}
  ],
  "bridge_entities": ["entities or descriptors that connect subclaims"],
  "decomposition_reasoning": "structural analysis of the query complexity",
  "should_decompose": boolean,
  "confidence": 0.0-1.0
}}

CLAIM: {claim}

Return ONLY valid JSON."""


def decompose_claim(
    query_text: str,
    llm_client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ClaimDecomposition:
    """
    Decompose a claim into atomic subclaims for multi-source seeding.

    Args:
        query_text: The claim text to decompose
        llm_client: Optional LLM client (supports generate() method)
        config: Configuration dict with decomposition parameters

    Returns:
        ClaimDecomposition with subclaims and metadata
    """
    config = config or {}
    min_entities = config.get('min_entities_for_decomposition', 2)
    timeout_ms = config.get('llm_timeout_ms', 2000)

    # Quick heuristic: If query is very short, likely single-hop
    if len(query_text.split()) < 5:
        logger.info(f"Query too short ({len(query_text.split())} words), skipping decomposition")
        return _create_fallback_decomposition(query_text, "Query too short for decomposition")

    # If no LLM client, use fallback
    if llm_client is None:
        logger.warning("No LLM client provided, using fallback decomposition")
        return _create_fallback_decomposition(query_text, "No LLM client available")

    try:
        # Format prompt
        prompt = DECOMPOSITION_PROMPT.format(claim=query_text)

        # Call LLM
        logger.info(f"Decomposing claim: {query_text[:100]}...")
        response = llm_client.generate(prompt, num_ctx=4096)

        # Parse JSON response
        result = json.loads(response)

        # Validate and construct ClaimDecomposition
        decomposition = _parse_llm_response(result, query_text, config)

        logger.info(
            f"Decomposition result: should_decompose={decomposition.should_decompose}, "
            f"subclaims={len(decomposition.subclaims)}, "
            f"confidence={decomposition.confidence:.2f}"
        )

        return decomposition

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        return _create_fallback_decomposition(query_text, f"JSON parse error: {e}")

    except TimeoutError:
        logger.warning(f"LLM decomposition timed out after {timeout_ms}ms")
        return _create_fallback_decomposition(query_text, "LLM timeout")

    except Exception as e:
        logger.error(f"Unexpected error during decomposition: {e}")
        return _create_fallback_decomposition(query_text, f"Error: {e}")


def _parse_llm_response(
    result: Dict[str, Any],
    query_text: str,
    config: Dict[str, Any]
) -> ClaimDecomposition:
    """
    Parse and validate LLM JSON response into ClaimDecomposition.

    Args:
        result: Parsed JSON from LLM
        query_text: Original query text
        config: Configuration dict

    Returns:
        Validated ClaimDecomposition
    """
    # Extract fields with defaults
    primary_subject = result.get('primary_subject', '')
    subclaims_data = result.get('subclaims', [])
    bridge_entities = result.get('bridge_entities', [])
    reasoning = result.get('decomposition_reasoning', '')
    should_decompose = result.get('should_decompose', True)
    confidence = result.get('confidence', 0.5)

    # Validation: If 0 subclaims, don't decompose
    if len(subclaims_data) == 0:
        logger.info("LLM returned 0 subclaims, setting should_decompose=False")
        should_decompose = False

    # Validation: If >3 subclaims, keep top 3 by priority
    if len(subclaims_data) > 3:
        logger.info(f"LLM returned {len(subclaims_data)} subclaims, keeping top 3 by priority")
        subclaims_data = _rank_and_trim_subclaims(subclaims_data, max_count=3)

    # Convert subclaims to Subclaim objects
    subclaims = []
    for idx, sc_data in enumerate(subclaims_data):
        subclaim = Subclaim(
            subclaim_id=sc_data.get('subclaim_id', f'sc_{idx+1}'),
            objective=sc_data.get('objective', ''),
            entities=sc_data.get('entities', []),
            relationship_type=sc_data.get('relationship_type', 'unknown'),
            priority=sc_data.get('priority', 'secondary'),
        )
        subclaims.append(subclaim)

    # Final validation: If low confidence, don't decompose
    if confidence < 0.4:
        logger.info(f"Low confidence ({confidence:.2f}), setting should_decompose=False")
        should_decompose = False

    return ClaimDecomposition(
        primary_subject=primary_subject,
        subclaims=subclaims,
        bridge_entities=bridge_entities,
        decomposition_reasoning=reasoning,
        should_decompose=should_decompose,
        confidence=confidence,
    )


def _rank_and_trim_subclaims(
    subclaims_data: List[Dict[str, Any]],
    max_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Rank subclaims by priority and keep top N.

    Priority order: primary > bridge > secondary

    Args:
        subclaims_data: List of subclaim dicts from LLM
        max_count: Maximum number of subclaims to keep

    Returns:
        Sorted and trimmed list of subclaim dicts
    """
    PRIORITY_ORDER = {'primary': 0, 'bridge': 1, 'secondary': 2}

    def get_priority_rank(sc: Dict[str, Any]) -> int:
        priority = sc.get('priority', 'secondary')
        return PRIORITY_ORDER.get(priority, 3)

    sorted_subclaims = sorted(subclaims_data, key=get_priority_rank)
    return sorted_subclaims[:max_count]


def _create_fallback_decomposition(
    query_text: str,
    reason: str
) -> ClaimDecomposition:
    """
    Create a fallback ClaimDecomposition when LLM is unavailable or fails.

    Args:
        query_text: Original query text
        reason: Reason for fallback

    Returns:
        ClaimDecomposition with should_decompose=False
    """
    return ClaimDecomposition(
        primary_subject=query_text[:50],  # Use first 50 chars as subject
        subclaims=[],
        bridge_entities=[],
        decomposition_reasoning=f"Fallback: {reason}",
        should_decompose=False,
        confidence=0.0,
    )
