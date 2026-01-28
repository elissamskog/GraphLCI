"""
Community Summarizer (v5)
Location: librarian/community_summarizer.py
Owner: LIBRARIAN Instance

Generates natural-language summaries for detected communities.
"""

from __future__ import annotations

import json
import logging
from typing import Dict

from schemas.contracts import (
    CommunityStructure,
    KnowledgeGraph,
    LLMClient,
)

logger = logging.getLogger(__name__)


def _parse_llm_response(text: str) -> dict:
    """Parse an LLM JSON response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def summarize_communities(
    graph: KnowledgeGraph,
    community_structure: CommunityStructure,
    llm_client: LLMClient,
) -> Dict[str, str]:
    """Generate summaries for each community using bottom-up processing.

    Level 0 communities are summarized from node content.  Higher-level
    communities are summarized from their child community summaries,
    producing a hierarchical bottom-up summary structure.

    Parameters
    ----------
    graph:
        The knowledge graph.
    community_structure:
        Detected community assignments.
    llm_client:
        LLM client for generating summaries.

    Returns
    -------
    Dict[str, str]
        Mapping of community_id to summary text.
    """
    summaries: Dict[str, str] = {}

    # Process levels in ascending order (bottom-up)
    sorted_levels = sorted(community_structure.levels.keys())

    for level in sorted_levels:
        comm_ids = community_structure.levels[level]
        for comm_id in comm_ids:
            community = community_structure.communities[comm_id]

            if level == 0:
                # Level 0: summarize from node content
                member_nodes = []
                for node_id in community.node_ids:
                    node = graph.nodes.get(node_id)
                    if node is not None:
                        member_nodes.append(node)

                if len(member_nodes) <= 2:
                    summary = "; ".join(n.canonical_text for n in member_nodes)
                    community.summary = summary
                    community.key_themes = [n.canonical_text for n in member_nodes]
                    summaries[comm_id] = summary
                    continue

                node_lines = []
                for node in member_nodes:
                    node_lines.append(f"- [{node.node_type}] \"{node.canonical_text}\"")
                content_block = "\n".join(node_lines)

                prompt = (
                    "Summarize the following group of related concepts/claims from a knowledge graph.\n"
                    "Identify 2-3 major themes, note any tensions or disagreements between the nodes, "
                    "and produce a 2-3 paragraph summary.\n\n"
                    f"Nodes:\n{content_block}\n\n"
                    "Respond in JSON: "
                    '{"summary": "...", "key_themes": ["...", ...], "tensions": "..." or null}'
                )
            else:
                # Level > 0: summarize from child community summaries
                child_ids = community.child_community_ids or []
                child_summaries = []
                for child_id in child_ids:
                    child_comm = community_structure.communities.get(child_id)
                    if child_comm and child_comm.summary:
                        child_summaries.append(child_comm.summary)

                if not child_summaries:
                    # Fall back to node content if no child summaries available
                    member_nodes = []
                    for node_id in community.node_ids:
                        node = graph.nodes.get(node_id)
                        if node is not None:
                            member_nodes.append(node)
                    child_summaries = [n.canonical_text for n in member_nodes[:10]]

                if len(child_summaries) <= 1:
                    summary = child_summaries[0] if child_summaries else ""
                    community.summary = summary
                    community.key_themes = []
                    summaries[comm_id] = summary
                    continue

                child_lines = [f"- Community summary: \"{s}\"" for s in child_summaries]
                content_block = "\n".join(child_lines)

                prompt = (
                    "Synthesize the following sub-community summaries into a higher-level summary.\n"
                    "Identify 2-3 major themes, note any tensions or disagreements between the "
                    "sub-communities, and produce a 2-3 paragraph summary.\n\n"
                    f"Sub-community summaries:\n{content_block}\n\n"
                    "Respond in JSON: "
                    '{"summary": "...", "key_themes": ["...", ...], "tensions": "..." or null}'
                )

            try:
                response = llm_client.generate(prompt)
                parsed = _parse_llm_response(response)
                summary = parsed.get("summary", "")
                key_themes = parsed.get("key_themes", [])
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Failed to parse LLM summary for community %s: %s", comm_id, exc)
                if level == 0:
                    member_nodes_fb = []
                    for node_id in community.node_ids:
                        node = graph.nodes.get(node_id)
                        if node is not None:
                            member_nodes_fb.append(node)
                    summary = "; ".join(n.canonical_text for n in member_nodes_fb[:5])
                    key_themes = [n.canonical_text for n in member_nodes_fb[:3]]
                else:
                    child_ids_fb = community.child_community_ids or []
                    child_sums = [
                        community_structure.communities[cid].summary
                        for cid in child_ids_fb
                        if cid in community_structure.communities
                        and community_structure.communities[cid].summary
                    ]
                    summary = " ".join(child_sums[:3]) if child_sums else ""
                    key_themes = []

            community.summary = summary
            community.key_themes = key_themes
            summaries[comm_id] = summary

    return summaries
