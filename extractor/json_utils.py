"""
Shared JSON Parsing Utilities (v5)
Location: extractor/json_utils.py
Owner: EXTRACTOR Instance

Robust JSON extraction from LLM responses, supporting both
object and array formats.
"""

from __future__ import annotations

import json
import re


def safe_str(value, default: str = "") -> str:
    """Coerce an LLM-returned value to str.

    Handles cases where the LLM returns a list, int, or other non-string
    type instead of the expected string.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def parse_json(raw: str) -> dict:
    """Extract a JSON object from an LLM response.

    Tries three strategies in order:
    1. Direct ``json.loads``
    2. Extract content from markdown ```json ... ``` fences
    3. Extract the first top-level ``{...}`` block via brace matching
    """
    # Strategy 1: direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: markdown fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: brace matching for first top-level object
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(raw[start : i + 1])
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, TypeError):
                        break

    raise ValueError(f"Could not parse JSON object from LLM response: {raw[:200]}")


def parse_json_list(raw: str) -> list:
    """Extract a JSON array from an LLM response.

    Tries three strategies in order:
    1. Direct ``json.loads``
    2. Extract content from markdown ```json ... ``` fences
    3. Extract the first top-level ``[...]`` block via bracket matching
    
    If the result is a single dict object, wraps it in a list.
    If the result is a dict with a list-valued field (e.g., {"entities": [...]}), extracts that list.
    """
    # Strategy 1: direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Check if dict contains a list (e.g., {"entities": [...]})
            for key, value in result.items():
                if isinstance(value, list):
                    return value
            # No list found — wrap the single object
            return [result]
    except (json.JSONDecodeError, TypeError):
        pass



    # Strategy 2: markdown fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: bracket matching for first top-level array
    start = raw.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(raw[start : i + 1])
                        if isinstance(result, list):
                            return result
                    except (json.JSONDecodeError, TypeError):
                        break

    # Strategy 4: brace matching for single object — wrap in list
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(raw[start : i + 1])
                        if isinstance(result, dict):
                            return [result]
                    except (json.JSONDecodeError, TypeError):
                        break

    raise ValueError(f"Could not parse JSON array from LLM response: {raw[:200]}")

