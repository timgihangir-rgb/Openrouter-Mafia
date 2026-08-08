"""Helper functions for parsing model responses and validating targets."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(response: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Tries three strategies in order:
    1. Direct ``json.loads(response)``.
    2. Find first ``{`` → last ``}`` and parse the substring.
    3. Regex extraction of individual ``"thoughts"`` and action fields as a
       last resort, reconstructing a minimal dict.

    Raises:
        ValueError: if no valid JSON (or JSON-like content) can be found.
    """
    response = (response or "").strip()
    if not response:
        raise ValueError("Empty response from model")

    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        pass

    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = response[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

    result: dict[str, str] = {}
    for key in ("thoughts", "action", "target", "vote", "statement"):
        pattern = rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|\d+)'
        match = re.search(pattern, response)
        if match:
            raw = match.group(1)
            if raw.startswith('"'):
                result[key] = json.loads(raw)
            else:
                result[key] = raw

    if not result:
        raise ValueError(f"No valid JSON or extractable fields found in response: {response[:200]}")

    if "thoughts" not in result:
        result["thoughts"] = "[Unable to extract thoughts]"
    return result


def validate_target(target_raw: str | int, alive_players: list[int]) -> int:
    """Validate that *target_raw* refers to an alive player index.

    Accepts an integer or a string like ``"3"`` or ``"player_3"``.
    Returns the validated player index.

    Raises:
        ValueError: if the target is not alive or not a valid index.
    """
    if isinstance(target_raw, int):
        target = target_raw
    elif isinstance(target_raw, str):
        cleaned = target_raw.strip().lower().replace("player_", "").replace("player ", "").strip()
        try:
            target = int(cleaned)
        except ValueError:
            raise ValueError(f"Cannot parse target '{target_raw}' as an integer player index")
    else:
        raise ValueError(f"Target must be int or str, got {type(target_raw)}")

    if target not in alive_players:
        raise ValueError(f"Target {target} is not an alive player. Alive players: {alive_players}")
    return target


def validate_vote(vote_raw: str | int, alive_players: list[int]) -> int:
    """Validate a vote target. Same semantics as :func:`validate_target`."""
    return validate_target(vote_raw, alive_players)
