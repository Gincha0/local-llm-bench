"""Output quality where there is one right answer: checked by code on every repetition.

The structured regime asks for four fields from a switch log, and only one set of values
is correct. Free-text answers have no single right answer and are graded by hand.
"""

from __future__ import annotations

import json
import re


def json_matches(answer: str, expected: dict[str, str]) -> bool:
    """Valid JSON, exactly the expected keys, and the expected values.

    Case, spaces, hyphens and underscores are ignored, so "link-down" counts as
    "link down": the check is whether the model found the right event, not how it
    spelled it.
    """
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict) or set(data) != set(expected):
        return False
    return all(_normal(data[key]) == _normal(value) for key, value in expected.items())


def _normal(value: object) -> str:
    return re.sub(r"[\s_-]+", " ", str(value)).strip().lower()
