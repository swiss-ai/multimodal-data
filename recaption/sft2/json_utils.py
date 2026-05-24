from __future__ import annotations

import json
import re
from typing import Any


def repair_json_fragment(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                if char not in '"\\/bfnrtu':
                    repaired.append("\\")
                repaired.append(char)
                escaped = False
                continue

            if char == "\\":
                repaired.append(char)
                escaped = True
                continue
            if char == '"':
                repaired.append(char)
                in_string = False
                continue
            if char == "\n":
                repaired.append("\\n")
                continue
            if char == "\r":
                repaired.append("\\r")
                continue
            if char == "\t":
                repaired.append("\\t")
                continue
            repaired.append(char)
            continue

        repaired.append(char)
        if char == '"':
            in_string = True

    if escaped:
        repaired.append("\\")

    return "".join(repaired)


def extract_json_object(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    fragment = candidate[start : end + 1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return json.loads(repair_json_fragment(fragment))
