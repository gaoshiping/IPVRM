from __future__ import annotations

import hashlib
import json
from typing import Any


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_role(role: str) -> str:
    normalized = ROLE_MAP.get(str(role).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported role value: {role}")
    return normalized


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    if "role" in message and "content" in message:
        return {
            "role": normalize_role(message["role"]),
            "content": message["content"],
        }
    if "from" in message and "value" in message:
        return {
            "role": normalize_role(message["from"]),
            "content": message["value"],
        }
    raise ValueError(f"Unsupported message format: {message}")


def normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return [normalize_message(message) for message in prompt]
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def append_assistant_message(prompt: Any, response: str) -> list[dict[str, Any]]:
    messages = normalize_prompt(prompt)
    return messages + [{"role": "assistant", "content": response}]


def canonical_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(normalize_prompt(prompt), ensure_ascii=False, sort_keys=True)


def derive_prompt_id(record: dict[str, Any], prompt_key: str, prompt_id_key: str = "prompt_id") -> str:
    if prompt_id_key in record and record[prompt_id_key] is not None and str(record[prompt_id_key]):
        return str(record[prompt_id_key])
    prompt = record.get(prompt_key)
    if prompt is None:
        raise KeyError(f"Missing prompt field '{prompt_key}'.")
    digest = hashlib.sha1(canonical_prompt(prompt).encode("utf-8")).hexdigest()
    return digest[:16]