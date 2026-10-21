"""Usage accounting helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["build_usage", "merge_usage"]


def build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    """Build an OpenAI compatible usage object."""

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }


def merge_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """Merge several usage objects."""

    prompt = 0
    completion = 0
    for usage in usages:
        if not usage:
            continue
        prompt += int(usage.get("prompt_tokens", 0))
        completion += int(usage.get("completion_tokens", 0))
    return build_usage(prompt, completion)
