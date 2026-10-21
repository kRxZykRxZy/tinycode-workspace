"""OpenAI compatibility constants and helpers."""

from __future__ import annotations

import time
from typing import Any

__all__ = [
    "API_VERSION",
    "DEFAULT_MODEL_ID",
    "OBJECT_MODEL",
    "OBJECT_CHAT_COMPLETION",
    "OBJECT_COMPLETION",
    "OBJECT_LIST",
    "ERROR_TYPES",
    "chat_completion_payload",
    "completion_payload",
    "model_payload",
    "model_list_payload",
    "embeddings_payload",
    "usage_payload",
    "now_seconds",
]

API_VERSION = "v1"
DEFAULT_MODEL_ID = "tinycode-local"

OBJECT_MODEL = "model"
OBJECT_LIST = "list"
OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_COMPLETION = "text_completion"

ERROR_TYPES = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_error": 429,
    "server_error": 500,
    "service_unavailable": 503,
}


def now_seconds() -> int:
    """Return the current unix timestamp in seconds."""

    return int(time.time())


def usage_payload(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI compatible usage object."""

    total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total),
    }


def chat_completion_payload(
    model_id: str,
    request_id: str,
    content: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    created: int | None = None,
    role: str = "assistant",
) -> dict[str, Any]:
    """Build an OpenAI compatible chat completion response."""

    return {
        "id": request_id,
        "object": OBJECT_CHAT_COMPLETION,
        "created": created if created is not None else now_seconds(),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": content},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_payload(prompt_tokens, completion_tokens),
    }


def completion_payload(
    model_id: str,
    request_id: str,
    text: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    created: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI compatible completion response."""

    return {
        "id": request_id,
        "object": OBJECT_COMPLETION,
        "created": created if created is not None else now_seconds(),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_payload(prompt_tokens, completion_tokens),
    }


def model_payload(model_id: str, owned_by: str = "tinycode", created: int | None = None) -> dict[str, Any]:
    """Build a single model description."""

    return {
        "id": model_id,
        "object": OBJECT_MODEL,
        "created": created if created is not None else now_seconds(),
        "owned_by": owned_by,
    }


def model_list_payload(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ``/v1/models`` response."""

    return {"object": OBJECT_LIST, "data": models}


def embeddings_payload(
    model_id: str,
    vectors: list[list[float]],
    prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Build an OpenAI compatible embeddings response."""

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": model_id,
        "usage": usage_payload(prompt_tokens, 0),
    }
