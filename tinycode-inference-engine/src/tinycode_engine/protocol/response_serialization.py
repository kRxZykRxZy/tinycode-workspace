"""Serialisation of the API responses."""

from __future__ import annotations

from typing import Any

from .openai_compat import (
    chat_completion_payload,
    completion_payload,
    embeddings_payload,
    model_list_payload,
    model_payload,
    now_seconds,
    usage_payload,
)

__all__ = [
    "serialize_chat_completion",
    "serialize_completion",
    "serialize_models",
    "serialize_embeddings",
    "serialize_responses_api",
]


def serialize_chat_completion(
    model_id: str,
    request_id: str,
    content: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Serialise a chat completion."""

    return chat_completion_payload(
        model_id=model_id,
        request_id=request_id,
        content=content,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        created=now_seconds(),
    )


def serialize_completion(
    model_id: str,
    request_id: str,
    text: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Serialise a text completion."""

    return completion_payload(
        model_id=model_id,
        request_id=request_id,
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        created=now_seconds(),
    )


def serialize_models(model_ids: list[str], owned_by: str = "tinycode") -> dict[str, Any]:
    """Serialise the model list."""

    return model_list_payload([model_payload(model_id, owned_by=owned_by) for model_id in model_ids])


def serialize_embeddings(
    model_id: str,
    vectors: list[list[float]],
    prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Serialise the embeddings response."""

    return embeddings_payload(model_id, vectors, prompt_tokens=prompt_tokens)


def serialize_responses_api(
    model_id: str,
    request_id: str,
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Serialise a minimal Responses API payload.

    Only the subset that the engine implements is returned; the endpoint
    documents its limitations.
    """

    return {
        "id": request_id,
        "object": "response",
        "created_at": now_seconds(),
        "model": model_id,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": usage_payload(prompt_tokens, completion_tokens),
    }
