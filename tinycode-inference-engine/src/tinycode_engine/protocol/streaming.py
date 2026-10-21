"""Server-sent event streaming for the OpenAI compatible API."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator

from .openai_compat import now_seconds

__all__ = ["sse_event", "chat_stream_events", "completion_stream_events", "async_sse"]


def sse_event(data: Any) -> str:
    """Format a payload as a server-sent event."""

    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_stream_events(
    model_id: str,
    request_id: str,
    chunks: Iterator[Any],
    created: int | None = None,
) -> Iterator[str]:
    """Turn the generated chunks into chat completion events."""

    timestamp = created if created is not None else now_seconds()
    first = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield sse_event(first)

    for chunk in chunks:
        event = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model_id,
            "choices": [
                {"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}
            ],
        }
        yield sse_event(event)

    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield sse_event(final)
    yield "data: [DONE]\n\n"


def completion_stream_events(
    model_id: str,
    request_id: str,
    chunks: Iterator[Any],
    created: int | None = None,
) -> Iterator[str]:
    """Turn the generated chunks into text completion events."""

    timestamp = created if created is not None else now_seconds()
    for chunk in chunks:
        event = {
            "id": request_id,
            "object": "text_completion",
            "created": timestamp,
            "model": model_id,
            "choices": [
                {"index": 0, "text": chunk.text, "logprobs": None, "finish_reason": None}
            ],
        }
        yield sse_event(event)

    final = {
        "id": request_id,
        "object": "text_completion",
        "created": timestamp,
        "model": model_id,
        "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": "stop"}],
    }
    yield sse_event(final)
    yield "data: [DONE]\n\n"


async def async_sse(source: Iterator[str]) -> AsyncIterator[str]:
    """Convert a synchronous event iterator into an asynchronous one."""

    for event in source:
        yield event
