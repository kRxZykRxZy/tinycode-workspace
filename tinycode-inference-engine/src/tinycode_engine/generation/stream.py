"""Streaming helpers for the OpenAI compatible API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["SSEWriter", "format_sse", "chat_chunk", "completion_chunk"]


def format_sse(data: Any, event: str | None = None) -> str:
    """Format a payload as a server-sent event."""

    payload = json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


@dataclass
class SSEWriter:
    """Serialises chunks into the SSE wire format."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, data: Any, event: str | None = None) -> str:
        chunk = format_sse(data, event)
        self.chunks.append(chunk)
        return chunk

    def done(self) -> str:
        chunk = "data: [DONE]\n\n"
        self.chunks.append(chunk)
        return chunk

    def __iter__(self) -> Iterator[str]:
        return iter(self.chunks)


def chat_chunk(
    model_id: str,
    request_id: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    created: int = 0,
) -> dict[str, Any]:
    """Build an OpenAI compatible chat completion chunk."""

    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return chunk


def completion_chunk(
    model_id: str,
    request_id: str,
    text: str,
    finish_reason: str | None = None,
    created: int = 0,
    logprobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI compatible completion chunk."""

    choice: dict[str, Any] = {
        "index": 0,
        "text": text,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
    }
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [choice],
    }
