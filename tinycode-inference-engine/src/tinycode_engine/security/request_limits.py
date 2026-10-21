"""Request size limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..protocol.errors import APIError

__all__ = ["RequestLimits", "check_request_size"]


@dataclass
class RequestLimits:
    """The size limits of the server."""

    max_body_bytes: int = 1 << 20
    max_prompt_tokens: int = 1024
    max_output_tokens: int = 1024
    max_stop_sequences: int = 16
    max_messages: int = 256
    max_tools: int = 32

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_body_bytes": self.max_body_bytes,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_stop_sequences": self.max_stop_sequences,
            "max_messages": self.max_messages,
            "max_tools": self.max_tools,
        }


def check_request_size(content_length: int | str | None, limits: RequestLimits) -> None:
    """Reject a request whose body is too large."""

    if content_length is None:
        return
    try:
        size = int(content_length)
    except (TypeError, ValueError):
        return
    if size > limits.max_body_bytes:
        raise APIError(
            f"The request body is larger than the limit of {limits.max_body_bytes} bytes.",
            "invalid_request_error",
            status_code=413,
        )
