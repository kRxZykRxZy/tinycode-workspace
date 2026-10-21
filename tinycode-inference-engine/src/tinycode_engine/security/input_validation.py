"""Input validation of the API layer."""

from __future__ import annotations

from typing import Any

from ..protocol.errors import APIError

__all__ = ["validate_text_input", "sanitize_metadata", "MAX_TEXT_LENGTH"]

#: The maximum length of a single text input.
MAX_TEXT_LENGTH = 128 * 1024


def validate_text_input(text: Any, name: str = "input") -> str:
    """Validate a text input."""

    if not isinstance(text, str):
        raise APIError(f"'{name}' must be a string.", "invalid_request_error", status_code=400)
    if len(text) > MAX_TEXT_LENGTH:
        raise APIError(
            f"'{name}' is longer than the supported {MAX_TEXT_LENGTH} characters.",
            "invalid_request_error",
            status_code=400,
        )
    return text


def sanitize_metadata(metadata: Any) -> dict[str, Any]:
    """Keep only small, scalar metadata values."""

    if not isinstance(metadata, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in list(metadata.items())[:32]:
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key[:64]] = value if not isinstance(value, str) else value[:512]
    return clean
