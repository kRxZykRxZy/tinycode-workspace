"""API key authentication for the HTTP layer.

The primitives live in :mod:`tinycode_engine.security.api_keys`; this
module adapts them to the API (it raises the OpenAI compatible error).
"""

from __future__ import annotations

from typing import Any

from ..security.api_keys import (
    ApiKeyStore,
    extract_bearer_token,
    generate_api_key,
    load_api_keys,
    write_api_key,
)

__all__ = [
    "ApiKeyStore",
    "extract_bearer_token",
    "generate_api_key",
    "load_api_keys",
    "verify_token",
    "write_api_key",
]


def verify_token(candidate: str | None, expected: str | None) -> bool:
    """Compare two tokens in constant time."""

    import hmac

    if not candidate or not expected:
        return False
    return hmac.compare_digest(str(candidate), str(expected))
