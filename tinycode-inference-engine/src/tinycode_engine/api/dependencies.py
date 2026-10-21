"""Dependencies of the API layer."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from ..protocol.errors import APIError
from ..runtime.engine import InferenceEngine
from ..security.rate_limits import RateLimiter
from ..security.request_limits import RequestLimits

__all__ = [
    "get_engine",
    "get_api_keys",
    "get_rate_limiter",
    "get_limits",
    "resolve_model",
    "prepare_prompt",
    "client_identifier",
]


def get_engine(request: Request) -> InferenceEngine:
    """Return the inference engine of the application."""

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise APIError("The engine is not available.", "server_error", status_code=503)
    return engine


def get_api_keys(request: Request) -> Any:
    """Return the API key store of the application."""

    store = getattr(request.app.state, "api_keys", None)
    if store is None:
        raise APIError("The server is not configured.", "server_error", status_code=503)
    return store


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the rate limiter of the application."""

    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return RateLimiter(requests_per_minute=0)
    return limiter


def get_limits(request: Request) -> RequestLimits:
    """Return the request limits of the application."""

    limits = getattr(request.app.state, "request_limits", None)
    if limits is None:
        return RequestLimits()
    return limits


def client_identifier(request: Request) -> str:
    """Return the identifier used for rate limiting."""

    api_key = request.headers.get("authorization", "")
    if api_key:
        import hashlib

        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    client = request.client
    return client.host if client else "unknown"


def resolve_model(engine: InferenceEngine, model_id: str) -> str:
    """Check that the requested model is loaded."""

    available = engine.model_ids()
    if not available:
        raise APIError(
            "No model is loaded into the server.",
            "service_unavailable",
            status_code=503,
        )
    if model_id not in available:
        raise APIError(
            f"The model '{model_id}' is not loaded. Available models: {', '.join(available)}.",
            "not_found_error",
            status_code=404,
        )
    return model_id


def prepare_prompt(
    engine: InferenceEngine,
    text: str,
    reserved_output_tokens: int,
) -> tuple[list[int], int]:
    """Tokenise a prompt and make it fit into the fixed context.

    Returns:
        ``(token_ids, prompt_token_count)``.
    """

    if engine.tokenizer is None:
        raise APIError("No tokenizer is loaded.", "service_unavailable", status_code=503)

    limit = 1024 - max(0, reserved_output_tokens)
    if limit <= 0:
        raise APIError(
            "The requested completion does not fit into the fixed 1024 token context.",
            "invalid_request_error",
            status_code=400,
        )

    token_ids = engine.tokenizer.encode(text)
    if len(token_ids) > limit:
        # Truncate from the left so that the instruction at the end of a
        # chat prompt survives.
        token_ids = token_ids[-limit:]
    return token_ids, len(token_ids)
