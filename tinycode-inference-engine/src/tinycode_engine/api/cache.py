"""The cache input route.

``POST /v1/cache/input`` warms the runtime caches for an input.  The
tokenised input is stored in the token cache and, by default, the KV
state of the input is computed once and stored in the prefix cache.  A
later request whose prompt starts with the cached input then only
computes the remaining tokens.

The cache key always contains the model identity, the model version and
the tokenizer fingerprint, so the state of an incompatible model is never
reused.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..model.tokenizer_adapter import chat_prompt
from ..protocol.errors import APIError
from ..utils.errors import ArtifactError, RequestError
from .dependencies import client_identifier, get_api_keys, get_engine, get_rate_limiter
from .schemas.cache import CacheInputRequest

__all__ = ["router", "cache_input", "cache_stats", "clear_cache"]

router = APIRouter(tags=["cache"])


@router.post("/v1/cache/input")
async def cache_input(
    payload: CacheInputRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
    rate_limiter: Any = Depends(get_rate_limiter),
) -> dict[str, Any]:
    """Cache an input so that later requests can reuse it."""

    api_keys.authenticate(request.headers.get("authorization"))
    rate_limiter.enforce(client_identifier(request))

    text = payload.input
    if payload.messages:
        messages = [
            {"role": message.role, "content": message.content or ""} for message in payload.messages
        ]
        rendered = chat_prompt(messages)
        text = f"{text}\n{rendered}" if text else rendered
    if not text:
        raise APIError(
            "Either 'input' or 'messages' must be provided.",
            "invalid_request_error",
            status_code=400,
        )

    try:
        summary = engine.cache_input(text, model_id=payload.model, compute_prefix=payload.prefix)
    except RequestError as exc:
        raise APIError(str(exc), "invalid_request_error", status_code=400) from exc
    except ArtifactError as exc:
        raise APIError(str(exc), "not_found_error", status_code=404) from exc

    return {
        "object": "cache.entry",
        "cached": True,
        "model": summary["model"],
        "tokens": summary["tokens"],
        "token_ids": summary["token_ids"],
        "prefix_state_stored": summary["prefix_state_stored"],
        "compute_ms": summary["compute_ms"],
        "stats": engine.cache_stats(),
    }


@router.get("/v1/cache")
async def cache_stats(
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
) -> dict[str, Any]:
    """Return the statistics of the runtime caches."""

    api_keys.authenticate(request.headers.get("authorization"))
    return {"object": "cache.stats", **engine.cache_stats()}


@router.delete("/v1/cache")
async def clear_cache(
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
) -> dict[str, Any]:
    """Clear the runtime caches."""

    api_keys.authenticate(request.headers.get("authorization"))
    stats = engine.clear_cache()
    return {"object": "cache.stats", "cleared": True, **stats}
