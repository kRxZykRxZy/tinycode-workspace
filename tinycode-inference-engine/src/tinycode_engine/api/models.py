"""Model discovery endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..protocol.response_serialization import serialize_models
from .dependencies import get_api_keys, get_engine

__all__ = ["router", "list_models"]

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models(
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
) -> dict[str, Any]:
    """List the models that are loaded into the server."""

    api_keys.authenticate(request.headers.get("authorization"))
    return serialize_models(engine.model_ids())
