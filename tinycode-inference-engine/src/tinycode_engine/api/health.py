"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .dependencies import get_engine

__all__ = ["router", "health", "readiness"]

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe: the process is running."""

    return {"status": "ok", "service": "tinycode-inference-engine"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, Any]:
    """Readiness probe: a model is loaded and the engine can serve requests."""

    try:
        engine = get_engine(request)
    except Exception:
        return {"status": "not_ready", "reason": "the engine is not available"}
    models = engine.model_ids()
    if not models:
        return {"status": "not_ready", "reason": "no model is loaded"}
    backend = engine.backend_selection.to_dict() if engine.backend_selection else None
    return {
        "status": "ready",
        "models": models,
        "backend": backend.get("backend") if backend else None,
        "context_length": 1024,
    }
