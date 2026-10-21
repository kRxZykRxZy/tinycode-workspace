"""Health reporting of the engine."""

from __future__ import annotations

from typing import Any

__all__ = ["health_report", "readiness_report"]


def health_report(engine: Any = None) -> dict[str, Any]:
    """Return the liveness information."""

    return {
        "status": "ok",
        "service": "tinycode-inference-engine",
        "context_length": 1024,
    }


def readiness_report(engine: Any = None) -> dict[str, Any]:
    """Return the readiness information."""

    if engine is None:
        return {"status": "not_ready", "reason": "no engine"}
    models = engine.model_ids()
    if not models:
        return {"status": "not_ready", "reason": "no model is loaded"}
    backend = engine.backend_selection.name if engine.backend_selection else "none"
    return {
        "status": "ready",
        "models": models,
        "backend": backend,
        "context_length": 1024,
    }
