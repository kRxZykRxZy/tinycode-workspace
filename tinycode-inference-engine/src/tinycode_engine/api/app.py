"""The FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from ..observability.metrics import MetricsCollector
from ..runtime.engine import EngineConfig, InferenceEngine
from ..security.api_keys import ApiKeyStore
from ..security.rate_limits import RateLimiter
from ..security.request_limits import RequestLimits
from . import cache, chat_completions, completions, embeddings, health, models, responses
from .errors import install_error_handlers
from .middleware import install_middleware

__all__ = ["create_app"]


def create_app(
    engine: InferenceEngine | None = None,
    config: EngineConfig | None = None,
) -> FastAPI:
    """Create the FastAPI application of the inference engine."""

    config = config or EngineConfig()
    engine = engine or InferenceEngine(config)

    app = FastAPI(
        title="TinyCode Inference Engine",
        version="0.1.0",
        description=(
            "A local, OpenAI compatible inference server for TinyCode models. "
            "The context length is fixed at 1024 tokens."
        ),
    )

    app.state.engine = engine
    app.state.config = config
    app.state.api_keys = ApiKeyStore(
        keys=list(config.api_keys), required=bool(config.require_api_key)
    )
    app.state.rate_limiter = RateLimiter(requests_per_minute=config.rate_limit_per_minute)
    app.state.request_limits = RequestLimits(
        max_body_bytes=config.request_body_limit_bytes,
        max_prompt_tokens=1024,
        max_output_tokens=1024,
    )
    app.state.metrics = MetricsCollector(engine)

    install_middleware(app, app.state.request_limits)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(chat_completions.router)
    app.include_router(embeddings.router)
    app.include_router(responses.router)
    app.include_router(cache.router)

    @app.get("/metrics", tags=["observability"])
    async def metrics() -> str:
        """Expose the runtime metrics in the Prometheus text format."""

        return app.state.metrics.render()

    @app.get("/", tags=["service"])
    async def root() -> dict[str, Any]:
        return {
            "service": "tinycode-inference-engine",
            "version": "0.1.0",
            "endpoints": [
                "GET /v1/models",
                "POST /v1/completions",
                "POST /v1/chat/completions",
                "POST /v1/embeddings",
                "POST /v1/responses",
                "POST /v1/cache/input",
                "GET /v1/cache",
                "DELETE /v1/cache",
                "GET /health",
                "GET /health/ready",
                "GET /metrics",
            ],
            "context_length": 1024,
        }

    return app
