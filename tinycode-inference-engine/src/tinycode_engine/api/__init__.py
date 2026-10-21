"""The HTTP API of the inference engine."""

from __future__ import annotations

from .app import create_app
from .authentication import ApiKeyStore
from .cache import router as cache_router
from .dependencies import get_engine, prepare_prompt, resolve_model
from .errors import install_error_handlers
from .health import router as health_router
from .middleware import install_middleware
from .usage import build_usage, merge_usage

__all__ = [
    "ApiKeyStore",
    "build_usage",
    "cache_router",
    "create_app",
    "get_engine",
    "health_router",
    "install_error_handlers",
    "install_middleware",
    "merge_usage",
    "prepare_prompt",
    "resolve_model",
]
