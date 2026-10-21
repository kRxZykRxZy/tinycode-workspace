"""Runtime errors of the inference engine."""

from __future__ import annotations

from ..utils.errors import BackendError, EngineError, ModelError

__all__ = ["EngineError", "BackendError", "ModelError", "KernelUnavailableError"]


class KernelUnavailableError(BackendError):
    """Raised when a native kernel is not available."""
