"""Exception hierarchy of the inference engine."""

from __future__ import annotations

__all__ = [
    "EngineError",
    "ConfigurationError",
    "ArtifactError",
    "ModelError",
    "BackendError",
    "RequestError",
    "SchedulingError",
    "BudgetError",
]


class EngineError(RuntimeError):
    """Base class for engine errors."""


class ConfigurationError(EngineError):
    """Invalid configuration."""


class ArtifactError(EngineError):
    """The model artifact is missing, malformed or incompatible."""


class ModelError(EngineError):
    """The model could not be executed."""


class BackendError(EngineError):
    """A hardware backend failed."""


class RequestError(EngineError):
    """The request could not be served."""


class SchedulingError(EngineError):
    """The scheduler could not admit or run the request."""


class BudgetError(EngineError):
    """The computation budget could not be honoured."""
