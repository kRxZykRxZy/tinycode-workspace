"""Exception hierarchy for the model builder."""

from __future__ import annotations

__all__ = [
    "TinyCodeBuilderError",
    "ConfigurationError",
    "HardwareError",
    "DataError",
    "TokenizerError",
    "TrainingError",
    "CheckpointError",
    "ExportError",
    "ArtifactError",
    "BudgetError",
]


class TinyCodeBuilderError(RuntimeError):
    """Base class for all builder errors."""


class ConfigurationError(TinyCodeBuilderError):
    """Invalid or contradictory configuration."""


class HardwareError(TinyCodeBuilderError):
    """Hardware detection or capability reporting failure."""


class DataError(TinyCodeBuilderError):
    """Dataset acquisition, filtering or packing failure."""


class TokenizerError(TinyCodeBuilderError):
    """Tokenizer training or loading failure."""


class TrainingError(TinyCodeBuilderError):
    """Training loop failure."""


class CheckpointError(TinyCodeBuilderError):
    """Checkpoint save, load or integrity failure."""


class ExportError(TinyCodeBuilderError):
    """Model export failure."""


class ArtifactError(TinyCodeBuilderError):
    """Artifact manifest validation failure."""


class BudgetError(TinyCodeBuilderError):
    """Adaptive budget controller failure."""
