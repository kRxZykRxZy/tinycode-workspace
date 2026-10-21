"""The ARM NEON backend."""

from __future__ import annotations

from .backend import ArmNeonBackend
from .capabilities import NeonCapabilities, detect_neon
from .operators import NeonOperators

__all__ = ["ArmNeonBackend", "NeonCapabilities", "NeonOperators", "detect_neon"]
