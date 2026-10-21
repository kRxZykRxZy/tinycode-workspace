"""Fallback handling.

The engine falls back to the CPU backend when an accelerator cannot be
initialised.  Model state is never migrated silently between backends:
the fallback always happens before the model is placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..backends.interface import Backend

__all__ = ["FallbackPlan", "FallbackPolicy"]


@dataclass
class FallbackPlan:
    """What the engine will do when the preferred backend fails."""

    preferred: str = "auto"
    fallback: str = "cpu"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred": self.preferred,
            "fallback": self.fallback,
            "reason": self.reason,
        }


class FallbackPolicy:
    """Decides whether a fallback is allowed."""

    def __init__(self, allow_cpu_fallback: bool = True) -> None:
        self.allow_cpu_fallback = allow_cpu_fallback

    def can_fallback(self, error: BaseException | None = None) -> bool:
        """Return whether the engine may fall back to the CPU backend."""

        if not self.allow_cpu_fallback:
            return False
        if error is None:
            return True
        # Errors that indicate a broken model state are not recoverable by
        # switching the backend.
        message = str(error).lower()
        for fatal in ("missing the weight", "vocabulary", "architecture", "manifest"):
            if fatal in message:
                return False
        return True

    def plan(self, preferred: str, reason: str) -> FallbackPlan:
        """Return the fallback plan."""

        return FallbackPlan(preferred=preferred, fallback="cpu", reason=reason)
