"""Computation policies.

A policy bounds how much *computation* a request may use.  Policies are
not context tiers: the context always stays at 1024 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["BudgetPolicy", "POLICIES", "get_policy", "policy_names"]


@dataclass(frozen=True)
class BudgetPolicy:
    """The bounds of one computation policy."""

    name: str
    min_generation_tokens: int = 64
    max_generation_tokens: int = 512
    max_internal_steps: int = 1
    allow_planning: bool = False
    allow_verification: bool = False
    max_verification_passes: int = 0
    description: str = ""

    def clamp(self, requested: int) -> int:
        """Clamp a requested output budget into the policy bounds."""

        return int(max(self.min_generation_tokens, min(self.max_generation_tokens, requested)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_generation_tokens": self.min_generation_tokens,
            "max_generation_tokens": self.max_generation_tokens,
            "max_internal_steps": self.max_internal_steps,
            "allow_planning": self.allow_planning,
            "allow_verification": self.allow_verification,
            "max_verification_passes": self.max_verification_passes,
            "description": self.description,
        }


POLICIES: dict[str, BudgetPolicy] = {
    "minimal": BudgetPolicy(
        name="minimal",
        min_generation_tokens=32,
        max_generation_tokens=128,
        max_internal_steps=1,
        description="A single generation pass.",
    ),
    "standard": BudgetPolicy(
        name="standard",
        min_generation_tokens=64,
        max_generation_tokens=384,
        max_internal_steps=2,
        description="One pass plus a bounded continuation.",
    ),
    "extended": BudgetPolicy(
        name="extended",
        min_generation_tokens=128,
        max_generation_tokens=768,
        max_internal_steps=4,
        allow_planning=True,
        description="Planning enabled policy for complex requests.",
    ),
    "verification": BudgetPolicy(
        name="verification",
        min_generation_tokens=96,
        max_generation_tokens=640,
        max_internal_steps=4,
        allow_planning=True,
        allow_verification=True,
        max_verification_passes=1,
        description="Policy that funds a bounded verification stage.",
    ),
}


def policy_names() -> list[str]:
    """Return the names of the supported policies."""

    return list(POLICIES)


def get_policy(name: str) -> BudgetPolicy:
    """Return a policy by name."""

    key = (name or "standard").lower()
    if key not in POLICIES:
        raise KeyError(f"Unknown budget policy: {name!r}. Known: {policy_names()}")
    return POLICIES[key]
