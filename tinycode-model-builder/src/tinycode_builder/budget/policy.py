"""Computation policies.

A policy describes *how much computation* a request may use.  Policies
are not context length tiers: the context always stays at 1024 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "POLICY_NAMES",
    "BudgetPolicy",
    "BUILTIN_POLICIES",
    "get_policy",
    "policy_to_dict",
]

POLICY_NAMES: tuple[str, ...] = ("minimal", "standard", "extended", "verification")


@dataclass
class BudgetPolicy:
    """Bounds and capabilities of one computation policy."""

    name: str
    min_generation_tokens: int = 64
    max_generation_tokens: int = 512
    max_internal_steps: int = 1
    allow_planning: bool = False
    allow_verification: bool = False
    max_verification_passes: int = 0
    target_confidence: float = 0.0
    description: str = ""

    def clamp_tokens(self, requested: int) -> int:
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
            "target_confidence": self.target_confidence,
            "description": self.description,
        }


BUILTIN_POLICIES: dict[str, BudgetPolicy] = {
    "minimal": BudgetPolicy(
        name="minimal",
        min_generation_tokens=32,
        max_generation_tokens=128,
        max_internal_steps=1,
        allow_planning=False,
        allow_verification=False,
        description="Cheapest supported policy: a single generation pass.",
    ),
    "standard": BudgetPolicy(
        name="standard",
        min_generation_tokens=64,
        max_generation_tokens=384,
        max_internal_steps=2,
        allow_planning=False,
        allow_verification=False,
        description="Default policy: one pass plus a bounded continuation.",
    ),
    "extended": BudgetPolicy(
        name="extended",
        min_generation_tokens=128,
        max_generation_tokens=768,
        max_internal_steps=4,
        allow_planning=True,
        allow_verification=False,
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


def get_policy(name: str) -> BudgetPolicy:
    """Return a builtin policy by name."""

    key = (name or "standard").lower()
    if key not in BUILTIN_POLICIES:
        raise KeyError(f"Unknown budget policy: {name!r}. Known policies: {sorted(BUILTIN_POLICIES)}")
    return BUILTIN_POLICIES[key]


def policy_to_dict(policy: BudgetPolicy) -> dict[str, Any]:
    """Serialise a policy."""

    return policy.to_dict()
