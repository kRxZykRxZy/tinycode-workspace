"""Training objectives for the budget controller.

The objective is *not* token probability: it rewards allocating the
smallest budget that still reaches a good verified outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "ControllerTargets",
    "budget_regression_loss",
    "decision_loss",
    "combined_loss",
    "compute_efficiency_score",
]


@dataclass
class ControllerTargets:
    """Training targets for a single request."""

    budget_multiplier: float = 1.0
    plan: int = 0
    verify: int = 0
    stop: int = 0
    quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_multiplier": self.budget_multiplier,
            "plan": self.plan,
            "verify": self.verify,
            "stop": self.stop,
            "quality": self.quality,
        }


def budget_regression_loss(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Mean squared error on the log budget multiplier."""

    if not predictions:
        return 0.0
    total = 0.0
    for prediction, target in zip(predictions, targets):
        diff = float(prediction) - float(target)
        total += diff * diff
    return total / len(predictions)


def decision_loss(logits: Sequence[float], targets: Sequence[int]) -> float:
    """Binary cross entropy on a decision logit."""

    import math

    if not logits:
        return 0.0
    total = 0.0
    for logit, target in zip(logits, targets):
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(logit)))))
        probability = min(max(probability, 1e-6), 1 - 1e-6)
        total += -(target * math.log(probability) + (1 - target) * math.log(1 - probability))
    return total / len(logits)


def combined_loss(
    budget_predictions: Sequence[float],
    budget_targets: Sequence[float],
    decision_logits: Sequence[float],
    decision_targets: Sequence[int],
    budget_weight: float = 1.0,
    decision_weight: float = 0.5,
) -> float:
    """Weighted combination of the budget and decision objectives."""

    return budget_weight * budget_regression_loss(
        budget_predictions, budget_targets
    ) + decision_weight * decision_loss(decision_logits, decision_targets)


def compute_efficiency_score(
    allocated_tokens: int, consumed_tokens: int, quality: float, reference_quality: float
) -> float:
    """Score how efficiently compute was used.

    The score rewards quality per token and penalises allocating more
    budget than was used.
    """

    if consumed_tokens <= 0:
        return 0.0
    quality_ratio = quality / reference_quality if reference_quality > 0 else 0.0
    utilisation = consumed_tokens / allocated_tokens if allocated_tokens > 0 else 1.0
    return float(quality_ratio * utilisation)
