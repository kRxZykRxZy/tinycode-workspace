"""Layer 2: task and complexity analysis.

The estimator starts with a transparent, deterministic baseline.  A
learned estimator is only used when evaluation data justifies it, so the
default implementation stays explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .request_analyzer import AnalyzedRequest

__all__ = ["ComplexityEstimate", "ComplexityEstimator", "CATEGORY_PRIORS"]

#: The prior complexity of every task category.
CATEGORY_PRIORS: dict[str, float] = {
    "code": 0.45,
    "explain": 0.25,
    "transform": 0.40,
    "debug": 0.60,
    "structured": 0.35,
}


@dataclass
class ComplexityEstimate:
    """The estimated difficulty of a request."""

    score: float = 0.0
    task_category: str = "code"
    components: dict[str, float] = None  # type: ignore[assignment]
    confidence: float = 0.5
    method: str = "deterministic-baseline"

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "task_category": self.task_category,
            "components": {key: round(value, 6) for key, value in self.components.items()},
            "confidence": self.confidence,
            "method": self.method,
        }


class ComplexityEstimator:
    """Estimates the complexity of a request from transparent signals."""

    def __init__(self, history: dict[str, float] | None = None) -> None:
        #: Average measured complexity per task category (filled from telemetry).
        self.history = dict(history or {})

    def estimate(
        self,
        request: AnalyzedRequest,
        uncertainty: float = 0.0,
        resource_pressure: float = 0.0,
    ) -> ComplexityEstimate:
        """Estimate the complexity of a request."""

        category = request.task_category
        prior = self.history.get(category, CATEGORY_PRIORS.get(category, 0.4))

        components: dict[str, float] = {"category_prior": prior}

        # The prompt length is a weak signal: it is capped so that a long
        # prompt alone cannot dominate the estimate.
        length_signal = min(1.0, request.prompt_tokens / 1024.0) * 0.10
        components["length"] = length_signal

        # A large output budget suggests a bigger answer.
        output_signal = min(1.0, request.requested_output_tokens / 512.0) * 0.10
        components["requested_output"] = output_signal

        # Several requested subtasks make a task genuinely harder.
        subtask_signal = min(1.0, max(0, request.subtask_count - 1) / 4.0) * 0.15
        components["subtasks"] = subtask_signal

        # Asking for verification adds work.
        verification_signal = 0.08 if request.wants_verification else 0.0
        components["verification"] = verification_signal

        # Model uncertainty is a direct signal of difficulty.
        uncertainty_signal = max(0.0, min(1.0, uncertainty)) * 0.20
        components["uncertainty"] = uncertainty_signal

        # Under resource pressure the estimate is reduced: less compute is
        # available, so the plan has to stay cheap.
        pressure_signal = max(0.0, min(1.0, resource_pressure)) * -0.10
        components["resource_pressure"] = pressure_signal

        score = max(0.0, min(1.0, sum(components.values())))
        return ComplexityEstimate(
            score=score,
            task_category=category,
            components=components,
            confidence=0.6 if request.prompt_tokens else 0.4,
            method="deterministic-baseline",
        )
