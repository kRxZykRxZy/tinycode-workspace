"""Stopping criteria for the training loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["StoppingCriteria", "StopReason", "TrainingStopper"]


class StopReason:
    """Canonical stop reasons."""

    COMPLETED = "completed"
    TOKEN_TARGET = "token_target_reached"
    TIME_LIMIT = "time_limit"
    STEP_LIMIT = "step_limit"
    NO_IMPROVEMENT = "no_improvement"
    RESOURCE_PRESSURE = "resource_pressure"
    EXTERNAL_REQUEST = "external_request"
    ERROR = "error"


@dataclass
class StoppingCriteria:
    """Thresholds that end a training run."""

    target_tokens: int = 200_000_000
    max_steps: int = 0
    max_hours: float = 0.0
    patience_evals: int = 0
    min_delta: float = 0.0


@dataclass
class TrainingStopper:
    """Evaluates the stopping criteria."""

    criteria: StoppingCriteria = field(default_factory=StoppingCriteria)
    started: float = field(default_factory=time.perf_counter)
    best_validation_loss: float = float("inf")
    evaluations_without_improvement: int = 0
    stopped: bool = False
    stop_reason: str = ""

    def check_tokens(self, tokens: int) -> str | None:
        if self.criteria.target_tokens and tokens >= self.criteria.target_tokens:
            return StopReason.TOKEN_TARGET
        return None

    def check_steps(self, step: int) -> str | None:
        if self.criteria.max_steps and step >= self.criteria.max_steps:
            return StopReason.STEP_LIMIT
        return None

    def check_time(self) -> str | None:
        if self.criteria.max_hours <= 0:
            return None
        elapsed_hours = (time.perf_counter() - self.started) / 3600.0
        if elapsed_hours >= self.criteria.max_hours:
            return StopReason.TIME_LIMIT
        return None

    def check_improvement(self, validation_loss: float) -> str | None:
        if validation_loss < self.best_validation_loss - self.criteria.min_delta:
            self.best_validation_loss = validation_loss
            self.evaluations_without_improvement = 0
            return None
        self.evaluations_without_improvement += 1
        if (
            self.criteria.patience_evals
            and self.evaluations_without_improvement >= self.criteria.patience_evals
        ):
            return StopReason.NO_IMPROVEMENT
        return None

    def evaluate(
        self,
        tokens: int,
        step: int,
        validation_loss: float | None = None,
        external_stop: Callable[[], bool] | None = None,
    ) -> str | None:
        """Return a stop reason when the run should end, otherwise ``None``."""

        if external_stop is not None and external_stop():
            return StopReason.EXTERNAL_REQUEST
        for reason in (
            self.check_tokens(tokens),
            self.check_steps(step),
            self.check_time(),
        ):
            if reason:
                return reason
        if validation_loss is not None:
            return self.check_improvement(validation_loss)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": self.criteria.__dict__,
            "best_validation_loss": self.best_validation_loss,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }
