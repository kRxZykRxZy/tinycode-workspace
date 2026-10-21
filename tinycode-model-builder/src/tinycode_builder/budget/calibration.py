"""Calibration of uncertainty signals.

The signals produced by the inference engine (entropy, token agreement,
verification outcome) are *heuristics*, not calibrated probabilities.  The
calibration module makes that explicit and provides a simple Platt style
rescaling that can be fitted on recorded data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = ["CalibrationModel", "fit_calibration", "apply_calibration", "calibration_report"]


@dataclass
class CalibrationModel:
    """A two parameter logistic calibration ``p = sigmoid(a * s + b)``."""

    a: float = 1.0
    b: float = 0.0
    samples: int = 0
    brier_score: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "samples": self.samples,
            "brier_score": self.brier_score,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationModel":
        return cls(
            a=float(payload.get("a", 1.0)),
            b=float(payload.get("b", 0.0)),
            samples=int(payload.get("samples", 0)),
            brier_score=payload.get("brier_score"),
            notes=list(payload.get("notes", []) or []),
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def apply_calibration(score: float, model: CalibrationModel) -> float:
    """Map a raw signal onto a calibrated probability."""

    return _sigmoid(model.a * float(score) + model.b)


def fit_calibration(
    scores: Sequence[float],
    outcomes: Sequence[float],
    iterations: int = 500,
    learning_rate: float = 0.05,
) -> CalibrationModel:
    """Fit a Platt style calibration on recorded ``(score, outcome)`` pairs.

    The fit uses plain gradient descent; it is deterministic and small
    enough to run inside the builder.
    """

    if len(scores) != len(outcomes):
        raise ValueError("scores and outcomes must have the same length")
    if not scores:
        return CalibrationModel(notes=["no calibration data"])

    a = 1.0
    b = 0.0
    n = float(len(scores))
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for score, outcome in zip(scores, outcomes):
            prediction = _sigmoid(a * float(score) + b)
            error = prediction - float(outcome)
            grad_a += error * float(score)
            grad_b += error
        grad_a /= n
        grad_b /= n
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b

    brier = sum(
        (_sigmoid(a * float(score) + b) - float(outcome)) ** 2 for score, outcome in zip(scores, outcomes)
    ) / n

    model = CalibrationModel(a=a, b=b, samples=len(scores), brier_score=brier)
    model.notes.append(
        "Calibration maps a heuristic signal onto a probability; it does not make the "
        "signal a measure of factual correctness."
    )
    return model


def calibration_report(model: CalibrationModel, scores: Sequence[float]) -> dict[str, Any]:
    """Summarise a calibration model over a set of scores."""

    calibrated = [apply_calibration(score, model) for score in scores]
    return {
        "model": model.to_dict(),
        "samples": len(calibrated),
        "mean_calibrated": (sum(calibrated) / len(calibrated)) if calibrated else 0.0,
        "min_calibrated": min(calibrated) if calibrated else 0.0,
        "max_calibrated": max(calibrated) if calibrated else 0.0,
    }
