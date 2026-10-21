"""Evaluation of the adaptive budget controller.

The controller is judged on *verified coding task performance under a
compute budget*, not on token probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .controller import BaselineBudgetController, BudgetController, ControllerDecision
from .features import BudgetFeatures
from .policy_dataset import PolicyDataset

__all__ = [
    "ControllerEvaluation",
    "evaluate_controller",
    "compare_controllers",
    "write_evaluation_report",
]


@dataclass
class ControllerEvaluation:
    """Aggregate metrics of a controller on a policy dataset."""

    name: str = "controller"
    samples: int = 0
    budget_mae: float = 0.0
    budget_bias: float = 0.0
    decision_accuracy: float = 0.0
    mean_allocated_tokens: float = 0.0
    mean_efficiency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "samples": self.samples,
            "budget_mae": round(self.budget_mae, 6),
            "budget_bias": round(self.budget_bias, 6),
            "decision_accuracy": round(self.decision_accuracy, 4),
            "mean_allocated_tokens": round(self.mean_allocated_tokens, 3),
            "mean_efficiency": round(self.mean_efficiency, 6),
        }


def _decisions_for(
    controller: BudgetController, dataset: PolicyDataset, requested_tokens: int
) -> list[ControllerDecision]:
    return [
        controller.decide(sample.features, requested_tokens) for sample in dataset.samples
    ]


def evaluate_controller(
    controller: BudgetController,
    dataset: PolicyDataset,
    requested_tokens: int = 256,
    reference_quality: float = 1.0,
) -> ControllerEvaluation:
    """Evaluate a controller against a policy dataset."""

    from .training_objectives import compute_efficiency_score

    if not dataset.samples:
        return ControllerEvaluation(name=controller.describe().get("kind", "controller"))

    decisions = _decisions_for(controller, dataset, requested_tokens)
    errors: list[float] = []
    biases: list[float] = []
    correct = 0
    total_decisions = 0
    allocated: list[int] = []
    efficiencies: list[float] = []

    for sample, decision in zip(dataset.samples, decisions):
        expected_multiplier = sample.targets.budget_multiplier
        actual_multiplier = decision.generation_tokens / max(1, requested_tokens)
        errors.append(abs(actual_multiplier - expected_multiplier))
        biases.append(actual_multiplier - expected_multiplier)

        predicted_plan = 1 if decision.allow_planning else 0
        predicted_verify = 1 if decision.verification_passes > 0 else 0
        total_decisions += 2
        correct += int(predicted_plan == sample.targets.plan)
        correct += int(predicted_verify == sample.targets.verify)

        allocated.append(decision.generation_tokens)
        qualities = [sample.targets.quality]
        consumed = min(decision.generation_tokens, max(1, int(requested_tokens * expected_multiplier)))
        efficiencies.append(
            compute_efficiency_score(
                decision.generation_tokens, consumed, qualities[0], reference_quality
            )
        )

    return ControllerEvaluation(
        name=controller.describe().get("kind", "controller"),
        samples=len(dataset.samples),
        budget_mae=sum(errors) / len(errors),
        budget_bias=sum(biases) / len(biases),
        decision_accuracy=correct / total_decisions if total_decisions else 0.0,
        mean_allocated_tokens=sum(allocated) / len(allocated),
        mean_efficiency=sum(efficiencies) / len(efficiencies),
    )


def compare_controllers(
    controllers: dict[str, BudgetController],
    dataset: PolicyDataset,
    requested_tokens: int = 256,
) -> dict[str, Any]:
    """Evaluate several controllers and compare them."""

    results = {
        name: evaluate_controller(controller, dataset, requested_tokens).to_dict()
        for name, controller in controllers.items()
    }
    baseline = results.get("baseline")
    payload: dict[str, Any] = {"results": results, "baseline": baseline}
    if baseline:
        payload["improvement"] = {
            name: round(
                result["mean_efficiency"] - baseline["mean_efficiency"], 6
            )
            for name, result in results.items()
        }
    return payload


def write_evaluation_report(payload: dict[str, Any], path: str | Path) -> Path:
    """Write an evaluation report to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def default_controllers() -> dict[str, BudgetController]:
    """Return the baseline controller used when no trained controller exists."""

    return {"baseline": BaselineBudgetController()}
