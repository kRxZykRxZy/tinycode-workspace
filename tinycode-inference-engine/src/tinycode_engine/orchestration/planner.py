"""Layer 4: planning and task decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..budget.request_analyzer import AnalyzedRequest

__all__ = ["PlanStep", "ExecutionPlan", "TaskDecomposer", "Planner"]


@dataclass
class PlanStep:
    """A single step of an execution plan."""

    action: str
    detail: str = ""
    tokens_hint: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "detail": self.detail, "tokens_hint": self.tokens_hint}


@dataclass
class ExecutionPlan:
    """A bounded execution plan."""

    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "rationale": self.rationale,
        }


class TaskDecomposer:
    """Splits a request into a small number of bounded subtasks."""

    MAX_SUBTASKS = 4

    def decompose(self, request: AnalyzedRequest) -> list[str]:
        """Return the subtasks of a request."""

        explicit = [
            line.strip("-* \t")
            for line in request.prompt.splitlines()
            if line.strip().startswith(("-", "*", "1.", "2.", "3.", "4."))
        ]
        if explicit:
            return explicit[: self.MAX_SUBTASKS]
        if request.subtask_count > 1:
            return [f"part {index + 1}" for index in range(min(self.MAX_SUBTASKS, request.subtask_count))]
        return []


class Planner:
    """Creates a compact plan for requests that benefit from one.

    The plan is bounded by the request budget and is never generated for
    trivial requests.
    """

    MAX_STEPS = 6

    def __init__(self, decomposer: TaskDecomposer | None = None) -> None:
        self.decomposer = decomposer or TaskDecomposer()

    def should_plan(self, request: AnalyzedRequest, complexity: float) -> bool:
        """Return whether a plan is warranted."""

        if request.subtask_count <= 1 and complexity < 0.55:
            return False
        return True

    def create_plan(
        self,
        request: AnalyzedRequest,
        complexity: float,
        max_tokens: int = 256,
    ) -> ExecutionPlan:
        """Create a bounded execution plan."""

        plan = ExecutionPlan()
        subtasks = self.decomposer.decompose(request)

        plan.steps.append(PlanStep("identify_request", "restate what the user asked for", 16))
        if subtasks:
            for index, subtask in enumerate(subtasks[: self.MAX_STEPS], start=1):
                plan.steps.append(PlanStep("subtask", f"{index}. {subtask}", 32))
        plan.steps.append(PlanStep("generate", "produce the candidate answer", max(32, max_tokens // 2)))
        if request.wants_verification:
            plan.steps.append(PlanStep("verify", "run the configured structural checks", 16))
        plan.steps.append(PlanStep("finalise", "assemble the final answer", 32))

        if len(plan.steps) > self.MAX_STEPS:
            plan.steps = plan.steps[: self.MAX_STEPS]
        plan.rationale = (
            f"complexity={complexity:.2f} subtasks={len(subtasks)} "
            f"tokens={max_tokens}"
        )
        return plan
