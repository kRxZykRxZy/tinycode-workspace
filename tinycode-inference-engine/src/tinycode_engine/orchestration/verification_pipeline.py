"""The verification stage of the orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..budget.verification import VerificationPipeline, VerificationResult

__all__ = ["VerificationOutcome", "VerificationStage"]


@dataclass
class VerificationOutcome:
    """The result of the verification stage."""

    result: VerificationResult
    justification: str = ""
    another_attempt_justified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "justification": self.justification,
            "another_attempt_justified": self.another_attempt_justified,
        }


class VerificationStage:
    """Runs the verification stage when the budget allows it."""

    def __init__(self, pipeline: VerificationPipeline | None = None) -> None:
        self.pipeline = pipeline or VerificationPipeline()

    def run(
        self,
        text: str,
        task_category: str = "code",
        expected_json: bool = False,
        passes_remaining: int = 0,
    ) -> VerificationOutcome:
        """Verify a candidate response."""

        if passes_remaining <= 0:
            return VerificationOutcome(
                result=VerificationResult(passed=True, checks=[], errors=[], score=1.0),
                justification="no verification budget was allocated",
                another_attempt_justified=False,
            )

        result = self.pipeline.verify(text, task_category=task_category, expected_json=expected_json)
        if result.passed:
            justification = "the structural checks passed"
            another = False
        else:
            justification = "; ".join(result.errors)
            another = True
        return VerificationOutcome(
            result=result,
            justification=justification,
            another_attempt_justified=another,
        )
