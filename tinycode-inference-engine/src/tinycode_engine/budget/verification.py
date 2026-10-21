"""Layer 8: verification.

The verifiers are *structural* checks.  Generated code is never executed:
that would require an isolated sandbox which the engine deliberately does
not provide.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["VerificationResult", "Verifier", "VerificationPipeline", "CHECKS"]

_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@dataclass
class VerificationResult:
    """The outcome of a verification run."""

    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": list(self.checks),
            "errors": list(self.errors),
            "score": round(self.score, 4),
            "note": "structural checks only; the response was not executed or proven correct",
        }


class Verifier:
    """A collection of structural checks."""

    def __init__(self, expected_json: bool = False, require_code_fence: bool = False) -> None:
        self.expected_json = expected_json
        self.require_code_fence = require_code_fence

    # ------------------------------------------------------------------
    def check_json(self, text: str) -> tuple[bool, str]:
        """Validate that the response contains a parsable JSON object."""

        candidates = [text.strip()]
        fenced = _CODE_FENCE_RE.findall(text)
        for _language, body in fenced:
            candidates.append(body.strip())
        for candidate in candidates:
            if not candidate:
                continue
            try:
                json.loads(candidate)
                return True, ""
            except json.JSONDecodeError:
                continue
        return False, "no parsable JSON object was found in the response"

    def check_code_fence(self, text: str) -> tuple[bool, str]:
        """Check that the response contains a fenced code block."""

        if _CODE_FENCE_RE.search(text):
            return True, ""
        return False, "the response does not contain a fenced code block"

    def check_non_empty(self, text: str) -> tuple[bool, str]:
        """Check that the response is not empty."""

        if text and text.strip():
            return True, ""
        return False, "the response is empty"

    def check_unbalanced_fences(self, text: str) -> tuple[bool, str]:
        """Check that the code fences are balanced."""

        if text.count("```") % 2 == 0:
            return True, ""
        return False, "the response contains an unbalanced code fence"

    # ------------------------------------------------------------------
    def verify(self, text: str, task_category: str = "code") -> VerificationResult:
        """Run every applicable check."""

        result = VerificationResult()
        checks = [
            ("non_empty", self.check_non_empty),
            ("balanced_fences", self.check_unbalanced_fences),
        ]
        if self.expected_json or task_category == "structured":
            checks.append(("json", self.check_json))
        if self.require_code_fence or task_category == "code":
            checks.append(("code_fence", self.check_code_fence))

        for name, check in checks:
            passed, message = check(text)
            result.checks.append({"name": name, "passed": passed, "message": message})
            if not passed:
                result.errors.append(f"{name}: {message}")
        result.passed = not result.errors
        result.score = sum(1 for check in result.checks if check["passed"]) / max(1, len(result.checks))
        return result


class VerificationPipeline:
    """Runs the verification stage of a request."""

    def __init__(self, verifier: Verifier | None = None) -> None:
        self.verifier = verifier or Verifier()

    def verify(
        self,
        text: str,
        task_category: str = "code",
        expected_json: bool = False,
    ) -> VerificationResult:
        """Verify a candidate response."""

        verifier = Verifier(expected_json=expected_json, require_code_fence=self.verifier.require_code_fence)
        return verifier.verify(text, task_category=task_category)
