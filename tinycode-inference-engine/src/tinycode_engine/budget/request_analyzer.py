"""Layer 1: request intake.

The analyser validates the request, extracts the explicit constraints and
normalises everything into an internal representation.  The original
intent of the user is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..utils.errors import RequestError

__all__ = ["AnalyzedRequest", "RequestAnalyzer", "TASK_CATEGORY_PATTERNS", "CONTEXT_LENGTH"]

#: The fixed context length of the platform.
CONTEXT_LENGTH = 1024

TASK_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "debug": ("fix", "bug", "error", "traceback", "debug", "why does", "not working", "exception"),
    "transform": ("translate", "convert", "refactor", "rewrite", "port", "migrate", "rename"),
    "explain": ("explain", "what does", "how does", "describe", "summarise", "summarize", "why"),
    "structured": ("json", "yaml", "xml", "table", "schema", "csv", "format"),
    "code": ("write", "implement", "function", "class", "script", "algorithm", "code"),
}

_VERIFICATION_PATTERNS = ("verify", "check", "test", "validate", "ensure", "review", "prove")
_CODE_FENCE = re.compile(r"```")


@dataclass
class AnalyzedRequest:
    """The normalised description of a request."""

    request_id: str = ""
    prompt: str = ""
    prompt_tokens: int = 0
    task_category: str = "code"
    requested_output_tokens: int = 128
    wants_verification: bool = False
    latency_target_ms: float | None = None
    requested_compute_budget: int | None = None
    subtask_count: int = 1
    contains_code_fence: bool = False
    requested_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "task_category": self.task_category,
            "requested_output_tokens": self.requested_output_tokens,
            "wants_verification": self.wants_verification,
            "latency_target_ms": self.latency_target_ms,
            "requested_compute_budget": self.requested_compute_budget,
            "subtask_count": self.subtask_count,
            "contains_code_fence": self.contains_code_fence,
            "requested_tools": list(self.requested_tools),
        }


class RequestAnalyzer:
    """Turns a raw request into an :class:`AnalyzedRequest`."""

    def __init__(self, max_prompt_tokens: int = CONTEXT_LENGTH) -> None:
        self.max_prompt_tokens = max_prompt_tokens

    # ------------------------------------------------------------------
    def classify(self, prompt: str) -> str:
        """Classify the task category of a prompt."""

        lowered = (prompt or "").lower()
        best_category = "code"
        best_score = 0
        for category, patterns in TASK_CATEGORY_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in lowered)
            if score > best_score:
                best_category = category
                best_score = score
        return best_category

    def count_subtasks(self, prompt: str) -> int:
        """Count the explicitly requested subtasks."""

        lines = [line for line in (prompt or "").splitlines() if line.strip()]
        bullets = [line for line in lines if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
        return max(1, len(bullets))

    def wants_verification(self, prompt: str) -> bool:
        lowered = (prompt or "").lower()
        return any(pattern in lowered for pattern in _VERIFICATION_PATTERNS)

    # ------------------------------------------------------------------
    def analyze(
        self,
        request_id: str,
        prompt: str,
        prompt_tokens: int,
        requested_output_tokens: int,
        latency_target_ms: float | None = None,
        requested_compute_budget: int | None = None,
        requested_tools: list[str] | None = None,
    ) -> AnalyzedRequest:
        """Analyse and validate a request."""

        if not prompt or not prompt.strip():
            raise RequestError("The prompt is empty.")
        if prompt_tokens > self.max_prompt_tokens:
            raise RequestError(
                f"The prompt uses {prompt_tokens} tokens which exceeds the fixed "
                f"{CONTEXT_LENGTH} token context."
            )
        if requested_output_tokens <= 0:
            raise RequestError("max_tokens must be greater than zero.")

        return AnalyzedRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            task_category=self.classify(prompt),
            requested_output_tokens=requested_output_tokens,
            wants_verification=self.wants_verification(prompt),
            latency_target_ms=latency_target_ms,
            requested_compute_budget=requested_compute_budget,
            subtask_count=self.count_subtasks(prompt),
            contains_code_fence=bool(_CODE_FENCE.search(prompt)),
            requested_tools=list(requested_tools or []),
        )

    def truncate_prompt(self, prompt: str, prompt_tokens: int, reserved_output: int) -> str:
        """Truncate a prompt so that it fits together with the completion."""

        limit = self.max_prompt_tokens - max(0, reserved_output)
        if limit <= 0:
            raise RequestError(
                "The requested completion does not fit into the fixed "
                f"{CONTEXT_LENGTH} token context."
            )
        if prompt_tokens <= limit:
            return prompt
        # Approximate the truncation with a character ratio; the exact
        # token boundary is enforced by the tokenizer afterwards.
        ratio = limit / max(1, prompt_tokens)
        keep = max(1, int(len(prompt) * ratio))
        return prompt[:keep]
