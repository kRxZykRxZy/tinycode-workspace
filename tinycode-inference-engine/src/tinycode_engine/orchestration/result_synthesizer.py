"""Assembly of the final answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["SynthesisResult", "ResultSynthesizer"]


@dataclass
class SynthesisResult:
    """The synthesised answer."""

    text: str = ""
    used_multiple_passes: bool = False
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_multiple_passes": self.used_multiple_passes,
            "notes": list(self.notes),
        }


class ResultSynthesizer:
    """Combines the results of several passes into the final answer.

    The synthesizer is conservative: when several passes produced
    different answers the *last* pass is kept and the difference is
    reported, because no verified ranking is available.
    """

    def synthesize(self, candidates: list[str]) -> SynthesisResult:
        """Synthesize the final answer from the candidate answers."""

        result = SynthesisResult()
        if not candidates:
            result.text = ""
            result.notes.append("no candidate was produced")
            return result

        result.text = candidates[-1]
        result.used_multiple_passes = len(candidates) > 1
        if result.used_multiple_passes:
            unique = {candidate.strip() for candidate in candidates}
            if len(unique) > 1:
                result.notes.append(
                    "several passes produced different answers; the final pass was kept "
                    "because no verified ranking is available"
                )
        return result
