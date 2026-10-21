"""Layer 6: confidence and uncertainty estimation.

The signals computed here are *heuristics*.  They are named accordingly
and are never presented as a measure of factual correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = ["UncertaintySignals", "ConfidenceEstimator", "entropy_of", "token_agreement"]


def entropy_of(logits: Any) -> float:
    """Return the Shannon entropy (in nats) of a distribution."""

    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        return 0.0
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    total = float(np.sum(probabilities))
    if total <= 0:
        return 0.0
    probabilities = probabilities / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy


def token_agreement(candidates: Sequence[Sequence[int]]) -> float:
    """Return the agreement between several independent candidate attempts."""

    if len(candidates) < 2:
        return 1.0
    reference = list(candidates[0])
    agreements: list[float] = []
    for candidate in candidates[1:]:
        length = min(len(reference), len(candidate))
        if length == 0:
            agreements.append(0.0)
            continue
        matches = sum(1 for index in range(length) if reference[index] == candidate[index])
        agreements.append(matches / float(length))
    return float(sum(agreements) / len(agreements))


@dataclass
class UncertaintySignals:
    """The measured uncertainty signals of a candidate answer."""

    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    low_probability_ratio: float = 0.0
    agreement: float | None = None
    verification_failed: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """A combined, explicitly heuristic uncertainty score in ``[0, 1]``."""

        entropy_component = min(1.0, self.mean_entropy / 4.0) * 0.5
        low_probability_component = min(1.0, self.low_probability_ratio) * 0.3
        agreement_component = 0.0
        if self.agreement is not None:
            agreement_component = (1.0 - self.agreement) * 0.2
        verification_component = 0.2 if self.verification_failed else 0.0
        return float(
            max(0.0, min(1.0, entropy_component + low_probability_component + agreement_component + verification_component))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_entropy": round(self.mean_entropy, 6),
            "max_entropy": round(self.max_entropy, 6),
            "low_probability_ratio": round(self.low_probability_ratio, 6),
            "agreement": None if self.agreement is None else round(self.agreement, 6),
            "verification_failed": self.verification_failed,
            "score": round(self.score, 6),
            "notes": list(self.notes),
            "interpretation": (
                "heuristic signals; they are not calibrated probabilities of correctness"
            ),
        }


class ConfidenceEstimator:
    """Computes the uncertainty signals of a generation."""

    def __init__(self, low_probability_threshold: float = 0.1) -> None:
        self.low_probability_threshold = low_probability_threshold

    def from_logits(self, logits_sequence: Sequence[Any]) -> UncertaintySignals:
        """Compute the signals from the logits of the generated tokens."""

        if not logits_sequence:
            return UncertaintySignals(notes=["no logits were recorded"])

        entropies: list[float] = []
        low_probability = 0
        for logits in logits_sequence:
            entropies.append(entropy_of(logits))
            values = np.asarray(logits, dtype=np.float64)
            shifted = values - np.max(values)
            probabilities = np.exp(shifted)
            total = float(np.sum(probabilities))
            if total > 0:
                probabilities = probabilities / total
                if float(np.max(probabilities)) < (1.0 - self.low_probability_threshold):
                    low_probability += 1

        mean_entropy = sum(entropies) / len(entropies)
        return UncertaintySignals(
            mean_entropy=mean_entropy,
            max_entropy=max(entropies),
            low_probability_ratio=low_probability / float(len(entropies)),
            notes=[f"measured over {len(entropies)} generated tokens"],
        )

    def combine(
        self,
        signals: UncertaintySignals,
        agreement: float | None = None,
        verification_failed: bool | None = None,
    ) -> UncertaintySignals:
        """Add the agreement and verification signals."""

        signals.agreement = agreement
        signals.verification_failed = verification_failed
        return signals
