"""Training metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RunningMetrics", "MetricSnapshot"]


@dataclass
class MetricSnapshot:
    """A point-in-time view of the training metrics."""

    step: int
    loss: float
    tokens: int
    tokens_per_second: float
    learning_rate: float
    elapsed_seconds: float
    gradient_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "tokens": self.tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "learning_rate": self.learning_rate,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "gradient_norm": self.gradient_norm,
        }


@dataclass
class RunningMetrics:
    """Accumulates training statistics."""

    step: int = 0
    tokens: int = 0
    micro_steps: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    started: float = field(default_factory=time.perf_counter)
    last_snapshot: MetricSnapshot | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def add_loss(self, loss: float) -> None:
        self.loss_sum += float(loss)
        self.loss_count += 1

    def add_tokens(self, tokens: int) -> None:
        self.tokens += int(tokens)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.loss_count if self.loss_count else 0.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed
        return self.tokens / elapsed if elapsed > 0 else 0.0

    def snapshot(self, learning_rate: float, gradient_norm: float | None = None) -> MetricSnapshot:
        """Create and record a snapshot of the current metrics."""

        snapshot = MetricSnapshot(
            step=self.step,
            loss=self.mean_loss,
            tokens=self.tokens,
            tokens_per_second=self.tokens_per_second,
            learning_rate=learning_rate,
            elapsed_seconds=self.elapsed,
            gradient_norm=gradient_norm,
        )
        self.last_snapshot = snapshot
        self.history.append(snapshot.to_dict())
        if len(self.history) > 5000:
            del self.history[:1000]
        return snapshot

    def reset_window(self) -> None:
        """Reset the loss window (used for periodic reporting)."""

        self.loss_sum = 0.0
        self.loss_count = 0

    def restore(self, payload: dict[str, Any]) -> None:
        """Restore the metrics from a snapshot produced by :meth:`to_dict`."""

        self.step = int(payload.get("step", self.step))
        self.tokens = int(payload.get("tokens", self.tokens))
        self.micro_steps = int(payload.get("micro_steps", self.micro_steps))
        mean_loss = float(payload.get("mean_loss", 0.0))
        self.loss_sum = mean_loss
        self.loss_count = 1 if mean_loss else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens": self.tokens,
            "micro_steps": self.micro_steps,
            "mean_loss": self.mean_loss,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "elapsed_seconds": round(self.elapsed, 3),
        }
