"""Recovery actions taken when the machine runs out of resources."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

from .thresholds import ResourceLevel, ResourceThresholds

__all__ = ["RecoveryAction", "RecoveryEvent", "RecoveryManager"]


@dataclass
class RecoveryAction:
    """A single recovery action."""

    name: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "detail": self.detail}


@dataclass
class RecoveryEvent:
    """Record of a recovery episode."""

    level: str
    actions: list[RecoveryAction] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "actions": [action.to_dict() for action in self.actions],
        }


class RecoveryManager:
    """Applies escalating recovery actions as memory pressure increases.

    Recovery never changes the context length and never changes the
    selected parameter target.
    """

    def __init__(self, thresholds: ResourceThresholds | None = None) -> None:
        self.thresholds = thresholds or ResourceThresholds()
        self.events: list[RecoveryEvent] = []
        self.micro_batch_scale = 1.0
        self.loader_workers_scale = 1.0
        self.paused = False

    # ------------------------------------------------------------------
    def record(self, level: ResourceLevel, actions: list[RecoveryAction], message: str) -> RecoveryEvent:
        event = RecoveryEvent(level=level.value, actions=actions, message=message)
        self.events.append(event)
        if len(self.events) > 200:
            del self.events[:100]
        return event

    def handle_pressure(self, level: ResourceLevel, available_memory_bytes: int | None = None) -> RecoveryEvent:
        """Apply the recovery actions appropriate for ``level``."""

        actions: list[RecoveryAction] = []

        if level == ResourceLevel.NORMAL:
            event = self.record(level, actions, "Resource usage normal.")
            return event

        if level == ResourceLevel.CAUTION:
            actions.append(RecoveryAction("release_cached_memory", "Freeing cached host memory"))
            self._release_memory()

        if level in (ResourceLevel.REDUCED, ResourceLevel.CRITICAL):
            self.micro_batch_scale = max(0.25, self.micro_batch_scale * 0.5)
            actions.append(
                RecoveryAction(
                    "reduce_micro_batch",
                    f"micro batch scale -> {self.micro_batch_scale:.2f}",
                )
            )
            self.loader_workers_scale = max(0.0, self.loader_workers_scale * 0.5)
            actions.append(
                RecoveryAction(
                    "reduce_loader_workers",
                    f"loader worker scale -> {self.loader_workers_scale:.2f}",
                )
            )

        if level == ResourceLevel.CRITICAL:
            self.paused = True
            actions.append(RecoveryAction("pause_training", "Waiting for memory to recover"))

        if level == ResourceLevel.OOM:
            self.micro_batch_scale = max(0.125, self.micro_batch_scale * 0.5)
            self.paused = False
            actions.append(
                RecoveryAction(
                    "reduce_micro_batch",
                    f"micro batch scale -> {self.micro_batch_scale:.2f}",
                )
            )
            self._release_memory()

        message = f"Resource level {level.value}"
        return self.record(level, actions, message)

    def handle_oom(self, error: BaseException | None = None) -> RecoveryEvent:
        """Handle an out-of-memory error raised by the framework."""

        self.micro_batch_scale = max(0.125, self.micro_batch_scale * 0.5)
        actions = [
            RecoveryAction(
                "reduce_micro_batch", f"micro batch scale -> {self.micro_batch_scale:.2f}"
            ),
            RecoveryAction("increase_gradient_accumulation", "Preserving the effective batch size"),
        ]
        self._release_memory()
        return self.record(ResourceLevel.OOM, actions, f"Out of memory: {error}")

    def resume(self) -> None:
        """Clear the paused state after a critical event."""

        self.paused = False

    def _release_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                torch.cuda.empty_cache()
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_scale": self.micro_batch_scale,
            "loader_workers_scale": self.loader_workers_scale,
            "paused": self.paused,
            "events": [event.to_dict() for event in self.events[-20:]],
        }
