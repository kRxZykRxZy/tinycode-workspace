"""Learning rate scheduling (linear warmup + cosine decay)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["SchedulerConfig", "WarmupCosineScheduler", "build_scheduler"]


@dataclass
class SchedulerConfig:
    """Configuration of the learning rate schedule."""

    warmup_steps: int = 500
    max_steps: int = 10_000
    min_lr: float = 3e-5
    max_lr: float = 3e-4


class WarmupCosineScheduler:
    """A small, dependency free scheduler operating on ``optimizer.param_groups``."""

    def __init__(self, optimizer: Any, config: SchedulerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        self.last_lr = config.max_lr
        self._apply(self._lr_at(0))

    def _lr_at(self, step: int) -> float:
        config = self.config
        warmup = max(1, config.warmup_steps)
        if step < warmup:
            return config.min_lr + (config.max_lr - config.min_lr) * (step / warmup)
        total = max(warmup + 1, config.max_steps)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + (config.max_lr - config.min_lr) * cosine

    def _apply(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr

    def step(self) -> float:
        """Advance the schedule by one optimizer step and return the new LR."""

        self.step_count += 1
        lr = self._lr_at(self.step_count)
        self._apply(lr)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "last_lr": self.last_lr}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.step_count = int(payload.get("step_count", 0))
        self._apply(self._lr_at(self.step_count))


def build_scheduler(optimizer: Any, config: SchedulerConfig) -> WarmupCosineScheduler:
    """Create the scheduler for ``optimizer``."""

    return WarmupCosineScheduler(optimizer, config)
