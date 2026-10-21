"""Request priorities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

__all__ = ["Priority", "PriorityClass", "compare_priorities"]


class Priority(IntEnum):
    """The priority classes of the scheduler."""

    INTERACTIVE = 10
    NORMAL = 20
    BACKGROUND = 30

    @classmethod
    def from_name(cls, name: str | None) -> "Priority":
        if not name:
            return cls.NORMAL
        try:
            return cls[str(name).upper()]
        except KeyError:
            return cls.NORMAL


@dataclass
class PriorityClass:
    """A priority with an optional deadline."""

    priority: Priority = Priority.NORMAL
    deadline: float | None = None
    enqueued_at: float = 0.0

    @property
    def value(self) -> int:
        return int(self.priority)

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority.name.lower(),
            "deadline": self.deadline,
            "enqueued_at": self.enqueued_at,
        }


def compare_priorities(left: PriorityClass, right: PriorityClass) -> int:
    """Compare two priorities: lower value means higher priority."""

    if left.value != right.value:
        return -1 if left.value < right.value else 1
    if left.deadline is not None and right.deadline is not None:
        if left.deadline != right.deadline:
            return -1 if left.deadline < right.deadline else 1
    if left.enqueued_at != right.enqueued_at:
        return -1 if left.enqueued_at < right.enqueued_at else 1
    return 0
