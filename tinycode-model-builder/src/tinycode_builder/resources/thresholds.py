"""Resource pressure thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["ResourceLevel", "ResourceThresholds", "thresholds_from_config"]


class ResourceLevel(str, Enum):
    """How close a resource is to exhaustion."""

    NORMAL = "normal"
    CAUTION = "caution"
    REDUCED = "reduced"
    CRITICAL = "critical"
    OOM = "oom"


@dataclass
class ResourceThresholds:
    """Watermark configuration for host memory, GPU memory and disk."""

    memory_high: float = 0.75
    memory_caution: float = 0.90
    memory_reduced: float = 0.94
    memory_critical: float = 0.95
    disk_free_min_mb: int = 2048

    def classify(self, used_ratio: float) -> ResourceLevel:
        """Map a usage ratio onto a resource level."""

        if used_ratio >= self.memory_critical:
            return ResourceLevel.CRITICAL
        if used_ratio >= self.memory_reduced:
            return ResourceLevel.REDUCED
        if used_ratio >= self.memory_caution:
            return ResourceLevel.CAUTION
        if used_ratio >= self.memory_high:
            return ResourceLevel.REDUCED
        return ResourceLevel.NORMAL


def thresholds_from_config(config: dict) -> "ResourceThresholds":
    """Build thresholds from the ``resource_policy`` configuration section."""

    policy = dict(config.get("resource_policy", {}) or {})
    return ResourceThresholds(
        memory_high=float(policy.get("cpu_memory_high", 0.75)),
        memory_caution=float(policy.get("cpu_memory_caution", 0.90)),
        memory_reduced=float(policy.get("cpu_memory_critical", 0.95)) - 0.01,
        memory_critical=float(policy.get("cpu_memory_critical", 0.95)),
        disk_free_min_mb=int(policy.get("disk_free_min_mb", 2048)),
    )
