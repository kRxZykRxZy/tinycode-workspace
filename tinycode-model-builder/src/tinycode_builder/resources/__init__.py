"""Resource management for the training loop."""

from __future__ import annotations

from .adaptive_batch import AdaptiveBatchPolicy, choose_micro_batch, estimate_bytes_per_sequence
from .memory_estimator import MemoryEstimate, estimate_training_memory
from .monitor import ResourceMonitor, ResourceSample
from .recovery import RecoveryAction, RecoveryEvent, RecoveryManager
from .thresholds import ResourceLevel, ResourceThresholds, thresholds_from_config
from .watchdog import Watchdog

__all__ = [
    "AdaptiveBatchPolicy",
    "MemoryEstimate",
    "RecoveryAction",
    "RecoveryEvent",
    "RecoveryManager",
    "ResourceLevel",
    "ResourceMonitor",
    "ResourceSample",
    "ResourceThresholds",
    "Watchdog",
    "choose_micro_batch",
    "estimate_bytes_per_sequence",
    "estimate_training_memory",
    "thresholds_from_config",
]
