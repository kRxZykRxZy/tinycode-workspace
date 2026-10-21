"""Dashboard state shared between the training loop and the HTTP layer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DashboardState", "TrainingSnapshot"]


@dataclass
class TrainingSnapshot:
    """The latest known state of the training run."""

    running: bool = False
    step: int = 0
    committed_training_tokens: int = 0
    target_training_tokens: int = 0
    validation_tokens: int = 0
    loss: float = 0.0
    validation_loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    eta_seconds: float | None = None
    model_name: str = "tinycode"
    parameter_count: int = 0
    parameter_target: int = 0
    gpu_utilisation: float | None = None
    gpu_memory_used: int | None = None
    gpu_memory_total: int | None = None
    cpu_utilisation: float | None = None
    system_ram_used_ratio: float | None = None
    disk_free_bytes: int | None = None
    dataset_documents: int = 0
    dataset_tokens: int = 0
    checkpoint_path: str | None = None
    checkpoint_step: int = 0
    drive_sync_ok: bool | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "target_training_tokens": self.target_training_tokens,
            "validation_tokens": self.validation_tokens,
            "loss": self.loss,
            "validation_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
            "tokens_per_second": self.tokens_per_second,
            "eta_seconds": self.eta_seconds,
            "model_name": self.model_name,
            "parameter_count": self.parameter_count,
            "parameter_target": self.parameter_target,
            "gpu_utilisation": self.gpu_utilisation,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "cpu_utilisation": self.cpu_utilisation,
            "system_ram_used_ratio": self.system_ram_used_ratio,
            "disk_free_bytes": self.disk_free_bytes,
            "dataset_documents": self.dataset_documents,
            "dataset_tokens": self.dataset_tokens,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "drive_sync_ok": self.drive_sync_ok,
            "updated_at": self.updated_at,
        }


class DashboardState:
    """Thread safe container for the dashboard payload."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.snapshot = TrainingSnapshot()
        self.logs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.pause_requested = False
        self.resume_requested = False
        self.stop_requested = False
        self.checkpoint_requested = False

    # ------------------------------------------------------------------
    def update(self, **fields: Any) -> TrainingSnapshot:
        with self._lock:
            for key, value in fields.items():
                if hasattr(self.snapshot, key):
                    setattr(self.snapshot, key, value)
            self.snapshot.updated_at = time.time()
            return self.snapshot

    def get(self) -> TrainingSnapshot:
        with self._lock:
            return self.snapshot

    def add_log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self.logs.append({"timestamp": time.time(), "level": level, "message": message})
            if len(self.logs) > 1000:
                del self.logs[:500]

    def add_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            if len(self.events) > 500:
                del self.events[:250]

    def recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.logs[-count:])

    # ------------------------------------------------------------------
    def request_pause(self) -> None:
        with self._lock:
            self.pause_requested = True
            self.resume_requested = False

    def request_resume(self) -> None:
        with self._lock:
            self.resume_requested = True
            self.pause_requested = False

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def request_checkpoint(self) -> None:
        with self._lock:
            self.checkpoint_requested = True

    def consume_flags(self) -> dict[str, bool]:
        """Return and clear the pending control flags."""

        with self._lock:
            flags = {
                "pause": self.pause_requested,
                "resume": self.resume_requested,
                "stop": self.stop_requested,
                "checkpoint": self.checkpoint_requested,
            }
            self.pause_requested = False
            self.resume_requested = False
            self.stop_requested = False
            self.checkpoint_requested = False
            return flags
