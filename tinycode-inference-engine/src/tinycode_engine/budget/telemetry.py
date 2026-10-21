"""Layer 10: feedback and calibration.

Only aggregate measurements are recorded: no request content is stored.
The records are meant for offline policy evaluation; they are never used
to train on private user requests without an explicit policy.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controller import BudgetFeatures

__all__ = ["BudgetRecord", "BudgetTelemetry", "record_to_dict"]


@dataclass
class BudgetRecord:
    """One aggregated measurement of a request."""

    request_id: str = ""
    task_category: str = "code"
    features: BudgetFeatures = field(default_factory=BudgetFeatures)
    policy: str = "standard"
    allocated_tokens: int = 0
    consumed_tokens: int = 0
    internal_steps: int = 1
    verification_runs: int = 0
    verification_passed: bool | None = None
    uncertainty_score: float = 0.0
    complexity_score: float = 0.0
    latency_ms: float = 0.0
    additional_computation_helped: bool | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_category": self.task_category,
            "features": self.features.to_dict(),
            "policy": self.policy,
            "allocated_tokens": self.allocated_tokens,
            "consumed_tokens": self.consumed_tokens,
            "internal_steps": self.internal_steps,
            "verification_runs": self.verification_runs,
            "verification_passed": self.verification_passed,
            "uncertainty_score": round(self.uncertainty_score, 6),
            "complexity_score": round(self.complexity_score, 6),
            "latency_ms": round(self.latency_ms, 3),
            "additional_computation_helped": self.additional_computation_helped,
            "timestamp": self.timestamp,
        }


def record_to_dict(record: BudgetRecord) -> dict[str, Any]:
    """Serialise a record for the offline policy dataset."""

    return record.to_dict()


class BudgetTelemetry:
    """Collects the budget measurements of the runtime."""

    def __init__(self, keep_in_memory: int = 1000, log_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self.records: list[BudgetRecord] = []
        self.keep_in_memory = keep_in_memory
        self.log_path = Path(log_path) if log_path else None
        self.counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    def record(self, record: BudgetRecord) -> None:
        """Store a record."""

        with self._lock:
            self.records.append(record)
            if len(self.records) > self.keep_in_memory:
                del self.records[: len(self.records) - self.keep_in_memory]
            key = f"policy.{record.policy}"
            self.counters[key] = self.counters.get(key, 0) + 1
            self.counters["requests"] = self.counters.get("requests", 0) + 1
        if self.log_path is not None:
            self._write(record)

    def _write(self, record: BudgetRecord) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        except OSError:  # pragma: no cover
            pass

    def snapshot(self) -> dict[str, Any]:
        """Return the aggregated counters."""

        with self._lock:
            return {
                "counters": dict(self.counters),
                "records": len(self.records),
            }

    def export(self, path: str | Path) -> Path:
        """Export the records as a policy dataset (JSON lines)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self.records)
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return target

    def category_statistics(self) -> dict[str, dict[str, float]]:
        """Return the per-category averages of the recorded measurements."""

        with self._lock:
            records = list(self.records)
        statistics: dict[str, dict[str, float]] = {}
        for record in records:
            entry = statistics.setdefault(
                record.task_category, {"count": 0, "allocated": 0.0, "consumed": 0.0, "latency": 0.0}
            )
            entry["count"] += 1
            entry["allocated"] += record.allocated_tokens
            entry["consumed"] += record.consumed_tokens
            entry["latency"] += record.latency_ms
        for entry in statistics.values():
            count = max(1.0, entry["count"])
            entry["mean_allocated"] = entry["allocated"] / count
            entry["mean_consumed"] = entry["consumed"] / count
            entry["mean_latency_ms"] = entry["latency"] / count
        return statistics
