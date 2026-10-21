"""Metrics exposed by the dashboard API."""

from __future__ import annotations

from typing import Any

from ..telemetry.counters import Counters
from .state import DashboardState

__all__ = ["build_metrics_payload", "build_prometheus_payload"]


def build_metrics_payload(
    state: DashboardState,
    counters: Counters | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the JSON payload served by ``/api/metrics``."""

    snapshot = state.get().to_dict()
    payload: dict[str, Any] = {
        "training": snapshot,
        "logs": state.recent_logs(50),
        "generated_at": snapshot.get("updated_at"),
    }
    if counters is not None:
        payload["counters"] = counters.snapshot()
    if extra:
        payload.update(extra)
    return payload


def build_prometheus_payload(
    state: DashboardState, counters: Counters | None = None
) -> str:
    """Return the metrics in the Prometheus text exposition format."""

    snapshot = state.get()
    lines: list[str] = []

    def emit(name: str, value: Any, help_text: str, metric_type: str = "gauge") -> None:
        if value is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        lines.append(f"{name} {value}")

    emit("tinycode_training_step", snapshot.step, "Current optimizer step.")
    emit(
        "tinycode_training_tokens_total",
        snapshot.committed_training_tokens,
        "Committed training tokens.",
        "counter",
    )
    emit(
        "tinycode_training_target_tokens",
        snapshot.target_training_tokens,
        "Configured training token target.",
    )
    emit("tinycode_training_loss", snapshot.loss, "Training loss.")
    emit("tinycode_validation_loss", snapshot.validation_loss, "Validation loss.")
    emit(
        "tinycode_tokens_per_second",
        snapshot.tokens_per_second,
        "Training throughput in tokens per second.",
    )
    emit("tinycode_parameters", snapshot.parameter_count, "Trainable parameter count.")
    emit("tinycode_gpu_memory_used_bytes", snapshot.gpu_memory_used, "GPU memory used.")
    emit("tinycode_disk_free_bytes", snapshot.disk_free_bytes, "Free disk space.")

    if counters is not None:
        for name, value in counters.snapshot().get("counters", {}).items():
            emit(f"tinycode_{name}_total", value, f"Counter {name}.", "counter")
        for name, value in counters.snapshot().get("gauges", {}).items():
            emit(f"tinycode_{name}", value, f"Gauge {name}.")
    return "\n".join(lines) + "\n"
