"""Runtime metrics.

The metrics that cannot be measured are reported as unavailable rather
than as a zero, so that a zero is never confused with "not measured".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Metric", "MetricsCollector", "UNAVAILABLE"]

UNAVAILABLE = "unavailable"


@dataclass
class Metric:
    """A single metric value."""

    name: str
    value: float | None = None
    help_text: str = ""
    metric_type: str = "gauge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "help": self.help_text,
            "type": self.metric_type,
            "available": self.value is not None,
        }


class MetricsCollector:
    """Collects and renders the metrics of the engine."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._extra: dict[str, float] = {}
        self.started_at = time.time()

    # ------------------------------------------------------------------
    def increment(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._extra[name] = self._extra.get(name, 0.0) + amount

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._extra[name] = value

    # ------------------------------------------------------------------
    def collect(self) -> list[Metric]:
        """Collect the metrics of the engine."""

        metrics: list[Metric] = []
        engine = self.engine
        raw = dict(getattr(engine, "metrics", {}) or {})

        def _add(name: str, value: Any, help_text: str, metric_type: str = "gauge") -> None:
            metrics.append(
                Metric(
                    name=name,
                    value=None if value is None else float(value),
                    help_text=help_text,
                    metric_type=metric_type,
                )
            )

        _add("tinycode_requests_received", raw.get("requests_received"), "Requests received.", "counter")
        _add("tinycode_requests_completed", raw.get("requests_completed"), "Requests completed.", "counter")
        _add("tinycode_requests_rejected", raw.get("requests_rejected"), "Requests rejected.", "counter")
        _add("tinycode_input_tokens", raw.get("input_tokens"), "Input tokens.", "counter")
        _add("tinycode_output_tokens", raw.get("output_tokens"), "Output tokens.", "counter")

        completed = float(raw.get("requests_completed", 0) or 0)
        elapsed_total = float(raw.get("response_ms_total", 0.0) or 0.0)
        _add(
            "tinycode_response_ms_average",
            (elapsed_total / completed) if completed else None,
            "Average end to end latency in milliseconds.",
        )
        first_token_total = float(raw.get("first_token_ms_total", 0.0) or 0.0)
        _add(
            "tinycode_time_to_first_token_ms_average",
            (first_token_total / completed) if completed else None,
            "Average time to the first token in milliseconds.",
        )

        queue_depth: int | None = None
        if engine is not None and getattr(engine, "scheduler", None) is not None:
            queue_depth = int(engine.scheduler.queue.stats()["depth"])
        _add("tinycode_queue_depth", queue_depth, "Number of queued requests.")

        active: int | None = None
        if engine is not None and getattr(engine, "scheduler", None) is not None:
            active = len(engine.scheduler.active)
        _add("tinycode_active_requests", active, "Number of active requests.")

        backend_name: str | None = None
        if engine is not None and getattr(engine, "backend_selection", None) is not None:
            backend_name = engine.backend_selection.name
        _add("tinycode_backend_selected", None, "The selected backend is reported as a label.")

        memory_bytes: int | None = None
        if engine is not None and getattr(engine, "memory", None) is not None:
            memory_bytes = int(engine.memory.snapshot().allocated_bytes)
        _add("tinycode_memory_allocated_bytes", memory_bytes, "Allocated runtime memory.")

        load_seconds: float | None = None
        if engine is not None:
            load_seconds = float(getattr(engine, "load_seconds", 0.0) or 0.0)
        _add("tinycode_model_load_seconds", load_seconds, "Time spent loading the model.")

        with self._lock:
            for name, value in self._extra.items():
                _add(f"tinycode_{name}", value, f"Custom counter {name}.", "counter")

        return metrics

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Render the metrics in the Prometheus text exposition format."""

        lines: list[str] = []
        backend = None
        if self.engine is not None and getattr(self.engine, "backend_selection", None) is not None:
            backend = self.engine.backend_selection.name
        for metric in self.collect():
            if metric.name == "tinycode_backend_selected":
                lines.append(f"# HELP {metric.name} {metric.help_text}")
                lines.append(f"# TYPE {metric.name} gauge")
                lines.append(f'{metric.name}{{backend="{backend}"}} 1')
                continue
            if metric.value is None:
                continue
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type}")
            lines.append(f"{metric.name} {metric.value}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        """Return the metrics as a JSON friendly dictionary."""

        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "metrics": [metric.to_dict() for metric in self.collect()],
        }
