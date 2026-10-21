"""Startup of the inference server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime.engine import EngineConfig, InferenceEngine
from ..utils.errors import ArtifactError

__all__ = ["StartupReport", "start_engine"]


class StartupReport:
    """What the startup sequence did."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.backend: str = "none"
        self.backend_reason: str = ""
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def ok(self) -> bool:
        return bool(self.models) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "models": list(self.models),
            "backend": self.backend,
            "backend_reason": self.backend_reason,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def start_engine(
    config: EngineConfig,
    engine: InferenceEngine | None = None,
) -> StartupReport:
    """Load the configured artifacts and prepare the engine."""

    report = StartupReport()
    engine = engine or InferenceEngine(config)

    paths = list(config.model_paths)
    if not paths:
        report.errors.append(
            "No model path was configured. Set 'model_paths' in the configuration or "
            "pass --model."
        )
        return report

    for index, path in enumerate(paths):
        directory = Path(path)
        if not directory.is_dir():
            report.errors.append(f"The model directory does not exist: {directory}")
            continue
        alias = config.default_model_alias if index == 0 else directory.name
        try:
            artifact = engine.load_artifact(str(directory), alias=alias)
        except ArtifactError as exc:
            report.errors.append(f"{directory}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive
            report.errors.append(f"{directory}: {exc}")
            continue
        report.models.append(alias)
        report.warnings.extend(artifact.warnings)

    if engine.backend_selection is not None:
        report.backend = engine.backend_selection.name
        report.backend_reason = engine.backend_selection.reason
    return report
