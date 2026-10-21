"""The engine service: configuration, engine and application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..api.app import create_app
from ..runtime.engine import EngineConfig, InferenceEngine
from ..server.startup import StartupReport, start_engine

__all__ = ["ServiceConfig", "EngineService", "load_config", "config_directory"]


def config_directory() -> Path:
    """Return the directory that holds the shipped configuration files."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Unable to locate the configuration directory.")


def load_config(directory: str | Path | None = None) -> EngineConfig:
    """Load the engine configuration from the YAML files.

    The context length is not configurable: it is fixed at 1024 tokens and
    any attempt to change it is rejected.
    """

    base = Path(directory) if directory else config_directory()
    payload: dict[str, Any] = {}
    if base.is_dir():
        for path in sorted(base.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(payload.get(key), dict):
                    payload[key].update(value)
                else:
                    payload[key] = value

    forbidden = {"context_length", "context_size", "max_context", "n_positions"}
    for key in list(payload):
        if key.lower() in forbidden:
            raise ValueError(
                f"The configuration file {base} tries to override the immutable "
                "context length."
            )

    env_model = os.environ.get("TINYCODE_MODEL")
    if env_model:
        paths = list(payload.get("model_paths", []) or [])
        paths.append(env_model)
        payload["model_paths"] = paths

    return EngineConfig.from_dict(payload)


@dataclass
class ServiceConfig:
    """How the service is hosted."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    reload: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "log_level": self.log_level,
            "reload": self.reload,
        }


@dataclass
class EngineService:
    """Owns the engine and the FastAPI application."""

    config: EngineConfig = field(default_factory=EngineConfig)
    service_config: ServiceConfig = field(default_factory=ServiceConfig)
    engine: InferenceEngine | None = None
    app: Any = None
    startup_report: StartupReport | None = None

    def build(self) -> "EngineService":
        """Create the engine and the application."""

        self.engine = InferenceEngine(self.config)
        self.app = create_app(self.engine, self.config)
        return self

    def start_models(self) -> StartupReport:
        """Load the configured artifacts."""

        if self.engine is None:
            self.build()
        assert self.engine is not None
        self.startup_report = start_engine(self.config, self.engine)
        return self.startup_report

    def run(self) -> None:
        """Run the HTTP server (blocking)."""

        import uvicorn

        if self.app is None:
            self.build()
        uvicorn.run(
            self.app,
            host=self.service_config.host,
            port=self.service_config.port,
            log_level=self.service_config.log_level,
        )

    def shutdown(self) -> None:
        """Release the resources."""

        from ..server.shutdown import shutdown_engine

        if self.engine is not None:
            shutdown_engine(self.engine)
