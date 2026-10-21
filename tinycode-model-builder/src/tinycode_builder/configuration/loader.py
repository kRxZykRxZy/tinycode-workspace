"""Loading of YAML configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .defaults import default_config, merge_defaults
from .immutable_constraints import is_context_length_override_attempt

__all__ = [
    "ConfigError",
    "load_yaml_file",
    "load_config",
    "load_config_directory",
    "config_root",
]


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def config_root() -> Path:
    """Return the directory holding the shipped YAML configuration files."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config"
        if candidate.is_dir():
            return candidate
    raise ConfigError("Unable to locate the builder config directory.")


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dictionary."""

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - depends on user input
        raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


def load_config_directory(directory: str | Path | None = None) -> dict[str, Any]:
    """Load every YAML file in ``directory`` merged over the built-in defaults.

    Files are merged in sorted order so that the result is deterministic.
    The context length is never configurable: any file that tries to set a
    context-length-like key is rejected.
    """

    directory = Path(directory) if directory is not None else config_root()
    if not directory.is_dir():
        raise ConfigError(f"Configuration directory not found: {directory}")

    merged = default_config()
    for path in sorted(directory.glob("*.yaml")):
        payload = load_yaml_file(path)
        offending = is_context_length_override_attempt(payload)
        if offending is not None:
            raise ConfigError(
                f"{path.name} attempts to override the immutable context length via "
                f"'{offending}'. The context length is fixed at 1024 tokens."
            )
        merged = merge_defaults(merged, payload)
    return merged


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load configuration from an optional file/directory plus explicit overrides.

    Environment variables are intentionally *not* consulted for the context
    length; see :mod:`tinycode_builder.configuration.immutable_constraints`.
    """

    merged = default_config()
    if path is not None:
        candidate = Path(path)
        if candidate.is_dir():
            merged = merge_defaults(merged, load_config_directory(candidate))
        else:
            merged = merge_defaults(merged, load_yaml_file(candidate))
    if overrides:
        offending = is_context_length_override_attempt(overrides)
        if offending is not None:
            raise ConfigError(
                f"Override '{offending}' attempts to change the immutable context length."
            )
        merged = merge_defaults(merged, overrides)
    return merged


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag (used only for non-constraint settings)."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
