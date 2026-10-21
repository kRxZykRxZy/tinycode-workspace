"""The state of an interactive builder session."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..configuration.defaults import default_config, merge_defaults
from ..configuration.validation import validate_config
from ..utils.paths import BuilderPaths

__all__ = ["BuilderSession", "SessionState"]


@dataclass
class SessionState:
    """Everything the session remembers between menu steps."""

    parameter_target: int = 10_000_000
    vocabulary_size: int = 98_304
    training_token_target: int = 200_000_000
    architecture: dict[str, Any] = field(default_factory=dict)
    actual_parameters: int = 0
    tokenizer_dir: str | None = None
    tokenizer_fingerprint: str | None = None
    dataset_dir: str | None = None
    train_shards: list[str] = field(default_factory=list)
    validation_shards: list[str] = field(default_factory=list)
    dataset_tokens: int = 0
    checkpoint_dir: str | None = None
    artifact_dir: str | None = None
    controller_dir: str | None = None
    last_training: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_target": self.parameter_target,
            "vocabulary_size": self.vocabulary_size,
            "training_token_target": self.training_token_target,
            "architecture": dict(self.architecture),
            "actual_parameters": self.actual_parameters,
            "tokenizer_dir": self.tokenizer_dir,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "dataset_dir": self.dataset_dir,
            "train_shards": list(self.train_shards),
            "validation_shards": list(self.validation_shards),
            "dataset_tokens": self.dataset_tokens,
            "checkpoint_dir": self.checkpoint_dir,
            "artifact_dir": self.artifact_dir,
            "controller_dir": self.controller_dir,
            "last_training": dict(self.last_training),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionState":
        known = {key: payload.get(key) for key in cls().to_dict() if key in payload}
        state = cls()
        for key, value in known.items():
            setattr(state, key, value)
        return state


class BuilderSession:
    """Holds the configuration, paths and discovered state of the session."""

    def __init__(self, root: str | Path | None = None, config: dict[str, Any] | None = None) -> None:
        self.paths = BuilderPaths.from_root(root).ensure()
        self.config = merge_defaults(default_config(), config or {})
        validate_config(self.config)
        self.state = SessionState(
            parameter_target=int(self.config["model"]["parameter_target"]),
            vocabulary_size=int(self.config["model"]["vocab_size"]),
            training_token_target=int(self.config["training"]["target_tokens"]),
        )
        self.hardware: Any = None
        self.state_file = self.paths.runs_dir / "session.json"

    # ------------------------------------------------------------------
    def update_config(self, section: str, values: dict[str, Any]) -> dict[str, Any]:
        """Update a configuration section after validating the result."""

        merged = merge_defaults(self.config, {section: values})
        validate_config(merged)
        self.config = merged
        self.save()
        return self.config

    def set_parameter_target(self, target: int) -> None:
        """Record the user selected parameter target."""

        self.state.parameter_target = int(target)
        self.update_config("model", {"parameter_target": int(target)})

    def set_training_target(self, tokens: int) -> None:
        """Record the user selected training token target."""

        self.state.training_token_target = int(tokens)
        self.update_config("training", {"target_tokens": int(tokens)})

    def save(self) -> Path:
        """Persist the session state."""

        self.state.updated_at = time.time()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self.state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.state_file

    def load(self) -> "BuilderSession":
        """Restore a previously saved session state."""

        if self.state_file.is_file():
            try:
                payload = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.state = SessionState.from_dict(payload)
            except (OSError, json.JSONDecodeError):
                pass
        return self

    def summary(self) -> str:
        """Return a human readable summary of the session."""

        lines = [
            "Session summary",
            "--------------",
            f"Parameter target   : {self.state.parameter_target:,}",
            f"Actual parameters  : {self.state.actual_parameters:,}",
            f"Vocabulary         : {self.state.vocabulary_size:,}",
            f"Training tokens    : {self.state.training_token_target:,}",
            f"Tokenizer          : {self.state.tokenizer_dir or 'not trained yet'}",
            f"Dataset            : {self.state.dataset_dir or 'not prepared yet'}",
            f"Train shards       : {len(self.state.train_shards)}",
            f"Validation shards  : {len(self.state.validation_shards)}",
            f"Artifact           : {self.state.artifact_dir or 'not exported yet'}",
        ]
        return "\n".join(lines)
