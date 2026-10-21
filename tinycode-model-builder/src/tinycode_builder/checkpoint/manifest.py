"""Checkpoint manifest describing the contents of a checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CheckpointManifest", "MANIFEST_VERSION"]

MANIFEST_VERSION = "1.0.0"


@dataclass
class CheckpointManifest:
    """Metadata stored next to every checkpoint."""

    version: str = MANIFEST_VERSION
    step: int = 0
    committed_training_tokens: int = 0
    validation_tokens: int = 0
    parameter_target: int = 0
    actual_parameters: int = 0
    context_length: int = 1024
    architecture: dict[str, Any] = field(default_factory=dict)
    tokenizer_fingerprint: str = ""
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "validation_tokens": self.validation_tokens,
            "parameter_target": self.parameter_target,
            "actual_parameters": self.actual_parameters,
            "context_length": self.context_length,
            "architecture": dict(self.architecture),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "files": dict(self.files),
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointManifest":
        return cls(
            version=str(payload.get("version", MANIFEST_VERSION)),
            step=int(payload.get("step", 0)),
            committed_training_tokens=int(payload.get("committed_training_tokens", 0)),
            validation_tokens=int(payload.get("validation_tokens", 0)),
            parameter_target=int(payload.get("parameter_target", 0)),
            actual_parameters=int(payload.get("actual_parameters", 0)),
            context_length=int(payload.get("context_length", 1024)),
            architecture=dict(payload.get("architecture", {}) or {}),
            tokenizer_fingerprint=str(payload.get("tokenizer_fingerprint", "")),
            files=dict(payload.get("files", {}) or {}),
            created_at=float(payload.get("created_at", 0.0)),
            metrics=dict(payload.get("metrics", {}) or {}),
            extra=dict(payload.get("extra", {}) or {}),
        )
