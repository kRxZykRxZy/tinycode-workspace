"""Resuming a training run from a checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.errors import CheckpointError

__all__ = ["ResumeState", "extract_resume_state", "restore_training_state"]


@dataclass
class ResumeState:
    """Everything needed to continue a training run."""

    step: int = 0
    committed_training_tokens: int = 0
    validation_tokens: int = 0
    model_state: Any = None
    optimizer_state: Any = None
    scheduler_state: Any = None
    scaler_state: Any = None
    rng_state: Any = None
    dataset_position: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    recovery_state: Any = None
    architecture: dict[str, Any] = field(default_factory=dict)
    parameter_target: int = 0
    actual_parameters: int = 0
    tokenizer_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "validation_tokens": self.validation_tokens,
            "metrics": dict(self.metrics),
            "architecture": dict(self.architecture),
            "parameter_target": self.parameter_target,
            "actual_parameters": self.actual_parameters,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }


def extract_resume_state(payload: dict[str, Any]) -> ResumeState:
    """Pull the resume relevant fields out of a checkpoint payload."""

    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dictionary.")
    return ResumeState(
        step=int(payload.get("step", 0)),
        committed_training_tokens=int(payload.get("committed_training_tokens", 0)),
        validation_tokens=int(payload.get("validation_tokens", 0)),
        model_state=payload.get("model_state"),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        scaler_state=payload.get("scaler_state"),
        rng_state=payload.get("rng_state"),
        dataset_position=payload.get("dataset_position"),
        metrics=dict(payload.get("metrics", {}) or {}),
        recovery_state=payload.get("recovery_state"),
        architecture=dict(payload.get("architecture", {}) or {}),
        parameter_target=int(payload.get("parameter_target", 0)),
        actual_parameters=int(payload.get("actual_parameters", 0)),
        tokenizer_fingerprint=str(payload.get("tokenizer_fingerprint", "")),
    )


def restore_training_state(
    trainer: Any,
    state: ResumeState,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a trainer from a :class:`ResumeState`.

    Returns:
        A summary describing what was restored.
    """

    restored: dict[str, Any] = {"step": state.step}

    trainer.step = int(state.step)
    trainer.committed_training_tokens = int(state.committed_training_tokens)
    trainer.validation_tokens = int(state.validation_tokens)

    if state.model_state is not None:
        trainer.model.load_state_dict(state.model_state)
        restored["model"] = True
    if state.optimizer_state is not None and trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(state.optimizer_state)
        restored["optimizer"] = True
    if state.scheduler_state is not None and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(state.scheduler_state)
        restored["scheduler"] = True
    if state.scaler_state is not None and trainer.scaler is not None:
        try:
            trainer.scaler.load_state_dict(state.scaler_state)
            restored["scaler"] = True
        except Exception:  # pragma: no cover - scaler mismatch is not fatal
            restored["scaler"] = False
    if restore_rng and state.rng_state is not None:
        try:
            trainer.restore_rng_state(state.rng_state)
            restored["rng"] = True
        except Exception:  # pragma: no cover
            restored["rng"] = False
    if state.dataset_position is not None:
        trainer.dataset_position = state.dataset_position
        restored["dataset_position"] = True
    if state.metrics:
        trainer.metrics.restore(state.metrics)
        restored["metrics"] = True
    if state.recovery_state is not None and hasattr(trainer, "recovery"):
        trainer.recovery.restore(state.recovery_state)
        restored["recovery"] = True
    return restored
