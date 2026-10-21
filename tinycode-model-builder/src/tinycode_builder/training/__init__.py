"""Training engine."""

from __future__ import annotations

from .gradient_accumulation import AccumulationPlan, scale_loss, unscale_and_clip
from .metrics import MetricSnapshot, RunningMetrics
from .optimizer import build_optimizer, parameter_groups
from .precision import PrecisionPlan, autocast_context, build_scaler, resolve_precision
from .scheduler import SchedulerConfig, WarmupCosineScheduler, build_scheduler
from .stopping import StopReason, StoppingCriteria, TrainingStopper
from .train_step import TrainStepResult, build_batch_tensors, run_train_step
from .trainer import Trainer, TrainingEvent, TrainingResult
from .validation import ValidationResult, evaluate_loss

__all__ = [
    "AccumulationPlan",
    "MetricSnapshot",
    "PrecisionPlan",
    "RunningMetrics",
    "SchedulerConfig",
    "StopReason",
    "StoppingCriteria",
    "TrainStepResult",
    "Trainer",
    "TrainingEvent",
    "TrainingResult",
    "TrainingStopper",
    "ValidationResult",
    "WarmupCosineScheduler",
    "autocast_context",
    "build_batch_tensors",
    "build_optimizer",
    "build_scaler",
    "build_scheduler",
    "evaluate_loss",
    "parameter_groups",
    "resolve_precision",
    "run_train_step",
    "scale_loss",
    "unscale_and_clip",
]
