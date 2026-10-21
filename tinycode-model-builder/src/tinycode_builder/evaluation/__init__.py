"""Model evaluation."""

from __future__ import annotations

from .benchmark_runner import BenchmarkRunner, run_evaluation
from .code_tasks import DEFAULT_CODE_TASKS, CodeTask, run_code_tasks, score_completion
from .loss import LossResult, batched_loss, sequence_loss
from .perplexity import compute_perplexity, perplexity_from_loss
from .reports import EvaluationReport, write_report

__all__ = [
    "BenchmarkRunner",
    "CodeTask",
    "DEFAULT_CODE_TASKS",
    "EvaluationReport",
    "LossResult",
    "batched_loss",
    "compute_perplexity",
    "perplexity_from_loss",
    "run_code_tasks",
    "run_evaluation",
    "score_completion",
    "sequence_loss",
    "write_report",
]
