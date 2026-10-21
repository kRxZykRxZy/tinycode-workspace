"""Model construction and parameter accounting."""

from __future__ import annotations

from .architecture import ARCHITECTURE_ID, TransformerConfig
from .causal_model import ModelOutput, TinyCodeCausalLM, build_model
from .factory import (
    ArchitectureSolution,
    NoExactArchitectureError,
    solve_architecture,
)
from .parameter_accounting import (
    ParameterBreakdown,
    ParameterCountMismatch,
    analytic_parameter_count,
    count_trainable_parameters,
    parameter_breakdown,
    verify_parameter_count,
)

__all__ = [
    "ARCHITECTURE_ID",
    "ArchitectureSolution",
    "ModelOutput",
    "NoExactArchitectureError",
    "ParameterBreakdown",
    "ParameterCountMismatch",
    "TinyCodeCausalLM",
    "TransformerConfig",
    "analytic_parameter_count",
    "build_model",
    "count_trainable_parameters",
    "parameter_breakdown",
    "solve_architecture",
    "verify_parameter_count",
]
