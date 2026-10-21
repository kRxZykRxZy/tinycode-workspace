"""Adaptive budget subsystem of the model builder."""

from __future__ import annotations

from .budget_targets import BudgetAllocation, BudgetLimits, RequestBudget, StopReason
from .calibration import CalibrationModel, apply_calibration, fit_calibration
from .controller import (
    BaselineBudgetController,
    BudgetController,
    ControllerDecision,
    TrainableBudgetController,
    load_controller,
    save_controller,
)
from .evaluation import ControllerEvaluation, compare_controllers, evaluate_controller
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, BudgetFeatures, feature_schema
from .policy import BUILTIN_POLICIES, BudgetPolicy, get_policy
from .policy_dataset import PolicyDataset, PolicySample, build_policy_dataset, save_policy_dataset
from .training_objectives import ControllerTargets, combined_loss, compute_efficiency_score

__all__ = [
    "BaselineBudgetController",
    "BudgetAllocation",
    "BudgetController",
    "BudgetFeatures",
    "BudgetLimits",
    "BudgetPolicy",
    "BUILTIN_POLICIES",
    "CalibrationModel",
    "ControllerDecision",
    "ControllerEvaluation",
    "ControllerTargets",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "PolicyDataset",
    "PolicySample",
    "RequestBudget",
    "StopReason",
    "TrainableBudgetController",
    "apply_calibration",
    "build_policy_dataset",
    "combined_loss",
    "compare_controllers",
    "compute_efficiency_score",
    "evaluate_controller",
    "feature_schema",
    "fit_calibration",
    "get_policy",
    "load_controller",
    "save_controller",
    "save_policy_dataset",
]
