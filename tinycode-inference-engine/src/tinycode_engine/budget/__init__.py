"""The adaptive computation budget layer.

The layer is split into the ten conceptual stages of the platform design;
every stage lives in its own module and can be tested on its own.
"""

from __future__ import annotations

from .budget_allocator import BudgetAllocator, ResourceState
from .complexity_estimator import ComplexityEstimate, ComplexityEstimator
from .confidence import ConfidenceEstimator, UncertaintySignals, entropy_of, token_agreement
from .controller import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    BaselineBudgetController,
    BudgetAllocation,
    BudgetController,
    BudgetFeatures,
    BudgetLimits,
    BudgetStopReason,
    ControllerDecision,
    RequestBudget,
    TrainedBudgetController,
    feature_vector,
)
from .policy import POLICIES, BudgetPolicy, get_policy
from .request_analyzer import AnalyzedRequest, RequestAnalyzer
from .safeguards import SafeguardPolicy, Safeguards
from .scheduler import BudgetScheduler, ContinuationDecision
from .stopping import FinalizationResult, Finalizer
from .telemetry import BudgetRecord, BudgetTelemetry
from .verification import VerificationPipeline, VerificationResult, Verifier

__all__ = [
    "AnalyzedRequest",
    "BaselineBudgetController",
    "BudgetAllocation",
    "BudgetAllocator",
    "BudgetController",
    "BudgetFeatures",
    "BudgetLimits",
    "BudgetPolicy",
    "BudgetRecord",
    "BudgetScheduler",
    "BudgetStopReason",
    "BudgetTelemetry",
    "ComplexityEstimate",
    "ComplexityEstimator",
    "ConfidenceEstimator",
    "ContinuationDecision",
    "ControllerDecision",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FinalizationResult",
    "Finalizer",
    "POLICIES",
    "RequestAnalyzer",
    "RequestBudget",
    "ResourceState",
    "SafeguardPolicy",
    "Safeguards",
    "TrainedBudgetController",
    "UncertaintySignals",
    "VerificationPipeline",
    "VerificationResult",
    "Verifier",
    "entropy_of",
    "feature_vector",
    "get_policy",
    "token_agreement",
]
