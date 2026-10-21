"""Tests of the adaptive budget layer."""

from __future__ import annotations

import pytest

from tinycode_engine.budget.budget_allocator import BudgetAllocator, ResourceState
from tinycode_engine.budget.complexity_estimator import ComplexityEstimator
from tinycode_engine.budget.confidence import ConfidenceEstimator, entropy_of
from tinycode_engine.budget.controller import (
    BaselineBudgetController,
    BudgetFeatures,
    BudgetLimits,
    BudgetStopReason,
)
from tinycode_engine.budget.policy import get_policy
from tinycode_engine.budget.request_analyzer import RequestAnalyzer
from tinycode_engine.budget.safeguards import Safeguards
from tinycode_engine.budget.scheduler import BudgetScheduler
from tinycode_engine.budget.verification import Verifier
from tinycode_engine.utils.errors import RequestError


def _analyzer() -> RequestAnalyzer:
    return RequestAnalyzer()


def test_the_request_is_classified() -> None:
    analyzer = _analyzer()
    assert analyzer.classify("fix the bug in this function") == "debug"
    assert analyzer.classify("explain what a loop does") == "explain"
    assert analyzer.classify("write a python function to add numbers") == "code"
    assert analyzer.classify("translate this to json") == "transform"


def test_an_empty_prompt_is_rejected() -> None:
    with pytest.raises(RequestError):
        _analyzer().analyze("id", "   ", 3, 128)


def test_an_oversized_prompt_is_rejected() -> None:
    with pytest.raises(RequestError):
        _analyzer().analyze("id", "hello", 2048, 128)


def test_the_complexity_estimate_is_bounded() -> None:
    analyzer = _analyzer()
    request = analyzer.analyze("id", "write a function", 20, 128)
    estimate = ComplexityEstimator().estimate(request)
    assert 0.0 <= estimate.score <= 1.0
    assert estimate.task_category == "code"


def test_the_baseline_policy_scales_with_complexity() -> None:
    controller = BaselineBudgetController()
    easy = controller.decide(BudgetFeatures(complexity_score=0.1), 256)
    hard = controller.decide(BudgetFeatures(complexity_score=0.9), 256)
    assert easy.policy == "minimal"
    assert hard.generation_tokens > easy.generation_tokens


def test_the_allocation_respects_the_server_limits() -> None:
    limits = BudgetLimits(max_generation_tokens=128, max_internal_steps=2)
    allocator = BudgetAllocator(limits=limits)
    analyzer = _analyzer()
    request = analyzer.analyze("id", "write a complicated function", 40, 4096)
    estimate = ComplexityEstimator().estimate(request)
    budget = allocator.allocate(request, estimate, uncertainty=0.5, resources=ResourceState())
    assert budget.allocation.generation_tokens <= 128
    assert budget.allocation.internal_steps <= 2


def test_a_user_budget_cannot_exceed_the_server_limits() -> None:
    from tinycode_engine.budget.safeguards import SafeguardPolicy

    limits = BudgetLimits(max_generation_tokens=64)
    safeguards = Safeguards(SafeguardPolicy(limits=limits))
    request = _analyzer().analyze("id", "hello", 5, 4096)
    estimate = ComplexityEstimator().estimate(request)
    budget = BudgetAllocator(limits=limits).allocate(
        request, estimate, uncertainty=0.0, resources=ResourceState()
    )
    assert budget.allocation.generation_tokens <= 64
    assert safeguards.clamp_max_tokens(100000) == 64


def test_the_scheduler_stops_when_the_budget_is_exhausted() -> None:
    allocator = BudgetAllocator()
    analyzer = _analyzer()
    request = analyzer.analyze("id", "hello", 5, 32)
    estimate = ComplexityEstimator().estimate(request)
    budget = allocator.allocate(request, estimate, uncertainty=0.0, resources=ResourceState())
    budget.consumed.generation_tokens = budget.allocation.generation_tokens
    budget.consumed.internal_steps = budget.allocation.internal_steps

    signals = ConfidenceEstimator().from_logits([])
    decision = BudgetScheduler().decide(budget, signals)
    assert decision.action in {"stop_budget_exhausted", "accept"}


def test_the_verifier_detects_invalid_json() -> None:
    verifier = Verifier(expected_json=True)
    result = verifier.verify("this is not json", task_category="structured")
    assert not result.passed
    assert result.errors


def test_the_verifier_accepts_valid_json() -> None:
    verifier = Verifier(expected_json=True)
    result = verifier.verify('{"name": "tinycode"}', task_category="structured")
    assert result.passed


def test_the_verifier_requires_a_code_fence_for_code_tasks() -> None:
    verifier = Verifier()
    assert not verifier.verify("def add(a, b): return a + b").passed
    assert verifier.verify("```python\ndef add(a, b): return a + b\n```").passed


def test_the_entropy_of_a_uniform_distribution_is_maximal() -> None:
    import numpy as np

    uniform = np.zeros(10, dtype=np.float32)
    peaked = np.array([10.0] + [0.0] * 9, dtype=np.float32)
    assert entropy_of(uniform) > entropy_of(peaked)


def test_the_stop_reason_is_tracked() -> None:
    assert BudgetStopReason.BUDGET_EXHAUSTED != BudgetStopReason.ACCEPTED


def test_the_policies_are_computation_policies_not_context_tiers() -> None:
    for name in ("minimal", "standard", "extended", "verification"):
        policy = get_policy(name)
        assert policy.max_generation_tokens <= 1024
