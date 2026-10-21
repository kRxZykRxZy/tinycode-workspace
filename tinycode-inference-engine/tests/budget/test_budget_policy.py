"""Tests of the budget policy behaviour under load."""

from __future__ import annotations

from tinycode_engine.budget.budget_allocator import BudgetAllocator, ResourceState
from tinycode_engine.budget.complexity_estimator import ComplexityEstimator
from tinycode_engine.budget.controller import BudgetLimits
from tinycode_engine.budget.request_analyzer import RequestAnalyzer


def _budget(complexity: float, resources: ResourceState, limits: BudgetLimits | None = None):
    analyzer = RequestAnalyzer()
    request = analyzer.analyze("id", "write a function that sorts a list", 32, 256)
    estimate = ComplexityEstimator().estimate(request)
    estimate.score = complexity
    allocator = BudgetAllocator(limits=limits or BudgetLimits())
    return allocator.allocate(request, estimate, uncertainty=0.2, resources=resources)


def test_queue_pressure_reduces_the_allocation() -> None:
    relaxed = _budget(0.6, ResourceState(queue_depth=0, queue_capacity=64))
    pressured = _budget(0.6, ResourceState(queue_depth=64, queue_capacity=64))
    assert pressured.allocation.generation_tokens < relaxed.allocation.generation_tokens


def test_memory_pressure_reduces_the_allocation() -> None:
    relaxed = _budget(0.6, ResourceState(memory_pressure=0.0))
    pressured = _budget(0.6, ResourceState(memory_pressure=1.0))
    assert pressured.allocation.generation_tokens <= relaxed.allocation.generation_tokens


def test_the_allocation_never_exceeds_the_limits() -> None:
    limits = BudgetLimits(max_generation_tokens=32, max_internal_steps=1, max_verification_passes=0)
    budget = _budget(0.95, ResourceState(queue_depth=64, queue_capacity=64), limits)
    assert budget.allocation.generation_tokens <= 32
    assert budget.allocation.internal_steps <= 1
    assert budget.allocation.verification_passes == 0


def test_the_deadline_is_set_from_the_latency_target() -> None:
    analyzer = RequestAnalyzer()
    request = analyzer.analyze("id", "hello", 5, 64)
    request.latency_target_ms = 5_000
    estimate = ComplexityEstimator().estimate(request)
    budget = BudgetAllocator().allocate(
        request, estimate, uncertainty=0.0, resources=ResourceState()
    )
    assert budget.allocation.deadline_ms == 5_000
    assert budget.deadline is not None
