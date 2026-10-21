"""Tests for the adaptive budget controller (builder side)."""

from __future__ import annotations

from tinycode_builder.budget.calibration import apply_calibration, fit_calibration
from tinycode_builder.budget.controller import BaselineBudgetController
from tinycode_builder.budget.features import FEATURE_NAMES, BudgetFeatures, feature_schema
from tinycode_builder.budget.policy import BUILTIN_POLICIES, get_policy


def test_feature_vector_matches_the_schema() -> None:
    features = BudgetFeatures(prompt_tokens=100, task_category="debug")
    vector = features.to_vector()
    assert len(vector) == len(FEATURE_NAMES)
    assert feature_schema()["version"] == "budget_features.v1"


def test_baseline_policy_scales_with_complexity() -> None:
    controller = BaselineBudgetController()
    easy = controller.decide(BudgetFeatures(complexity_score=0.1), 256)
    hard = controller.decide(BudgetFeatures(complexity_score=0.9), 256)
    assert easy.policy == "minimal"
    assert hard.policy in {"extended", "verification"}
    assert hard.generation_tokens >= easy.generation_tokens


def test_baseline_respects_a_latency_target() -> None:
    controller = BaselineBudgetController()
    relaxed = controller.decide(
        BudgetFeatures(complexity_score=0.4, latency_target_ms=60_000), 512
    )
    tight = controller.decide(
        BudgetFeatures(complexity_score=0.4, latency_target_ms=2_000), 512
    )
    assert tight.generation_tokens <= relaxed.generation_tokens


def test_verification_is_only_enabled_when_justified() -> None:
    controller = BaselineBudgetController()
    without = controller.decide(BudgetFeatures(complexity_score=0.3), 256)
    with_verification = controller.decide(
        BudgetFeatures(complexity_score=0.9, wants_verification=True), 256
    )
    assert without.verification_passes == 0
    assert with_verification.verification_passes >= 1


def test_policy_bounds_are_respected() -> None:
    policy = get_policy("standard")
    assert policy.clamp_tokens(10_000) == policy.max_generation_tokens
    assert policy.clamp_tokens(1) == policy.min_generation_tokens
    assert set(BUILTIN_POLICIES) == {"minimal", "standard", "extended", "verification"}


def test_calibration_is_monotonic_and_bounded() -> None:
    model = fit_calibration([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    low = apply_calibration(0.1, model)
    high = apply_calibration(0.9, model)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= low


def test_calibration_without_data_is_safe() -> None:
    model = fit_calibration([], [])
    assert 0.0 <= apply_calibration(0.5, model) <= 1.0
