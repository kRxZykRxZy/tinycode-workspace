"""Tests for the resource recovery policy."""

from __future__ import annotations

from tinycode_builder.resources.adaptive_batch import AdaptiveBatchPolicy, choose_micro_batch
from tinycode_builder.resources.memory_estimator import estimate_training_memory
from tinycode_builder.resources.recovery import RecoveryManager
from tinycode_builder.resources.thresholds import ResourceLevel, ResourceThresholds


def test_thresholds_classify_usage() -> None:
    thresholds = ResourceThresholds()
    assert thresholds.classify(0.5) == ResourceLevel.NORMAL
    assert thresholds.classify(0.80) == ResourceLevel.REDUCED
    assert thresholds.classify(0.92) == ResourceLevel.CAUTION
    assert thresholds.classify(0.97) == ResourceLevel.CRITICAL


def test_recovery_reduces_the_micro_batch() -> None:
    manager = RecoveryManager()
    manager.handle_pressure(ResourceLevel.REDUCED)
    assert manager.micro_batch_scale < 1.0
    assert manager.paused is False


def test_critical_pressure_pauses_training() -> None:
    manager = RecoveryManager()
    manager.handle_pressure(ResourceLevel.CRITICAL)
    assert manager.paused is True
    manager.resume()
    assert manager.paused is False


def test_oom_recovery_keeps_going() -> None:
    manager = RecoveryManager()
    event = manager.handle_oom(RuntimeError("out of memory"))
    assert event.level == ResourceLevel.OOM.value
    assert manager.paused is False
    assert manager.micro_batch_scale < 1.0


def test_micro_batch_choice_respects_the_bounds() -> None:
    policy = AdaptiveBatchPolicy(min_micro_batch=1, max_micro_batch=8, target_tokens_per_micro_batch=4096)
    chosen = choose_micro_batch(policy, available_memory_bytes=10**9, estimated_bytes_per_sequence=1024)
    assert 1 <= chosen <= 8


def test_micro_batch_choice_shrinks_under_pressure() -> None:
    policy = AdaptiveBatchPolicy(min_micro_batch=1, max_micro_batch=64)
    generous = choose_micro_batch(policy, 10**12, 1024, current=16)
    tight = choose_micro_batch(policy, 10**6, 1024, current=16)
    assert tight <= generous


def test_memory_estimate_scales_with_the_model() -> None:
    small = estimate_training_memory(1_000_000, 1, 1024, 128, 4)
    large = estimate_training_memory(100_000_000, 1, 1024, 512, 12)
    assert large.total_bytes > small.total_bytes
    assert large.total_gb > 0
