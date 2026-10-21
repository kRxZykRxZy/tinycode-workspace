"""Tests for the parameter accounting and the architecture solver."""

from __future__ import annotations

import pytest

from tinycode_builder.configuration.parameter_parser import SUPPORTED_PARAMETER_TARGETS
from tinycode_builder.model.architecture import TransformerConfig
from tinycode_builder.model.factory import NoExactArchitectureError, solve_architecture
from tinycode_builder.model.parameter_accounting import (
    ParameterCountMismatch,
    analytic_parameter_count,
    count_trainable_parameters,
    parameter_breakdown,
    verify_parameter_count,
)


def test_analytic_count_matches_the_live_model(tiny_config: TransformerConfig) -> None:
    from tinycode_builder.model.causal_model import build_model

    model = build_model(tiny_config)
    assert count_trainable_parameters(model) == analytic_parameter_count(tiny_config)


def test_verify_parameter_count_detects_drift(tiny_config: TransformerConfig) -> None:
    from tinycode_builder.model.causal_model import build_model

    model = build_model(tiny_config)
    assert verify_parameter_count(model, tiny_config) > 0

    with pytest.raises(ParameterCountMismatch):
        other = TransformerConfig(**{**tiny_config.to_dict(), "d_ff": tiny_config.d_ff + 1})
        verify_parameter_count(model, other)


def test_breakdown_sums_to_the_total(tiny_config: TransformerConfig) -> None:
    breakdown = parameter_breakdown(tiny_config)
    assert breakdown.total == sum(
        [
            breakdown.token_embedding,
            breakdown.position_embedding,
            breakdown.transformer_layers,
            breakdown.final_norm,
            breakdown.output_head,
        ]
    )
    assert breakdown.total == analytic_parameter_count(tiny_config)


def test_tied_embeddings_do_not_add_head_parameters() -> None:
    untied = TransformerConfig(
        vocab_size=512, d_model=32, n_layers=1, n_heads=4, d_ff=64, tie_embeddings=False
    )
    tied = TransformerConfig(
        vocab_size=512, d_model=32, n_layers=1, n_heads=4, d_ff=64, tie_embeddings=True
    )
    assert analytic_parameter_count(tied) == analytic_parameter_count(untied) - 512 * 32


@pytest.mark.parametrize("target", [1_000_000, 10_000_000])
def test_solver_reaches_the_target_exactly(target: int) -> None:
    solution = solve_architecture(target, vocab_preference=[98_304, 16_384, 4_096])
    assert solution.actual == target
    assert analytic_parameter_count(solution.config) == target
    assert solution.config.context_length == 1024


def test_solver_prefers_the_requested_vocabulary() -> None:
    solution = solve_architecture(10_000_000, vocab_preference=[98_304, 16_384])
    assert solution.config.vocab_size == 98_304
    assert solution.vocabulary_fallback is False


def test_solver_reports_a_vocabulary_fallback() -> None:
    solution = solve_architecture(1_000_000, vocab_preference=[98_304, 4_096])
    assert solution.actual == 1_000_000
    if solution.vocabulary_fallback:
        assert solution.warnings


def test_solver_rejects_impossible_targets() -> None:
    with pytest.raises(NoExactArchitectureError):
        solve_architecture(7, vocab_preference=[4_096])


def test_architecture_ids_are_stable() -> None:
    solution = solve_architecture(10_000_000, vocab_preference=[98_304])
    assert solution.config.architecture_id == "tinycode-decoder-v1"
    assert solution.config.context_length == 1024


def test_all_supported_targets_are_reachable_or_explained() -> None:
    for target in SUPPORTED_PARAMETER_TARGETS:
        try:
            solution = solve_architecture(target, vocab_preference=[98_304, 49_152, 32_768, 16_384, 8_192])
        except NoExactArchitectureError as exc:  # pragma: no cover - documented limitation
            pytest.fail(f"target {target} is not reachable: {exc}")
        assert solution.actual == target
