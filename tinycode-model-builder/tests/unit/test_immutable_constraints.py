"""Tests for the immutable context length constraint."""

from __future__ import annotations

import pytest

from tinycode_builder.configuration.immutable_constraints import (
    CONTEXT_LENGTH,
    ContextLengthViolation,
    assert_context_length,
    is_context_length_override_attempt,
    max_prompt_tokens,
    split_token_budget,
)


def test_context_length_is_1024() -> None:
    assert CONTEXT_LENGTH == 1024


def test_assert_context_length_accepts_only_1024() -> None:
    assert assert_context_length(1024) == 1024
    for value in (0, 1, 512, 1023, 1025, 2048, 4096):
        with pytest.raises(ContextLengthViolation):
            assert_context_length(value)


def test_max_prompt_tokens_reserves_output_room() -> None:
    assert max_prompt_tokens(0) == 1024
    assert max_prompt_tokens(300) == 724


def test_max_prompt_tokens_rejects_impossible_reservation() -> None:
    with pytest.raises(ValueError):
        max_prompt_tokens(1024)
    with pytest.raises(ValueError):
        max_prompt_tokens(2000)


def test_split_token_budget_keeps_total_inside_the_context() -> None:
    prompt, output = split_token_budget(900, 300)
    assert prompt + output == 1024
    assert output == 124


def test_split_token_budget_rejects_oversized_prompt() -> None:
    with pytest.raises(ContextLengthViolation):
        split_token_budget(2000, 10)


@pytest.mark.parametrize(
    "key",
    [
        "context_length",
        "max_context",
        "n_positions",
        "seq_len",
        "rope_scaling",
        "max_position_embeddings",
    ],
)
def test_context_override_attempts_are_detected(key: str) -> None:
    assert is_context_length_override_attempt({key: 4096}) == key


def test_unrelated_keys_are_not_flagged() -> None:
    assert is_context_length_override_attempt({"batch_size": 4}) is None
    assert is_context_length_override_attempt({}) is None
