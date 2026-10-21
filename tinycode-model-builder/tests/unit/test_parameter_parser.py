"""Tests for the parameter target parser."""

from __future__ import annotations

import pytest

from tinycode_builder.configuration.parameter_parser import (
    DEFAULT_PARAMETER_TARGET,
    SUPPORTED_PARAMETER_TARGETS,
    ParameterTargetError,
    format_parameter_target,
    parse_parameter_target,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1M", 1_000_000),
        ("10M", 10_000_000),
        ("25m", 25_000_000),
        ("50M", 50_000_000),
        ("100M", 100_000_000),
        ("500m", 500_000_000),
        ("1B", 1_000_000_000),
        ("1b", 1_000_000_000),
        ("1000000", 1_000_000),
        (" 10 M ", 10_000_000),
        ("10_000_000", 10_000_000),
    ],
)
def test_parse_parameter_target(text: str, expected: int) -> None:
    parsed = parse_parameter_target(text)
    assert parsed.value == expected


@pytest.mark.parametrize("text", ["", "abc", "10X", "-5M", "1.5M", "M"])
def test_parse_parameter_target_rejects_garbage(text: str) -> None:
    with pytest.raises(ParameterTargetError):
        parse_parameter_target(text)


def test_default_target_is_supported() -> None:
    assert DEFAULT_PARAMETER_TARGET in SUPPORTED_PARAMETER_TARGETS


def test_supported_targets_are_unique_and_positive() -> None:
    assert len(set(SUPPORTED_PARAMETER_TARGETS)) == len(SUPPORTED_PARAMETER_TARGETS)
    assert all(value > 0 for value in SUPPORTED_PARAMETER_TARGETS)


def test_format_round_trip() -> None:
    for value in SUPPORTED_PARAMETER_TARGETS:
        assert parse_parameter_target(format_parameter_target(value)).value == value


def test_integer_input_is_accepted() -> None:
    parsed = parse_parameter_target(10_000_000)
    assert parsed.value == 10_000_000
    assert parsed.supported


def test_boolean_input_is_rejected() -> None:
    with pytest.raises(ParameterTargetError):
        parse_parameter_target(True)
