"""Validation of the merged builder configuration."""

from __future__ import annotations

from typing import Any

from .immutable_constraints import CONTEXT_LENGTH
from .parameter_parser import SUPPORTED_PARAMETER_TARGETS

__all__ = ["ConfigValidationError", "validate_config", "describe_config_issues"]


class ConfigValidationError(ValueError):
    """Raised when the merged configuration is not usable."""


_REQUIRED_SECTIONS = (
    "project",
    "model",
    "training",
    "dataset",
    "tokenizer",
    "optimization",
    "checkpoint",
    "resource_policy",
    "evaluation",
    "export",
    "dashboard",
    "budget_controller",
)


def describe_config_issues(config: dict[str, Any]) -> list[str]:
    """Return a list of human readable configuration problems."""

    issues: list[str] = []

    for section in _REQUIRED_SECTIONS:
        if section not in config:
            issues.append(f"missing configuration section '{section}'")

    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}

    target = model.get("parameter_target")
    if not isinstance(target, int) or target <= 0:
        issues.append("model.parameter_target must be a positive integer")
    elif target not in SUPPORTED_PARAMETER_TARGETS:
        issues.append(
            f"model.parameter_target {target} is not one of the supported targets "
            f"{list(SUPPORTED_PARAMETER_TARGETS)}"
        )

    vocab = model.get("vocab_size")
    if not isinstance(vocab, int) or vocab <= 0:
        issues.append("model.vocab_size must be a positive integer")

    tokens = training.get("target_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        issues.append("training.target_tokens must be a positive integer")

    micro = training.get("micro_batch_size", 1)
    accum = training.get("gradient_accumulation_steps", 1)
    if micro <= 0:
        issues.append("training.micro_batch_size must be >= 1")
    if accum <= 0:
        issues.append("training.gradient_accumulation_steps must be >= 1")

    tokenizer_vocab = config.get("tokenizer", {}).get("vocab_size")
    if isinstance(tokenizer_vocab, int) and tokenizer_vocab <= 0:
        issues.append("tokenizer.vocab_size must be a positive integer")

    resource = config.get("resource_policy", {}) or {}
    high = resource.get("cpu_memory_high", 0.75)
    critical = resource.get("cpu_memory_critical", 0.95)
    if not 0 < high <= critical <= 1:
        issues.append("resource_policy watermarks must satisfy 0 < high <= critical <= 1")

    return issues


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the configuration and return it unchanged.

    Raises:
        ConfigValidationError: when the configuration contains problems.
    """

    issues = describe_config_issues(config)
    if issues:
        raise ConfigValidationError("; ".join(issues))
    return config


def validate_context_usage(config: dict[str, Any]) -> int:
    """Return the fixed context length, asserting that config does not change it."""

    model = config.get("model", {}) or {}
    declared = model.get("context_length")
    if declared is not None and declared != CONTEXT_LENGTH:
        raise ConfigValidationError(
            f"model.context_length must be {CONTEXT_LENGTH}, got {declared}"
        )
    return CONTEXT_LENGTH
