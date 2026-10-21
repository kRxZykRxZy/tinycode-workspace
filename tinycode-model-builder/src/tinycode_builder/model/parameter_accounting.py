"""Strict trainable-parameter accounting.

The builder guarantees that the *actual* number of trainable parameters
equals the user selected target exactly.  Two independent mechanisms are
provided:

``analytic_parameter_count``
    A closed form formula that does not need a model instance.  It is the
    formula used by the architecture solver.

``count_trainable_parameters``
    Counts ``requires_grad`` tensors of a live module.  The builder asserts
    that the two agree, which protects against drift between the formula
    and the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .architecture import TransformerConfig

__all__ = [
    "ParameterBreakdown",
    "analytic_parameter_count",
    "count_trainable_parameters",
    "parameter_breakdown",
    "verify_parameter_count",
    "ParameterCountMismatch",
]


class ParameterCountMismatch(AssertionError):
    """Raised when the analytic and measured parameter counts disagree."""


@dataclass(frozen=True)
class ParameterBreakdown:
    """Per-component trainable parameter counts."""

    token_embedding: int
    position_embedding: int
    transformer_layers: int
    final_norm: int
    output_head: int

    @property
    def total(self) -> int:
        return (
            self.token_embedding
            + self.position_embedding
            + self.transformer_layers
            + self.final_norm
            + self.output_head
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "transformer_layers": self.transformer_layers,
            "final_norm": self.final_norm,
            "output_head": self.output_head,
            "total": self.total,
        }


def _per_layer_parameters(config: TransformerConfig, d_ff: int | None = None) -> dict[str, int]:
    """Return the parameter counts of a single transformer block."""

    d = config.d_model
    ff = config.d_ff if d_ff is None else d_ff
    norm = config.norm_parameters_per_norm
    attention_weights = 4 * d * d
    attention_biases = 4 * d if config.qkv_bias else 0
    ff_weights = d * ff + ff * d
    ff_biases = (ff + d) if config.ff_bias else 0
    return {
        "norm1": norm,
        "attention": attention_weights + attention_biases,
        "norm2": norm,
        "feed_forward": ff_weights + ff_biases,
    }


def parameter_breakdown(config: TransformerConfig) -> ParameterBreakdown:
    """Return the analytic breakdown of trainable parameters."""

    d = config.d_model
    token_embedding = config.vocab_size * d
    position_embedding = (
        config.context_length * d if config.position_encoding == "learned" else 0
    )
    per_layer = _per_layer_parameters(config)
    layer_total = sum(per_layer.values())
    transformer_layers = config.n_layers * layer_total
    final_norm = config.norm_parameters_per_norm
    if config.tie_embeddings:
        output_head = 0
    else:
        output_head = config.vocab_size * d + (config.vocab_size if config.head_bias else 0)
    return ParameterBreakdown(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        transformer_layers=transformer_layers,
        final_norm=final_norm,
        output_head=output_head,
    )


def analytic_parameter_count(config: TransformerConfig) -> int:
    """Return the exact number of trainable parameters implied by ``config``."""

    return parameter_breakdown(config).total


def count_trainable_parameters(module: Any) -> int:
    """Count the trainable parameters of a live ``torch`` module."""

    total = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return int(total)


def _named_trainable_tensors(module: Any) -> Iterable[tuple[str, Any]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def verify_parameter_count(module: Any, config: TransformerConfig) -> int:
    """Verify that the live module matches the analytic count.

    Raises:
        ParameterCountMismatch: when the measured count differs from the
            analytic count or from ``config``'s expectation.
    """

    measured = count_trainable_parameters(module)
    analytic = analytic_parameter_count(config)
    if measured != analytic:
        raise ParameterCountMismatch(
            f"Measured trainable parameters ({measured}) do not match the analytic "
            f"count ({analytic}) for configuration {config.to_dict()}"
        )
    return measured
