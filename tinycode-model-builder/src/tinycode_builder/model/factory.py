"""Architecture solver: find a configuration with an exact parameter count.

The solver searches a structured space of decoder architectures

    (vocabulary, embedding tying, positional encoding, normalisation type,
     bias configuration) x (d_model, n_layers, d_ff)

and keeps the candidate whose *analytic* trainable parameter count is
exactly the requested target.  Because ``d_ff`` enters the formula
linearly, an exact solution exists whenever the remaining budget divides
evenly by the number of feed-forward coefficients, which the search
exploits instead of adding ballast parameters.

The formula implemented here is mirrored (independently) in the inference
engine; the cross repository integration test asserts that both agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .architecture import ARCHITECTURE_ID, TransformerConfig
from .parameter_accounting import analytic_parameter_count

__all__ = [
    "ArchitectureSolution",
    "ArchitectureSolverError",
    "NoExactArchitectureError",
    "solve_architecture",
    "choose_head_count",
]


class ArchitectureSolverError(RuntimeError):
    """Base error for architecture solving."""


class NoExactArchitectureError(ArchitectureSolverError):
    """Raised when no architecture matches the requested parameter target."""


@dataclass
class ArchitectureSolution:
    """An architecture whose trainable parameter count matches the target exactly."""

    config: TransformerConfig
    target: int
    actual: int
    vocabulary_fallback: bool = False
    warnings: list[str] = field(default_factory=list)
    score: tuple[float, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "target": self.target,
            "actual": self.actual,
            "vocabulary_fallback": self.vocabulary_fallback,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# fast analytic helpers (no dataclass construction inside the hot loop)
# --------------------------------------------------------------------------
def _norm_params(d_model: int, norm_type: str) -> int:
    return 2 * d_model if norm_type == "layernorm" else d_model


def _base_count(
    vocab: int,
    context_length: int,
    d_model: int,
    n_layers: int,
    norm_per_norm: int,
    qkv_bias: bool,
    tie_embeddings: bool,
    head_bias: bool,
    learned_positions: bool,
) -> int:
    """Parameter count excluding the feed-forward width (``d_ff``)."""

    total = vocab * d_model
    if learned_positions:
        total += context_length * d_model
    per_layer = (
        2 * norm_per_norm
        + 4 * d_model * d_model
        + (4 * d_model if qkv_bias else 0)
    )
    total += n_layers * per_layer
    total += norm_per_norm
    if not tie_embeddings:
        total += vocab * d_model
        if head_bias:
            total += vocab
    return total


def choose_head_count(d_model: int, min_head_dim: int = 16) -> int:
    """Choose a sensible attention head count for ``d_model``."""

    for n_heads in (8, 12, 6, 4, 16, 2, 1):
        if d_model % n_heads == 0:
            head_dim = d_model // n_heads
            if head_dim >= max(min_head_dim // 2, 4):
                return n_heads
    for n_heads in range(1, d_model + 1):
        if d_model % n_heads == 0:
            return n_heads
    return 1


def _score(
    config: TransformerConfig,
    preferred_vocab: int,
    preferred_ratio: float,
    min_head_dim: int,
) -> tuple[float, ...]:
    """Lower is better."""

    vocab_penalty = 0 if config.vocab_size == preferred_vocab else 1_000_000
    head_dim = config.head_dim
    head_penalty = 0 if head_dim >= min_head_dim else (100 - head_dim) * 10
    ratio_error = abs(config.ff_ratio - preferred_ratio)
    layer_penalty = abs(config.n_layers - 8)
    return (vocab_penalty, head_penalty, round(ratio_error, 4), layer_penalty, config.d_model)


def solve_architecture(
    target: int,
    vocab_preference: Iterable[int] = (98_304, 49_152, 32_768, 16_384, 8_192),
    preferred_vocab: int | None = None,
    max_layers: int = 64,
    preferred_ff_ratio: float = 4.0,
    min_head_dim: int = 16,
    context_length: int = 1024,
    min_d_model: int = 16,
    max_d_model: int = 4096,
    d_model_step: int = 4,
) -> ArchitectureSolution:
    """Find an architecture with exactly ``target`` trainable parameters.

    Args:
        target: requested number of trainable parameters.
        vocab_preference: vocabulary sizes to try, most preferred first.
        preferred_vocab: vocabulary that must be used when achievable.
        max_layers: maximum number of transformer blocks to consider.
        preferred_ff_ratio: preferred feed-forward expansion ratio.
        min_head_dim: preferred minimum head dimension.
        context_length: fixed context length (always 1024).

    Returns:
        The best scoring exact solution.

    Raises:
        NoExactArchitectureError: when no exact solution exists.
    """

    if target <= 0:
        raise ArchitectureSolverError("Parameter target must be positive.")

    candidates = [v for v in vocab_preference if v > 0]
    if preferred_vocab is not None:
        if preferred_vocab not in candidates:
            candidates.insert(0, preferred_vocab)
        reference_vocab = preferred_vocab
    else:
        if not candidates:
            raise ArchitectureSolverError("At least one vocabulary size must be provided.")
        reference_vocab = candidates[0]

    for vocab in candidates:
        solution = _solve_for_vocab(
            target=target,
            vocab=vocab,
            max_layers=max_layers,
            preferred_ff_ratio=preferred_ff_ratio,
            min_head_dim=min_head_dim,
            context_length=context_length,
            reference_vocab=reference_vocab,
            min_d_model=min_d_model,
            max_d_model=max_d_model,
            d_model_step=d_model_step,
        )
        if solution is not None:
            solution.vocabulary_fallback = vocab != reference_vocab
            if solution.vocabulary_fallback:
                solution.warnings.append(
                    f"Vocabulary reduced to {vocab} to reach the {target:,} parameter "
                    f"target exactly; the preferred vocabulary of {reference_vocab:,} "
                    "cannot fit this target."
                )
            return solution

    raise NoExactArchitectureError(
        f"No supported architecture reaches exactly {target:,} trainable parameters. "
        "Try a different supported target or allow a reduced vocabulary."
    )


def _solve_for_vocab(
    target: int,
    vocab: int,
    max_layers: int,
    preferred_ff_ratio: float,
    min_head_dim: int,
    context_length: int,
    reference_vocab: int,
    min_d_model: int,
    max_d_model: int,
    d_model_step: int,
) -> ArchitectureSolution | None:
    best: ArchitectureSolution | None = None
    best_score: tuple[float, ...] | None = None

    for tie_embeddings in (False, True):
        head_bias_options = (False,) if tie_embeddings else (False, True)
        for position_encoding in ("learned", "sinusoidal"):
            learned_positions = position_encoding == "learned"
            for norm_type in ("layernorm", "rmsnorm"):
                norm_per = _norm_params(min_d_model, norm_type)
                for qkv_bias in (True, False):
                    for ff_bias in (True, False):
                        for head_bias in head_bias_options:
                            found, found_score = _scan_dimensions(
                                target=target,
                                vocab=vocab,
                                tie_embeddings=tie_embeddings,
                                learned_positions=learned_positions,
                                norm_type=norm_type,
                                qkv_bias=qkv_bias,
                                ff_bias=ff_bias,
                                head_bias=head_bias,
                                max_layers=max_layers,
                                preferred_ff_ratio=preferred_ff_ratio,
                                min_head_dim=min_head_dim,
                                context_length=context_length,
                                reference_vocab=reference_vocab,
                                min_d_model=min_d_model,
                                max_d_model=max_d_model,
                                d_model_step=d_model_step,
                                best=best,
                                best_score=best_score,
                            )
                            best = found
                            best_score = found_score
    return best


def _scan_dimensions(
    *,
    target: int,
    vocab: int,
    tie_embeddings: bool,
    learned_positions: bool,
    norm_type: str,
    qkv_bias: bool,
    ff_bias: bool,
    head_bias: bool,
    max_layers: int,
    preferred_ff_ratio: float,
    min_head_dim: int,
    context_length: int,
    reference_vocab: int,
    min_d_model: int,
    max_d_model: int,
    d_model_step: int,
    best: ArchitectureSolution | None,
    best_score: tuple[float, ...] | None,
) -> tuple[ArchitectureSolution | None, tuple[float, ...] | None]:
    # An upper bound on d_model: the embedding matrix alone must fit.
    embedding_weight = vocab * (1 if tie_embeddings else 2)
    d_model_limit = min(max_d_model, max(min_d_model, target // max(embedding_weight, 1)))

    d_model = min_d_model
    while d_model <= d_model_limit:
        norm_per = _norm_params(d_model, norm_type)
        per_layer = 2 * norm_per + 4 * d_model * d_model + (4 * d_model if qkv_bias else 0)
        constant = (
            vocab * d_model
            + (context_length * d_model if learned_positions else 0)
            + norm_per
            + (
                vocab * d_model + (vocab if head_bias else 0)
                if not tie_embeddings
                else 0
            )
        )
        if constant >= target:
            d_model += d_model_step
            continue
        # Only the layer counts whose base still fits into the target are
        # worth probing.
        layer_limit = min(max_layers, (target - constant) // max(1, per_layer))
        for n_layers in range(1, max(1, layer_limit) + 1):
            base = constant + n_layers * per_layer
            remaining = target - base
            if remaining <= 0:
                break  # more layers only increase the count
            coefficient = n_layers * (2 * d_model + (1 if ff_bias else 0))
            if remaining % coefficient != 0:
                continue
            d_ff = remaining // coefficient
            if d_ff < 1:
                continue
            config = TransformerConfig(
                vocab_size=vocab,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=choose_head_count(d_model, min_head_dim),
                d_ff=d_ff,
                context_length=context_length,
                norm_type=norm_type,
                activation="silu",
                position_encoding="learned" if learned_positions else "sinusoidal",
                qkv_bias=qkv_bias,
                ff_bias=ff_bias,
                head_bias=head_bias,
                tie_embeddings=tie_embeddings,
            )
            actual = analytic_parameter_count(config)
            if actual != target:  # pragma: no cover - guards formula drift
                continue
            score = _score(config, reference_vocab, preferred_ff_ratio, min_head_dim)
            if best_score is None or score < best_score:
                warnings: list[str] = []
                if config.ff_ratio < 1.0 or config.ff_ratio > 12.0:
                    warnings.append(
                        f"Unusual feed-forward ratio {config.ff_ratio:.2f}; the exact "
                        "parameter target constrained the architecture."
                    )
                if config.head_dim < min_head_dim:
                    warnings.append(
                        f"Head dimension {config.head_dim} is below the preferred "
                        f"minimum {min_head_dim}."
                    )
                best = ArchitectureSolution(
                    config=config,
                    target=target,
                    actual=actual,
                    warnings=warnings,
                    score=score,
                )
                best_score = score
        d_model += d_model_step
    return best, best_score
