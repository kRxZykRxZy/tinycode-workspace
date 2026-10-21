"""Sampling: temperature, top-p and top-k."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["apply_temperature", "top_p_filter", "top_k_filter", "sample_token"]


def apply_temperature(logits: Any, temperature: float) -> Any:
    """Scale the logits by the temperature."""

    values = np.asarray(logits, dtype=np.float32)
    if temperature <= 0:
        return values
    return values / float(temperature)


def top_k_filter(logits: Any, k: int) -> Any:
    """Keep only the ``k`` highest scoring tokens."""

    values = np.asarray(logits, dtype=np.float32).copy()
    if k <= 0 or k >= values.shape[-1]:
        return values
    threshold = np.partition(values, -k)[-k]
    values[values < threshold] = -np.inf
    return values


def top_p_filter(logits: Any, top_p: float) -> Any:
    """Keep the smallest set of tokens whose probability mass reaches ``top_p``."""

    values = np.asarray(logits, dtype=np.float32).copy()
    if top_p <= 0 or top_p >= 1:
        return values
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    total = np.sum(probabilities)
    if total <= 0:  # pragma: no cover - defensive
        return values
    probabilities = probabilities / total
    ordered = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[ordered])
    cutoff = int(np.searchsorted(cumulative, top_p) + 1)
    keep = ordered[:cutoff]
    mask = np.full_like(values, -np.inf)
    mask[keep] = values[keep]
    return mask


def sample_token(
    logits: Any,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    rng: Any = None,
) -> int:
    """Sample a token from the (processed) logits."""

    values = apply_temperature(logits, temperature)
    if top_k:
        values = top_k_filter(values, top_k)
    if top_p and top_p < 1.0:
        values = top_p_filter(values, top_p)

    if temperature <= 0:
        return int(np.argmax(values))

    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(values))
    probabilities = probabilities / total
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(len(probabilities), p=probabilities))
