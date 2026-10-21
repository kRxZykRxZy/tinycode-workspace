"""Attention for the numpy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["split_heads", "merge_heads", "attention", "attention_cached"]


def split_heads(x: Any, n_heads: int) -> Any:
    """Split ``(seq, dim)`` into ``(heads, seq, head_dim)``."""

    seq, dim = x.shape
    head_dim = dim // n_heads
    return x.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)


def merge_heads(x: Any) -> Any:
    """Merge ``(heads, seq, head_dim)`` back into ``(seq, dim)``."""

    heads, seq, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq, heads * head_dim)


def _scores(query: Any, key: Any, scale: float) -> Any:
    return np.matmul(query, np.swapaxes(key, -1, -2)) * scale


def _softmax(x: Any) -> Any:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def attention(query: Any, key: Any, value: Any, scale: float | None = None) -> Any:
    """Causal attention over a complete sequence.

    Args:
        query: ``(heads, seq, head_dim)``
        key: ``(heads, seq, head_dim)``
        value: ``(heads, seq, head_dim)``
    """

    head_dim = query.shape[-1]
    factor = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    scores = _scores(query, key, factor)
    seq = scores.shape[-1]
    mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
    weights = _softmax(scores + mask)
    return np.matmul(weights, value)


def attention_cached(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    cache_position: int,
    scale: float | None = None,
) -> Any:
    """Attention against a KV cache holding ``cache_position`` entries.

    The queries correspond to the *last* ``query.shape[1]`` positions of
    the cache, so a causal mask is applied when more than one query is
    processed at once (the prefill).
    """

    head_dim = query.shape[-1]
    factor = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    keys = key_cache[:, :cache_position]
    values = value_cache[:, :cache_position]
    scores = _scores(query, keys, factor)

    query_length = query.shape[1]
    if query_length > 1:
        # The query at the relative position i corresponds to the absolute
        # position cache_position - query_length + i and may only attend to
        # the keys up to that position.
        offset = cache_position - query_length
        mask = np.triu(
            np.full((query_length, cache_position), -np.inf, dtype=np.float32),
            k=offset + 1,
        )
        scores = scores + mask

    weights = _softmax(scores)
    return np.matmul(weights, values)
