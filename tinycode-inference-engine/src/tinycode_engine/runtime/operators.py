"""The operators used by the TinyCode model.

All operators are implemented with numpy so that the engine works without
PyTorch.  The native library (see ``native/``) can accelerate a subset of
them; :class:`tinycode_engine.backends.cpu.operators.CpuOperators` selects
between the native and the numpy implementation at runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "OperatorSet",
    "NumpyOperators",
    "embedding_lookup",
    "linear",
    "layer_norm",
    "rms_norm",
    "softmax",
    "silu",
    "gelu",
    "sinusoidal_positions",
    "causal_attention",
    "attention_with_cache",
    "apply_rope",
]


def embedding_lookup(weight: Any, token_ids: Any) -> Any:
    """Look up embeddings for ``token_ids``."""

    ids = np.asarray(token_ids, dtype=np.int64)
    return weight[ids]


def linear(x: Any, weight: Any, bias: Any = None) -> Any:
    """Compute ``x @ weight.T (+ bias)``."""

    result = np.matmul(x, weight.T)
    if bias is not None:
        result = result + bias
    return result


def layer_norm(x: Any, gain: Any, bias: Any = None, eps: float = 1e-5) -> Any:
    """Layer normalisation over the last dimension."""

    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(variance + eps)
    normed = normed * gain
    if bias is not None:
        normed = normed + bias
    return normed


def rms_norm(x: Any, gain: Any, eps: float = 1e-5) -> Any:
    """Root mean square normalisation over the last dimension."""

    variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(variance + eps))
    return normed * gain


def softmax(x: Any, axis: int = -1) -> Any:
    """Numerically stable softmax."""

    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def silu(x: Any) -> Any:
    """SiLU (swish) activation."""

    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-x))


def gelu(x: Any) -> Any:
    """Tanh approximation of the GELU activation."""

    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def sinusoidal_positions(length: int, dim: int) -> Any:
    """Return sinusoidal position embeddings of shape ``(length, dim)``."""

    position = np.arange(length, dtype=np.float32)[:, None]
    index = np.arange(0, dim, 2, dtype=np.float32)
    div_term = np.exp(-np.log(10000.0) * index / dim)
    encoding = np.zeros((length, dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(position * div_term)
    encoding[:, 1::2] = np.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    """Apply rotary position embeddings to ``x`` of shape (..., seq, dim)."""

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def causal_attention(
    query: Any,
    key: Any,
    value: Any,
    scale: float | None = None,
    mask: Any = None,
) -> Any:
    """Scaled dot product attention with a causal mask."""

    head_dim = query.shape[-1]
    factor = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * factor
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, value)


def attention_with_cache(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    cache_position: int,
    scale: float | None = None,
) -> Any:
    """Attention against a KV cache that grows by one position per call.

    When more than one query is processed at once (the prefill) a causal
    mask is applied, because the queries correspond to the last positions
    of the cache.
    """

    head_dim = query.shape[-1]
    factor = scale if scale is not None else 1.0 / np.sqrt(head_dim)
    keys = key_cache[:, :cache_position]
    values = value_cache[:, :cache_position]
    scores = np.matmul(query, np.swapaxes(keys, -1, -2)) * factor

    query_length = query.shape[1]
    if query_length > 1:
        offset = cache_position - query_length
        mask = np.triu(
            np.full((query_length, cache_position), -np.inf, dtype=np.float32),
            k=offset + 1,
        )
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    return np.matmul(weights, values)


class OperatorSet:
    """The operator interface used by the model adapter."""

    name = "numpy"

    def embedding(self, weight: Any, token_ids: Any) -> Any:
        return embedding_lookup(weight, token_ids)

    def linear(self, x: Any, weight: Any, bias: Any = None) -> Any:
        return linear(x, weight, bias)

    def rms_norm(self, x: Any, gain: Any, eps: float = 1e-5) -> Any:
        return rms_norm(x, gain, eps=eps)

    def layer_norm(self, x: Any, gain: Any, bias: Any = None, eps: float = 1e-5) -> Any:
        return layer_norm(x, gain, bias, eps=eps)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return softmax(x, axis=axis)

    def silu(self, x: Any) -> Any:
        return silu(x)

    def gelu(self, x: Any) -> Any:
        return gelu(x)

    def attention_with_cache(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        cache_position: int,
        scale: float | None = None,
    ) -> Any:
        return attention_with_cache(query, key_cache, value_cache, cache_position, scale)


class NumpyOperators(OperatorSet):
    """The reference implementation, written with numpy only."""

    name = "numpy"
