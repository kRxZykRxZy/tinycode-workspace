"""Normalisation helpers for the numpy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["apply_norm", "rms_norm", "layer_norm"]


def rms_norm(x: Any, gain: Any, eps: float = 1e-5) -> Any:
    """Root mean square normalisation."""

    variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(variance + eps))) * gain


def layer_norm(x: Any, gain: Any, bias: Any = None, eps: float = 1e-5) -> Any:
    """Layer normalisation."""

    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(variance + eps)
    normed = normed * gain
    if bias is not None:
        normed = normed + bias
    return normed


def apply_norm(x: Any, norm_type: str, gain: Any, bias: Any = None, eps: float = 1e-5) -> Any:
    """Dispatch to the requested normalisation."""

    if norm_type == "rmsnorm":
        return rms_norm(x, gain, eps=eps)
    if norm_type == "layernorm":
        return layer_norm(x, gain, bias, eps=eps)
    raise ValueError(f"Unsupported norm type: {norm_type!r}")
