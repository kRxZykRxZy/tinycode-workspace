"""Precision handling for the runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Precision", "resolve_precision", "cast_weights", "dtype_size"]

DTYPES: dict[str, Any] = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float32,  # numpy has no bfloat16; kept in float32
    "int8": np.int8,
    "int4": np.uint8,
}


class Precision:
    """Helpers for the supported precisions."""

    @staticmethod
    def numpy_dtype(name: str) -> Any:
        return DTYPES.get((name or "float32").lower(), np.float32)

    @staticmethod
    def is_supported(name: str) -> bool:
        return (name or "").lower() in DTYPES


def dtype_size(name: str) -> int:
    """Return the number of bytes per element of a precision."""

    return int(np.dtype(Precision.numpy_dtype(name)).itemsize)


def resolve_precision(requested: str, supported: set[str] | frozenset[str]) -> str:
    """Pick a precision that the backend actually supports."""

    requested = (requested or "float32").lower()
    if requested in supported:
        return requested
    for fallback in ("float32", "float16"):
        if fallback in supported:
            return fallback
    raise ValueError(
        f"No supported precision in {sorted(supported)}; requested {requested!r}."
    )


def cast_weights(weights: dict[str, Any], precision: str) -> dict[str, Any]:
    """Cast the weights to the requested precision."""

    dtype = Precision.numpy_dtype(precision)
    casted: dict[str, Any] = {}
    for name, array in weights.items():
        if isinstance(array, np.ndarray) and array.dtype != dtype and array.dtype.kind == "f":
            casted[name] = array.astype(dtype)
        else:
            casted[name] = array
    return casted
