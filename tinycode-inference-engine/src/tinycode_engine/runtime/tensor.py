"""Tensor helpers.

The engine works on numpy arrays.  This module provides the small amount
of tensor bookkeeping that the runtime needs: dtype handling, contiguous
copies and shape validation.  It deliberately does not pretend to be a
universal tensor abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = ["TensorInfo", "as_float32", "ensure_contiguous", "shape_of", "bytes_of", "flatten_weights"]


@dataclass
class TensorInfo:
    """Metadata about a tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
        }


def shape_of(array: Any) -> tuple[int, ...]:
    """Return the shape of an array-like object."""

    return tuple(int(dim) for dim in array.shape)


def bytes_of(array: Any) -> int:
    """Return the number of bytes occupied by an array."""

    return int(array.nbytes)


def ensure_contiguous(array: Any) -> Any:
    """Return a C-contiguous copy of ``array`` when necessary."""

    if not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array)
    return array


def as_float32(array: Any) -> Any:
    """Return a float32 view/copy of ``array``."""

    if array.dtype == np.float32:
        return array
    return array.astype(np.float32)


def flatten_weights(weights: dict[str, Any]) -> dict[str, TensorInfo]:
    """Return metadata for every tensor in a weight dictionary."""

    info: dict[str, TensorInfo] = {}
    for name, array in weights.items():
        if not isinstance(array, np.ndarray):
            continue
        info[name] = TensorInfo(
            name=name,
            shape=shape_of(array),
            dtype=str(array.dtype),
            size_bytes=bytes_of(array),
        )
    return info


def stack(values: Iterable[float]) -> Any:
    """Stack a sequence of scalars into a float32 array."""

    return np.asarray(list(values), dtype=np.float32)
