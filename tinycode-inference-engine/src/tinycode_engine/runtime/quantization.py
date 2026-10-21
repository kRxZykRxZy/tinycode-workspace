"""Dequantisation of quantised weights.

The builder can produce int8 and int4 quantised tensors.  The engine
restores them to float32 at load time so that the execution path stays
simple and numerically well defined.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["dequantize_state_dict", "dequantize_int8", "dequantize_int4_group", "is_quantized"]


def is_quantized(name: str) -> bool:
    """Return whether a weight name refers to a quantised tensor."""

    return ".quant" in name or name.endswith("_q")


def dequantize_int8(data: Any, scale: Any, zero_point: Any = None) -> Any:
    """Restore an int8 tensor that was quantised per output channel."""

    values = np.asarray(data, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 1:
        scale = scale[:, None]
    restored = values * scale
    if zero_point is not None:
        restored = restored + np.asarray(zero_point, dtype=np.float32)
    return restored


def dequantize_int4_group(
    data: Any, scale: Any, zero_point: Any, group_size: int = 32
) -> Any:
    """Restore a group-wise int4 tensor."""

    values = np.asarray(data, dtype=np.float32)
    rows, cols = values.shape
    groups = (cols + group_size - 1) // group_size
    padded = values[:, : groups * group_size]
    reshaped = padded.reshape(rows, groups, group_size)
    scale = np.asarray(scale, dtype=np.float32)
    zero_point = np.asarray(zero_point, dtype=np.float32)
    restored = reshaped * scale[:, :, None] + zero_point[:, :, None]
    restored = restored.reshape(rows, groups * group_size)
    return restored[:, :cols]


def dequantize_state_dict(weights: dict[str, Any]) -> dict[str, Any]:
    """Restore every quantised tensor of a weight dictionary."""

    result: dict[str, Any] = {}
    for name, array in weights.items():
        if not is_quantized(name):
            result[name] = array
            continue
        base = name.split(".quant")[0]
        scale = weights.get(f"{base}.scale")
        zero_point = weights.get(f"{base}.zero_point")
        group_size = weights.get(f"{base}.group_size")
        bits = int(weights.get(f"{base}.bits", 8))
        if scale is None:
            result[name] = array
            continue
        if bits == 4:
            restored = dequantize_int4_group(
                array, scale, zero_point if zero_point is not None else 0, int(group_size or 32)
            )
        else:
            restored = dequantize_int8(array, scale, zero_point)
        result[base] = restored
    return result
