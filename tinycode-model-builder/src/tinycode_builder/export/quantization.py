"""Quantisation utilities.

Two formats are implemented and verified by round trip:

``int8_dynamic``
    Per output channel symmetric int8 quantisation of 2-D weights.

``int4_group``
    Group-wise (group size 32) affine int4 quantisation of 2-D weights.

Both formats store the original tensor name, the scale and the zero point
so that the inference engine can dequantise deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "QuantizationResult",
    "quantize_tensor",
    "dequantize_tensor",
    "quantization_error",
    "quantize_state_dict",
    "dequantize_state_dict",
    "SUPPORTED_FORMATS",
]

SUPPORTED_FORMATS = ("int8_dynamic", "int4_group")

GROUP_SIZE = 32


@dataclass
class QuantizationResult:
    """A quantised tensor plus the metadata needed to restore it."""

    name: str
    data: Any
    scale: Any
    zero_point: Any
    shape: tuple[int, ...]
    bits: int
    group_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "bits": self.bits,
            "group_size": self.group_size,
            "scale_shape": list(self.scale.shape) if hasattr(self.scale, "shape") else None,
        }


def quantize_tensor(name: str, tensor: Any, bits: int = 8, group_size: int = GROUP_SIZE) -> QuantizationResult:
    """Quantise a 2-D tensor."""

    import torch

    if tensor.dim() != 2:
        raise ValueError(f"Only 2-D tensors can be quantised, got {tensor.dim()} dimensions")
    if bits == 8:
        scale = tensor.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        quantised = torch.round(tensor.detach() / scale).clamp(-127, 127)
        return QuantizationResult(
            name=name,
            data=quantised.to(torch.int8),
            scale=scale.to(torch.float32),
            zero_point=torch.zeros(scale.shape, dtype=torch.float32),
            shape=tuple(tensor.shape),
            bits=8,
            group_size=0,
        )
    if bits == 4:
        rows, cols = tensor.shape
        groups = (cols + group_size - 1) // group_size
        padded = tensor.detach()
        if cols % group_size != 0:
            padding = torch.zeros(
                rows, groups * group_size - cols, dtype=padded.dtype, device=padded.device
            )
            padded = torch.cat([padded, padding], dim=1)
        reshaped = padded.reshape(rows, groups, group_size)
        minimum = reshaped.amin(dim=-1, keepdim=True)
        maximum = reshaped.amax(dim=-1, keepdim=True)
        scale = ((maximum - minimum) / 15.0).clamp(min=1e-8)
        quantised = torch.round((reshaped - minimum) / scale).clamp(0, 15)
        return QuantizationResult(
            name=name,
            data=quantised.reshape(rows, groups * group_size).to(torch.uint8),
            scale=scale.squeeze(-1).to(torch.float32),
            zero_point=minimum.squeeze(-1).to(torch.float32),
            shape=(rows, cols),
            bits=4,
            group_size=group_size,
        )
    raise ValueError(f"Unsupported bit width: {bits}")


def dequantize_tensor(result: QuantizationResult) -> Any:
    """Restore a quantised tensor."""

    import torch

    if result.bits == 8:
        return result.data.to(torch.float32) * result.scale + result.zero_point
    rows, cols = result.shape
    groups = (cols + result.group_size - 1) // result.group_size
    quantised = result.data.reshape(rows, groups, result.group_size).to(torch.float32)
    restored = quantised * result.scale.unsqueeze(-1) + result.zero_point.unsqueeze(-1)
    restored = restored.reshape(rows, groups * result.group_size)
    return restored[:, :cols]


def quantization_error(original: Any, restored: Any) -> float:
    """Return the mean absolute error between two tensors."""

    import torch

    difference = (original.detach().float() - restored.detach().float()).abs()
    return float(difference.mean().item()) if difference.numel() else 0.0


def quantize_state_dict(
    state_dict: dict[str, Any], bits: int = 8, skip: tuple[str, ...] = ("norm", "bias")
) -> dict[str, QuantizationResult]:
    """Quantise every 2-D tensor of a state dictionary."""

    results: dict[str, QuantizationResult] = {}
    for name, tensor in state_dict.items():
        if any(marker in name for marker in skip):
            continue
        if not hasattr(tensor, "dim") or tensor.dim() != 2:
            continue
        results[name] = quantize_tensor(name, tensor, bits=bits)
    return results


def dequantize_state_dict(
    results: dict[str, QuantizationResult], state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild a state dictionary from quantised tensors."""

    restored: dict[str, Any] = {}
    for name, tensor in state_dict.items():
        if name in results:
            restored[name] = dequantize_tensor(results[name])
        else:
            restored[name] = tensor
    return restored
