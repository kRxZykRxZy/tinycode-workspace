"""Precision handling: autocast contexts and gradient scaling."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["PrecisionPlan", "resolve_precision", "build_scaler", "autocast_context"]


@dataclass
class PrecisionPlan:
    """The precision that will be used for training."""

    dtype: str = "float32"
    use_amp: bool = False
    use_grad_scaler: bool = False
    device: str = "cpu"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "use_amp": self.use_amp,
            "use_grad_scaler": self.use_grad_scaler,
            "device": self.device,
            "reason": self.reason,
        }


def resolve_precision(requested: str, device: str, cuda_supports_bf16: bool = False) -> PrecisionPlan:
    """Choose the precision used for training.

    ``"auto"`` selects bfloat16 on capable CUDA devices, float16 otherwise
    and float32 on CPU.
    """

    requested = (requested or "auto").lower()
    if requested == "auto":
        if device.startswith("cuda"):
            if cuda_supports_bf16:
                return PrecisionPlan("bfloat16", True, False, device, "auto-selected bfloat16")
            return PrecisionPlan("float16", True, True, device, "auto-selected float16")
        return PrecisionPlan("float32", False, False, device, "CPU training uses float32")
    if requested in {"float32", "fp32"}:
        return PrecisionPlan("float32", False, False, device, "explicit float32")
    if requested in {"float16", "fp16"}:
        if not device.startswith("cuda"):
            return PrecisionPlan("float32", False, False, device, "float16 requires CUDA")
        return PrecisionPlan("float16", True, True, device, "explicit float16")
    if requested in {"bfloat16", "bf16"}:
        if not device.startswith("cuda"):
            return PrecisionPlan("float32", False, False, device, "bfloat16 requires CUDA")
        if not cuda_supports_bf16:
            return PrecisionPlan("float16", True, True, device, "bfloat16 unsupported; using float16")
        return PrecisionPlan("bfloat16", True, False, device, "explicit bfloat16")
    raise ValueError(f"Unknown precision: {requested!r}")


def build_scaler(plan: PrecisionPlan) -> Any:
    """Create a gradient scaler when the precision plan requires one."""

    if not plan.use_grad_scaler:
        return None
    try:
        import torch

        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            return torch.cuda.amp.GradScaler()
    except Exception:  # pragma: no cover
        return None


def autocast_context(plan: PrecisionPlan) -> Any:
    """Return an autocast context manager for the plan (or a no-op)."""

    if not plan.use_amp:
        return contextlib.nullcontext()
    try:
        import torch

        dtype = torch.bfloat16 if plan.dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    except Exception:  # pragma: no cover
        return contextlib.nullcontext()


def autocast_dtype(plan: PrecisionPlan) -> Any:
    """Return the torch dtype used by the plan."""

    import torch

    if plan.dtype == "bfloat16":
        return torch.bfloat16
    if plan.dtype == "float16":
        return torch.float16
    return torch.float32


def iter_autocast(plan: PrecisionPlan) -> Iterator[Any]:  # pragma: no cover - helper
    """Yield the autocast context (kept for symmetry with other helpers)."""

    yield autocast_context(plan)
