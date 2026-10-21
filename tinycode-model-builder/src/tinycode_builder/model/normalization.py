"""Normalisation layers used by the TinyCode decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "LayerNorm", "build_norm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation with a learned gain.

    Implemented as ``x / sqrt(mean(x^2) + eps) * gain`` which matches the
    reference implementation used by the inference engine.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return f"dim={self.weight.numel()}, eps={self.eps}"


def build_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Instantiate the requested normalisation layer."""

    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm type: {norm_type!r}")


class LayerNorm(nn.LayerNorm):
    """Alias kept so that callers can reference the class directly."""
