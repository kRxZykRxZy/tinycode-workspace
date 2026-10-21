"""Causal multi-head self attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import TransformerConfig
from .normalization import build_norm

__all__ = ["CausalSelfAttention", "TransformerBlock", "build_attention"]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self attention with separate Q/K/V projections."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.config.d_model)
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm2 = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)
        self.feed_forward = nn.Linear(config.d_model, config.d_ff, bias=config.ff_bias)
        self.feed_forward_out = nn.Linear(config.d_ff, config.d_model, bias=config.ff_bias)
        self.activation = config.activation

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unknown activation: {self.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        hidden = self._activate(self.feed_forward(self.norm2(x)))
        x = x + self.feed_forward_out(hidden)
        return x


def build_attention(config: TransformerConfig) -> TransformerBlock:
    """Instantiate a single transformer block."""

    return TransformerBlock(config)
