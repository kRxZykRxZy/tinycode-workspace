"""Torch based operators and model execution for the CUDA backend.

The implementation mirrors the numpy reference exactly so that both paths
produce the same logits (within the usual floating point tolerance).
"""

from __future__ import annotations

from typing import Any

from ...runtime.operators import OperatorSet

__all__ = ["TorchOperators", "TorchTinyCodeModel"]


class TorchOperators(OperatorSet):
    """Operators implemented with PyTorch tensors."""

    name = "cuda"

    def embedding(self, weight: Any, token_ids: Any) -> Any:
        import torch.nn.functional as F

        return F.embedding(token_ids, weight)

    def linear(self, x: Any, weight: Any, bias: Any = None) -> Any:
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def rms_norm(self, x: Any, gain: Any, eps: float = 1e-5) -> Any:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * gain

    def layer_norm(self, x: Any, gain: Any, bias: Any = None, eps: float = 1e-5) -> Any:
        import torch.nn.functional as F

        return F.layer_norm(x, (x.shape[-1],), gain, bias, eps)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        import torch

        return torch.softmax(x, dim=axis)

    def silu(self, x: Any) -> Any:
        import torch.nn.functional as F

        return F.silu(x)

    def gelu(self, x: Any) -> Any:
        import torch.nn.functional as F

        return F.gelu(x, approximate="tanh")

    def attention_with_cache(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        cache_position: int,
        scale: float | None = None,
    ) -> Any:
        import torch.nn.functional as F

        keys = key_cache[:, :cache_position]
        values = value_cache[:, :cache_position]
        return F.scaled_dot_product_attention(query, keys, values, is_causal=False, scale=scale)


class TorchTinyCodeModel:
    """A torch implementation of the TinyCode decoder."""

    def __init__(self, config: dict[str, Any], weights: dict[str, Any], device: Any = None) -> None:
        import torch

        self.torch = torch
        self.config = dict(config)
        self.device = device or torch.device("cuda")
        self.vocab_size = int(self.config["vocab_size"])
        self.d_model = int(self.config["d_model"])
        self.n_layers = int(self.config["n_layers"])
        self.n_heads = int(self.config["n_heads"])
        self.d_ff = int(self.config["d_ff"])
        self.context_length = int(self.config.get("context_length", 1024))
        self.head_dim = self.d_model // self.n_heads
        self.norm_type = str(self.config.get("norm_type", "rmsnorm"))
        self.norm_eps = float(self.config.get("norm_eps", 1e-5))
        self.activation = str(self.config.get("activation", "silu"))
        self.position_encoding = str(self.config.get("position_encoding", "learned"))
        self.scale = 1.0 / (self.head_dim**0.5)

        import numpy as np

        self.weights = {
            name: torch.as_tensor(np.ascontiguousarray(array)).to(self.device)
            for name, array in weights.items()
            if isinstance(array, np.ndarray)
        }

    def _linear(self, x: Any, prefix: str) -> Any:
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights.get(f"{prefix}.bias")
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def _norm(self, x: Any, gain: Any, bias: Any) -> Any:
        if self.norm_type == "rmsnorm":
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(variance + self.norm_eps) * gain
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), gain, bias, self.norm_eps)

    def forward(self, input_ids: Any, cache: Any = None) -> Any:
        """Run the model and return the logits of the last position."""

        torch = self.torch
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(-1)
        hidden = self.weights["embeddings.token.weight"][ids]
        if self.position_encoding == "learned":
            positions = self.weights["embeddings.position.weight"][: ids.numel()]
        else:
            positions = _sinusoidal(ids.numel(), self.d_model, self.device)
        hidden = hidden + positions

        seq = hidden.shape[0]
        for layer in range(self.n_layers):
            prefix = f"layers.{layer}"
            normed = self._norm(
                hidden,
                self.weights[f"{prefix}.norm1.weight"],
                self.weights.get(f"{prefix}.norm1.bias"),
            )
            query = self._split(self._linear(normed, f"{prefix}.attention.q_proj"))
            key = self._split(self._linear(normed, f"{prefix}.attention.k_proj"))
            value = self._split(self._linear(normed, f"{prefix}.attention.v_proj"))

            if cache is not None:
                start = cache.length
                cache.keys[layer][:, start : start + seq, :] = key
                cache.values[layer][:, start : start + seq, :] = value
                total = start + seq
            else:
                total = seq

            context = torch.nn.functional.scaled_dot_product_attention(
                query, cache.keys[layer][:, :total], cache.values[layer][:, :total], is_causal=False,
                scale=self.scale,
            )
            attended = self._linear(self._merge(context), f"{prefix}.attention.o_proj")
            hidden = hidden + attended

            normed = self._norm(
                hidden,
                self.weights[f"{prefix}.norm2.weight"],
                self.weights.get(f"{prefix}.norm2.bias"),
            )
            inner = self._linear(normed, f"{prefix}.feed_forward")
            if self.activation == "silu":
                inner = torch.nn.functional.silu(inner)
            else:
                inner = torch.nn.functional.gelu(inner, approximate="tanh")
            hidden = hidden + self._linear(inner, f"{prefix}.feed_forward_out")

        if cache is not None:
            cache.advance(seq)

        hidden = self._norm(
            hidden,
            self.weights["final_norm.weight"],
            self.weights.get("final_norm.bias"),
        )
        output_weight = self.weights.get("lm_head.lm_head.weight")
        if output_weight is None:
            output_weight = self.weights["embeddings.token.weight"]
        output_bias = self.weights.get("lm_head.lm_head.bias")
        logits = hidden[-1] @ output_weight.T
        if output_bias is not None:
            logits = logits + output_bias
        return logits

    def _split(self, x: Any) -> Any:
        seq = x.shape[0]
        return x.reshape(seq, self.n_heads, self.head_dim).transpose(0, 1)

    def _merge(self, x: Any) -> Any:
        heads, seq, head_dim = x.shape
        return x.transpose(0, 1).reshape(seq, heads * head_dim)


def _sinusoidal(length: int, dim: int, device: Any) -> Any:
    import math

    import torch

    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    index = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    div_term = torch.exp(-math.log(10000.0) * index / dim)
    encoding = torch.zeros(length, dim, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding
