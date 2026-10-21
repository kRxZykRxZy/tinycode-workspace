"""The numpy implementation of the TinyCode model.

The forward pass is implemented here rather than delegated to a training
framework, which keeps the inference engine independent from the model
builder and makes the execution path explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cache.kv_cache import KVCache
from ..runtime.operators import NumpyOperators, OperatorSet
from ..utils.errors import ArtifactError, ModelError
from .attention import merge_heads, split_heads
from .interface import ModelCapabilities, ModelInterface
from .output_head import project_logits
from .positional_encoding import position_embeddings

__all__ = ["TinyCodeModel"]


class TinyCodeModel(ModelInterface):
    """Executes the TinyCode decoder architecture."""

    name = "tinycode"

    def __init__(
        self,
        config: dict[str, Any],
        weights: dict[str, Any],
        operators: OperatorSet | None = None,
    ) -> None:
        self.config = dict(config)
        self.weights = weights
        self.ops = operators or NumpyOperators()
        self._validate_weights()

        self.vocab_size = int(self.config["vocab_size"])
        self.d_model = int(self.config["d_model"])
        self.n_layers = int(self.config["n_layers"])
        self.n_heads = int(self.config["n_heads"])
        self.d_ff = int(self.config["d_ff"])
        self._context_length = int(self.config.get("context_length", 1024))
        self.head_dim = self.d_model // self.n_heads
        self.norm_type = str(self.config.get("norm_type", "rmsnorm"))
        self.norm_eps = float(self.config.get("norm_eps", 1e-5))
        self.activation = str(self.config.get("activation", "silu"))
        self.position_encoding = str(self.config.get("position_encoding", "learned"))
        self.qkv_bias = bool(self.config.get("qkv_bias", True))
        self.ff_bias = bool(self.config.get("ff_bias", True))
        self.tie_embeddings = bool(self.config.get("tie_embeddings", False))
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self._token_embedding = self._require("embeddings.token.weight")
        self._position_weight = self.weights.get("embeddings.position.weight")
        if self.position_encoding == "sinusoidal":
            self._position_weight = None
        self._output_weight = self.weights.get("lm_head.lm_head.weight")
        self._output_bias = self.weights.get("lm_head.lm_head.bias")
        if self._output_weight is None and self.tie_embeddings:
            self._output_weight = self._token_embedding

        self._cache: KVCache | None = None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _require(self, name: str) -> Any:
        if name not in self.weights:
            raise ArtifactError(f"The artifact is missing the weight '{name}'.")
        return self.weights[name]

    def _validate_weights(self) -> None:
        required = ["embeddings.token.weight"]
        for layer in range(int(self.config.get("n_layers", 0))):
            required.extend(
                [
                    f"layers.{layer}.attention.q_proj.weight",
                    f"layers.{layer}.attention.k_proj.weight",
                    f"layers.{layer}.attention.v_proj.weight",
                    f"layers.{layer}.attention.o_proj.weight",
                    f"layers.{layer}.feed_forward.weight",
                    f"layers.{layer}.feed_forward_out.weight",
                ]
            )
        missing = [name for name in required if name not in self.weights]
        if missing:
            raise ArtifactError(
                "The artifact does not contain the expected weights: " + ", ".join(missing[:8])
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def context_length(self) -> int:
        """The fixed context length of the model."""

        return self._context_length

    def _allocate_cache(self) -> KVCache:
        return KVCache(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            context_length=self.context_length,
        )

    def _embed(self, token_ids: Any) -> Any:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ModelError("Cannot run the model on an empty input.")
        if ids.max() >= self.vocab_size or ids.min() < 0:
            raise ModelError("Token id outside of the model vocabulary.")
        hidden = self._token_embedding[ids]
        positions = position_embeddings(
            self.position_encoding,
            ids.size,
            self.d_model,
            self._position_weight,
        )
        return hidden + positions

    def _activate(self, x: Any) -> Any:
        if self.activation == "silu":
            return self.ops.silu(x)
        if self.activation == "gelu":
            return self.ops.gelu(x)
        raise ModelError(f"Unsupported activation: {self.activation!r}")

    def _norm(self, x: Any, gain: Any, bias: Any) -> Any:
        if self.norm_type == "rmsnorm":
            return self.ops.rms_norm(x, gain, self.norm_eps)
        return self.ops.layer_norm(x, gain, bias, self.norm_eps)

    def _linear(self, x: Any, prefix: str) -> Any:
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights.get(f"{prefix}.bias")
        return self.ops.linear(x, weight, bias)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def forward(self, input_ids: Any, cache: Any = None) -> Any:
        """Run the model over ``input_ids`` and return the last logits.

        Args:
            input_ids: a 1-D array of token ids.
            cache: an optional :class:`KVCache`; a new one is created when
                it is omitted.
        """

        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if cache is None:
            cache = self._allocate_cache()
            self._cache = cache
        if not cache.can_append(ids.size):
            raise ModelError(
                f"The prompt of {ids.size} tokens does not fit into the fixed "
                f"{self.context_length} token context (cache holds {cache.length})."
            )

        hidden = self._embed(ids)
        seq = hidden.shape[0]

        for layer in range(self.n_layers):
            prefix = f"layers.{layer}"
            norm1_gain = self.weights[f"{prefix}.norm1.weight"]
            norm1_bias = self.weights.get(f"{prefix}.norm1.bias")
            normed = self._norm(hidden, norm1_gain, norm1_bias)

            query = split_heads(self._linear(normed, f"{prefix}.attention.q_proj"), self.n_heads)
            key = split_heads(self._linear(normed, f"{prefix}.attention.k_proj"), self.n_heads)
            value = split_heads(self._linear(normed, f"{prefix}.attention.v_proj"), self.n_heads)

            if cache.length:
                cache.keys[layer][:, cache.length : cache.length + seq, :] = key
                cache.values[layer][:, cache.length : cache.length + seq, :] = value
            else:
                cache.keys[layer][:, :seq, :] = key
                cache.values[layer][:, :seq, :] = value

            total = cache.length + seq
            context = self.ops.attention_with_cache(
                query, cache.keys[layer], cache.values[layer], total, self.scale
            )
            attended = self._linear(merge_heads(context), f"{prefix}.attention.o_proj")
            hidden = hidden + attended

            norm2_gain = self.weights[f"{prefix}.norm2.weight"]
            norm2_bias = self.weights.get(f"{prefix}.norm2.bias")
            normed = self._norm(hidden, norm2_gain, norm2_bias)
            inner = self._linear(normed, f"{prefix}.feed_forward")
            hidden = hidden + self._linear(self._activate(inner), f"{prefix}.feed_forward_out")

        cache.advance(seq)

        final_gain = self.weights["final_norm.weight"]
        final_bias = self.weights.get("final_norm.bias")
        hidden = self._norm(hidden, final_gain, final_bias)

        last = hidden[-1]
        if self._output_weight is None:
            raise ArtifactError("The artifact does not contain an output head.")
        return project_logits(last, self._output_weight, self._output_bias)

    def decode_step(self, token_id: int, cache: KVCache) -> Any:
        """Run a single decoding step against an existing cache."""

        return self.forward(np.asarray([token_id], dtype=np.int64), cache=cache)

    def reset_cache(self) -> None:
        """Drop the internal cache."""

        self._cache = None

    def capabilities(self) -> ModelCapabilities:
        total = int(sum(array.size for array in self.weights.values()))
        return ModelCapabilities(
            context_length=self._context_length,
            vocab_size=self.vocab_size,
            parameter_count=total,
            supports_kv_cache=True,
            supports_batching=False,
            name=self.name,
        )
