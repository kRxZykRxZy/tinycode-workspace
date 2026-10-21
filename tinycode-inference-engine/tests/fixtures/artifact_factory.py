"""Helpers that build the deterministic test artifacts.

The functions live in their own module so that the test modules can
import them without relying on the pytest import mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CONTEXT_LENGTH = 1024

VOCAB = [
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|code|>",
    "<|endofprompt|>",
    "<unk>",
    "def",
    "return",
    "function",
    "a",
    "b",
    "add",
    "sum",
    "the",
    "and",
    "code",
    "python",
    "write",
    "hello",
    "world",
    "test",
    "print",
    "int",
    "str",
    "=",
    "(",
    ")",
    ":",
    "\n",
]


def build_weights(
    vocab_size: int = len(VOCAB),
    d_model: int = 16,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ff: int = 32,
    context_length: int = CONTEXT_LENGTH,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Create a small, deterministic set of weights."""

    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {
        "embeddings.token.weight": rng.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32),
        "embeddings.position.weight": rng.normal(0, 0.02, (context_length, d_model)).astype(np.float32),
        "final_norm.weight": np.ones(d_model, dtype=np.float32),
    }
    for layer in range(n_layers):
        prefix = f"layers.{layer}"
        weights[f"{prefix}.norm1.weight"] = np.ones(d_model, dtype=np.float32)
        weights[f"{prefix}.norm2.weight"] = np.ones(d_model, dtype=np.float32)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.attention.{projection}.weight"] = rng.normal(
                0, 0.02, (d_model, d_model)
            ).astype(np.float32)
            weights[f"{prefix}.attention.{projection}.bias"] = np.zeros(d_model, dtype=np.float32)
        weights[f"{prefix}.feed_forward.weight"] = rng.normal(0, 0.02, (d_ff, d_model)).astype(
            np.float32
        )
        weights[f"{prefix}.feed_forward.bias"] = np.zeros(d_ff, dtype=np.float32)
        weights[f"{prefix}.feed_forward_out.weight"] = rng.normal(0, 0.02, (d_model, d_ff)).astype(
            np.float32
        )
        weights[f"{prefix}.feed_forward_out.bias"] = np.zeros(d_model, dtype=np.float32)
    weights["lm_head.lm_head.weight"] = rng.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32)
    return weights


def write_tokenizer(directory: Path) -> Path:
    """Write a tiny deterministic tokenizer."""

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {token: index for index, token in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = directory / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def build_artifact(
    directory: Path,
    vocab_size: int = len(VOCAB),
    d_model: int = 16,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ff: int = 32,
    context_length: int = CONTEXT_LENGTH,
    with_controller: bool = False,
) -> Path:
    """Create a complete, loadable artifact directory."""

    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    weights = build_weights(vocab_size, d_model, n_layers, n_heads, d_ff, context_length)
    save_file(weights, str(directory / "model.safetensors"))
    tokenizer_path = write_tokenizer(directory)

    from tinycode_engine.artifacts.integrity import file_digest

    architecture = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "context_length": context_length,
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "activation": "silu",
        "position_encoding": "learned",
        "qkv_bias": True,
        "ff_bias": True,
        "head_bias": False,
        "tie_embeddings": False,
        "architecture_id": "tinycode-decoder-v1",
    }
    parameter_count = int(sum(array.size for array in weights.values()))

    manifest = {
        "artifact_format_version": "1.0.0",
        "created_at": 0.0,
        "model": {
            "name": "tinycode-test",
            "version": "0.1.0",
            "architecture_id": "tinycode-decoder-v1",
        },
        "architecture": architecture,
        "parameters": {
            "requested_target": parameter_count,
            "actual_trainable": parameter_count,
            "verified": True,
            "breakdown": {},
        },
        "tokenizer": {
            "fingerprint": "sha256:" + file_digest(tokenizer_path),
            "vocab_size": vocab_size,
            "files": {"tokenizer.json": {"sha256": file_digest(tokenizer_path)}},
        },
        "weights": {
            "format": "safetensors",
            "file": "model.safetensors",
            "sha256": file_digest(directory / "model.safetensors"),
            "dtype": "float32",
        },
        "precisions": ["float32"],
        "required_operators": [
            "embedding",
            "linear",
            "rmsnorm",
            "causal_attention",
            "softmax",
            "silu",
            "sampling",
        ],
        "controller": {"present": False},
        "training": {},
        "export": {},
        "compatibility": {
            "min_engine_version": "0.1.0",
            "min_contract_version": "1.0.0",
            "supported_backends": ["cpu"],
        },
        "files": {
            "model.safetensors": {
                "sha256": file_digest(directory / "model.safetensors"),
                "size": (directory / "model.safetensors").stat().st_size,
            },
            "tokenizer.json": {
                "sha256": file_digest(tokenizer_path),
                "size": tokenizer_path.stat().st_size,
            },
        },
    }
    (directory / "artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory
