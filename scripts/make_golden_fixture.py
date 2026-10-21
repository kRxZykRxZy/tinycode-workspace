#!/usr/bin/env python
"""Create the golden artifact that both repositories use.

Usage:
    python scripts/make_golden_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = WORKSPACE_ROOT / "contracts"
GOLDEN = CONTRACTS / "golden" / "tinycode-golden"

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

CONTEXT_LENGTH = 1024


def build_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260101)
    vocab_size = len(VOCAB)
    d_model = 16
    n_layers = 2
    n_heads = 4
    d_ff = 32

    weights: dict[str, np.ndarray] = {
        "embeddings.token.weight": rng.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32),
        "embeddings.position.weight": rng.normal(
            0, 0.02, (CONTEXT_LENGTH, d_model)
        ).astype(np.float32),
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
        weights[f"{prefix}.feed_forward.weight"] = rng.normal(
            0, 0.02, (d_ff, d_model)
        ).astype(np.float32)
        weights[f"{prefix}.feed_forward.bias"] = np.zeros(d_ff, dtype=np.float32)
        weights[f"{prefix}.feed_forward_out.weight"] = rng.normal(
            0, 0.02, (d_model, d_ff)
        ).astype(np.float32)
        weights[f"{prefix}.feed_forward_out.bias"] = np.zeros(d_model, dtype=np.float32)
    weights["lm_head.lm_head.weight"] = rng.normal(
        0, 0.02, (vocab_size, d_model)
    ).astype(np.float32)
    return weights


def write_tokenizer(directory: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {token: index for index, token in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = directory / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from safetensors.numpy import save_file

    GOLDEN.mkdir(parents=True, exist_ok=True)
    weights = build_weights()
    save_file(weights, str(GOLDEN / "model.safetensors"))
    tokenizer_path = write_tokenizer(GOLDEN)
    (GOLDEN / "tokenizer_config.json").write_text(
        json.dumps({"vocab_size": len(VOCAB), "model_type": "wordlevel"}, indent=2) + "\n",
        encoding="utf-8",
    )

    parameter_count = int(sum(array.size for array in weights.values()))
    manifest = {
        "artifact_format_version": "1.0.0",
        "created_at": 0.0,
        "model": {
            "name": "tinycode-golden",
            "version": "0.1.0",
            "architecture_id": "tinycode-decoder-v1",
        },
        "architecture": {
            "vocab_size": len(VOCAB),
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 32,
            "context_length": CONTEXT_LENGTH,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "activation": "silu",
            "position_encoding": "learned",
            "qkv_bias": True,
            "ff_bias": True,
            "head_bias": False,
            "tie_embeddings": False,
            "architecture_id": "tinycode-decoder-v1",
        },
        "parameters": {
            "requested_target": parameter_count,
            "actual_trainable": parameter_count,
            "verified": True,
            "breakdown": {},
        },
        "tokenizer": {
            "fingerprint": "sha256:" + file_digest(tokenizer_path),
            "vocab_size": len(VOCAB),
            "files": {"tokenizer.json": {"sha256": file_digest(tokenizer_path)}},
        },
        "weights": {
            "format": "safetensors",
            "file": "model.safetensors",
            "sha256": file_digest(GOLDEN / "model.safetensors"),
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
        "export": {"tool": "scripts/make_golden_fixture.py"},
        "compatibility": {
            "min_engine_version": "0.1.0",
            "min_contract_version": "1.0.0",
            "supported_backends": ["cpu"],
        },
        "files": {
            "model.safetensors": {
                "sha256": file_digest(GOLDEN / "model.safetensors"),
                "size": (GOLDEN / "model.safetensors").stat().st_size,
            },
            "tokenizer.json": {
                "sha256": file_digest(tokenizer_path),
                "size": tokenizer_path.stat().st_size,
            },
        },
    }
    (GOLDEN / "artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"golden fixture written to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
