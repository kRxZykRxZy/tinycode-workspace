"""Default configuration values for the model builder.

These defaults are the single source of truth used by the interactive
session, the YAML files under ``config/`` and the tests.  YAML files may
override any of these values *except* the context length.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = ["DEFAULT_CONFIG", "default_config", "merge_defaults", "copy_defaults"]

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "tinycode",
        "version": "0.1.0",
        "runs_dir": "runs",
        "artifacts_dir": "artifacts",
        "data_dir": "data",
        "checkpoint_dir": "checkpoints",
        "report_dir": "reports",
    },
    "model": {
        "parameter_target": 10_000_000,
        "vocab_size": 98_304,
        "vocab_preference": [98_304, 49_152, 32_768, 16_384, 8_192],
        "architecture_id": "tinycode-decoder-v1",
        "max_layers": 64,
        "min_head_dim": 16,
        "preferred_ff_ratio": 4.0,
        "activation": "silu",
    },
    "training": {
        "target_tokens": 200_000_000,
        "batch_size": 8,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3.0e-4,
        "min_learning_rate": 3.0e-5,
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "precision": "auto",
        "seed": 1337,
        "log_every": 20,
        "eval_every": 500,
        "checkpoint_every": 1_000,
        "max_hours": 0.0,
        "val_split": 0.005,
    },
    "dataset": {
        "source": "the-stack",
        "repo_id": "bigcode/the-stack-smol",
        "fallback_repo_ids": ["bigcode/the-stack-smol", "bigcode/the-stack-dedup"],
        "subset": "default",
        "languages": [],
        "streaming": True,
        "max_documents": 0,
        "min_chars": 64,
        "max_chars": 200_000,
        "dedup": True,
        "dedup_hash_bits": 48,
        "normalize": True,
        "shard_size_mb": 512,
        "num_workers": 4,
    },
    "tokenizer": {
        "vocab_size": 98_304,
        "model_type": "bpe",
        "byte_fallback": True,
        "special_tokens": [
            "<|endoftext|>",
            "<|pad|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|code|>",
            "<|endofprompt|>",
        ],
    },
    "optimization": {
        "memory_high_watermark": 0.75,
        "memory_caution_watermark": 0.90,
        "memory_critical_watermark": 0.95,
        "target_micro_batch_tokens": 4096,
        "min_micro_batch": 1,
        "max_micro_batch": 64,
        "reduce_loader_workers": True,
    },
    "checkpoint": {
        "keep_last": 3,
        "keep_every": 10,
        "verify_integrity": True,
        "atomic_write": True,
    },
    "resource_policy": {
        "cpu_memory_high": 0.75,
        "cpu_memory_caution": 0.90,
        "cpu_memory_critical": 0.95,
        "gpu_memory_high": 0.75,
        "gpu_memory_caution": 0.90,
        "gpu_memory_critical": 0.95,
        "disk_free_min_mb": 2048,
        "poll_interval_seconds": 5.0,
    },
    "evaluation": {
        "perplexity_documents": 256,
        "code_task_prompts": 32,
        "report_dir": "reports",
        "include_perplexity": True,
        "include_code_tasks": True,
    },
    "export": {
        "formats": ["pytorch", "safetensors"],
        "include_tokenizer": True,
        "include_controller": True,
        "dtype": "float32",
        "quantize": False,
        "quantization_format": "int8_dynamic",
        "gguf": False,
    },
    "dashboard": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8765,
        "auth_required": True,
        "auth_token": None,
        "cloudflare_tunnel": False,
        "refresh_seconds": 5,
    },
    "budget_controller": {
        "hidden_sizes": [64, 32],
        "learning_rate": 1.0e-3,
        "epochs": 50,
        "feature_schema": "budget_features.v1",
        "enabled": True,
    },
}


def default_config() -> dict[str, Any]:
    """Return a deep copy of the default configuration."""

    return deepcopy(DEFAULT_CONFIG)


def copy_defaults() -> dict[str, Any]:
    """Alias of :func:`default_config`."""

    return default_config()


def merge_defaults(
    base: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (defaults when ``base`` is None)."""

    result = deepcopy(base) if base is not None else default_config()
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_defaults(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
