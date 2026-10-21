"""Integration test: a real (tiny) training run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinycode_builder.checkpoint.manager import CheckpointManager
from tinycode_builder.configuration.defaults import default_config
from tinycode_builder.model.architecture import TransformerConfig
from tinycode_builder.training.trainer import Trainer


def _write_shard(path: Path, sequences: int = 4, length: int = 1024) -> Path:
    rng = np.random.default_rng(1234)
    data = rng.integers(0, 200, size=(sequences, length), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    return path


def test_training_performs_real_optimizer_updates(tmp_path: Path) -> None:
    config = default_config()
    config["training"]["target_tokens"] = 8_192
    config["training"]["batch_size"] = 2
    config["training"]["micro_batch_size"] = 1
    config["training"]["log_every"] = 1
    config["training"]["checkpoint_every"] = 1

    model_config = TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        context_length=1024,
        norm_type="rmsnorm",
    )

    train_shard = _write_shard(tmp_path / "train" / "shard-00000.npy")
    validation_shard = _write_shard(tmp_path / "validation" / "shard-00000.npy", sequences=2)

    trainer = Trainer(
        config=config,
        model_config=model_config,
        run_dir=tmp_path / "run",
        train_shards=[train_shard],
        validation_shards=[validation_shard],
        checkpoint_manager=CheckpointManager(tmp_path / "ckpt"),
    )
    trainer.build()

    before = [p.detach().clone() for p in trainer.model.parameters()]
    result = trainer.train(max_steps=2)

    assert result.steps == 2
    assert result.committed_training_tokens > 0

    after = list(trainer.model.parameters())
    changed = any(
        not torch_allclose(old, new) for old, new in zip(before, after)
    )
    assert changed, "the optimizer did not update any parameter"


def torch_allclose(a, b, tolerance: float = 1e-8) -> bool:
    """Small helper that avoids importing torch at module import time."""

    import torch

    return bool(torch.allclose(a, b, atol=tolerance))


def test_smoke_test_reports_success(tmp_path: Path) -> None:
    config = default_config()
    model_config = TransformerConfig(
        vocab_size=128, d_model=16, n_layers=1, n_heads=4, d_ff=32, context_length=1024
    )
    trainer = Trainer(
        config=config,
        model_config=model_config,
        run_dir=tmp_path / "run",
        checkpoint_manager=CheckpointManager(tmp_path / "ckpt"),
    )
    report = trainer.smoke_test()
    assert report["forward_loss"] > 0
    assert report["optimizer_step"] == "ok"
    assert report["checkpoint_roundtrip"] == "ok"
