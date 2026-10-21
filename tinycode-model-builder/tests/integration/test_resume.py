"""Integration test: resuming a training run restores the state."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinycode_builder.checkpoint.manager import CheckpointManager
from tinycode_builder.checkpoint.resume import extract_resume_state, restore_training_state
from tinycode_builder.configuration.defaults import default_config
from tinycode_builder.model.architecture import TransformerConfig
from tinycode_builder.training.trainer import Trainer


def _write_shard(path: Path, sequences: int = 4) -> Path:
    rng = np.random.default_rng(7)
    data = rng.integers(0, 200, size=(sequences, 1024), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    return path


def test_resume_restores_step_and_weights(tmp_path: Path) -> None:
    config = default_config()
    config["training"]["batch_size"] = 2
    config["training"]["micro_batch_size"] = 1
    config["training"]["target_tokens"] = 100_000

    model_config = TransformerConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, d_ff=64, context_length=1024,
        norm_type="rmsnorm",
    )
    shard = _write_shard(tmp_path / "train" / "shard-00000.npy")
    manager = CheckpointManager(tmp_path / "ckpt")

    trainer = Trainer(
        config=config,
        model_config=model_config,
        run_dir=tmp_path / "run",
        train_shards=[shard],
        checkpoint_manager=manager,
    )
    trainer.build()
    trainer.train(max_steps=2)

    checkpoint = manager.latest()
    assert checkpoint is not None
    payload = manager.load(checkpoint)
    state = extract_resume_state(payload)
    assert state.step == 2
    assert state.committed_training_tokens > 0

    resumed = Trainer(
        config=config,
        model_config=model_config,
        run_dir=tmp_path / "run2",
        train_shards=[shard],
        checkpoint_manager=manager,
    )
    resumed.build()
    restored = restore_training_state(resumed, state)
    assert restored.get("model") is True
    assert restored.get("optimizer") is True
    assert resumed.step == 2
    assert resumed.committed_training_tokens == state.committed_training_tokens
