"""Tests for checkpointing, integrity and retention."""

from __future__ import annotations

from pathlib import Path

from tinycode_builder.checkpoint.integrity import file_digest, verify_files
from tinycode_builder.checkpoint.manager import CheckpointManager
from tinycode_builder.checkpoint.manifest import CheckpointManifest
from tinycode_builder.checkpoint.resume import extract_resume_state
from tinycode_builder.checkpoint.retention import RetentionPolicy, apply_retention, select_for_deletion
from tinycode_builder.checkpoint.serializer import deserialize_checkpoint, serialize_checkpoint


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ckpt")
    manifest = CheckpointManifest(step=7, committed_training_tokens=1024)
    info = manager.save({"model_state": {"w": 1}, "step": 7}, step=7, manifest=manifest)

    assert info.path.is_file()
    payload = manager.load(info.path)
    assert payload["model_state"] == {"w": 1}

    state = extract_resume_state(payload)
    assert state.step == 7


def test_previous_checkpoint_is_preserved(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ckpt", retention=RetentionPolicy(keep_last=5))
    manager.save({"step": 1}, step=1)
    manager.save({"step": 2}, step=2)
    assert (tmp_path / "ckpt" / "checkpoint-00000001.pt").is_file()
    assert (tmp_path / "ckpt" / "checkpoint-00000002.pt").is_file()


def test_integrity_verification_detects_tampering(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ckpt")
    info = manager.save({"step": 1}, step=1)
    digest = file_digest(info.path)
    assert verify_files(tmp_path / "ckpt", {info.path.name: digest}).ok

    info.path.write_bytes(b"corrupted")
    result = verify_files(tmp_path / "ckpt", {info.path.name: digest})
    assert not result.ok
    assert result.mismatched


def test_retention_removes_old_checkpoints(tmp_path: Path) -> None:
    directory = tmp_path / "ckpt"
    directory.mkdir()
    for step in range(1, 6):
        (directory / f"checkpoint-{step:08d}.pt").write_bytes(b"x")
    policy = RetentionPolicy(keep_last=2, keep_every=0)
    removed = apply_retention(directory, policy)
    assert len(removed) == 3
    assert (directory / "checkpoint-00000005.pt").is_file()
    assert (directory / "checkpoint-00000004.pt").is_file()


def test_retention_selection_is_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / f"checkpoint-{step:08d}.pt" for step in range(1, 6)]
    policy = RetentionPolicy(keep_last=2, keep_every=0)
    assert select_for_deletion(paths, policy) == paths[:3]


def test_serializer_writes_atomically(tmp_path: Path) -> None:
    target = tmp_path / "state.pt"
    serialize_checkpoint({"a": 1}, target)
    assert deserialize_checkpoint(target) == {"a": 1}
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
