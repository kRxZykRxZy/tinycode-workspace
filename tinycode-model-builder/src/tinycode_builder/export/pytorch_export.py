"""PyTorch export: weights-only files plus a resumable checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["export_pytorch", "export_resumable_checkpoint"]


def export_pytorch(
    model: Any,
    output_dir: str | Path,
    filename: str = "model.pt",
) -> dict[str, Any]:
    """Export the model weights as a plain ``torch.save`` state dictionary."""

    import torch

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    payload = {
        "state_dict": model.state_dict(),
        "architecture": model.config.to_dict() if hasattr(model, "config") else {},
    }
    torch.save(payload, path)
    from ..checkpoint.integrity import file_digest

    return {
        "format": "pytorch",
        "file": path.name,
        "path": str(path),
        "sha256": file_digest(path),
        "size": path.stat().st_size,
        "dtype": "float32",
    }


def export_resumable_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: str | Path,
    step: int,
    committed_training_tokens: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a checkpoint that can be used to resume training."""

    from ..checkpoint.integrity import file_digest
    from ..checkpoint.serializer import serialize_checkpoint

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"checkpoint-{step:08d}.pt"
    payload = {
        "step": step,
        "committed_training_tokens": committed_training_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "architecture": model.config.to_dict() if hasattr(model, "config") else {},
    }
    payload.update(extra or {})
    serialize_checkpoint(payload, path)
    return {
        "format": "pytorch-checkpoint",
        "file": path.name,
        "path": str(path),
        "sha256": file_digest(path),
        "size": path.stat().st_size,
    }
