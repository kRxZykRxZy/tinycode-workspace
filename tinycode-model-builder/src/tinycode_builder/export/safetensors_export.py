"""Safetensors export.

Safetensors stores plain tensors without pickle, which makes the artifact
safe to load from an untrusted process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["export_safetensors", "load_safetensors"]


def export_safetensors(
    model: Any,
    output_dir: str | Path,
    filename: str = "model.safetensors",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export the model weights as a safetensors file."""

    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "The 'safetensors' package is required for safetensors export."
        ) from exc

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename

    state = model.state_dict()
    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    save_file(tensors, str(path), metadata=metadata or {"format": "pt"})

    from ..checkpoint.integrity import file_digest

    return {
        "format": "safetensors",
        "file": path.name,
        "path": str(path),
        "sha256": file_digest(path),
        "size": path.stat().st_size,
        "dtype": str(next(iter(tensors.values())).dtype).replace("torch.", "")
        if tensors
        else "float32",
    }


def load_safetensors(path: str | Path) -> dict[str, Any]:
    """Load tensors from a safetensors file."""

    from safetensors.torch import load_file

    return load_file(str(path))
