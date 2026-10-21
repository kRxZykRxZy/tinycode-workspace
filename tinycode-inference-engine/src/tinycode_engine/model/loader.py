"""Loading of model weights into numpy arrays.

Only formats that the engine can actually read are supported: safetensors
(``model.safetensors``) and the PyTorch state dictionary written by the
builder (``model.pt``).  PyTorch is optional: when it is not installed the
safetensors format is the only supported one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.errors import ArtifactError

__all__ = ["load_weights", "load_safetensors", "load_torch_state_dict"]


def load_safetensors(path: str | Path) -> dict[str, Any]:
    """Load a safetensors file into numpy arrays."""

    try:
        from safetensors.numpy import load_file
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ArtifactError(f"The 'safetensors' package is required: {exc}") from exc

    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Weight file not found: {path}")
    try:
        return dict(load_file(str(path)))
    except Exception as exc:
        raise ArtifactError(f"Failed to read {path}: {exc}") from exc


def load_torch_state_dict(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint into numpy arrays."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ArtifactError(
            "Reading 'model.pt' requires PyTorch; export safetensors instead."
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Weight file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactError(f"Failed to read {path}: {exc}") from exc

    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path} does not contain a state dictionary.")

    weights: dict[str, Any] = {}
    for name, tensor in payload.items():
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        weights[name] = tensor
    return weights


def load_weights(directory: str | Path, preferred: str | None = None) -> tuple[dict[str, Any], str]:
    """Load the weights of an artifact directory.

    Returns:
        ``(weights, format_name)``.
    """

    directory = Path(directory)
    candidates: list[tuple[Path, str]] = []
    safetensors = directory / "model.safetensors"
    torch_file = directory / "model.pt"
    if safetensors.is_file():
        candidates.append((safetensors, "safetensors"))
    if torch_file.is_file():
        candidates.append((torch_file, "pytorch"))
    if not candidates:
        raise ArtifactError(
            f"No supported weight file found in {directory} "
            "(expected model.safetensors or model.pt)."
        )

    order = {"safetensors": 0, "pytorch": 1}
    candidates.sort(key=lambda item: order[item[1]])
    if preferred:
        for path, name in candidates:
            if name == preferred:
                candidates.remove((path, name))
                candidates.insert(0, (path, name))
                break

    path, name = candidates[0]
    if name == "safetensors":
        return load_safetensors(path), name
    return load_torch_state_dict(path), name
