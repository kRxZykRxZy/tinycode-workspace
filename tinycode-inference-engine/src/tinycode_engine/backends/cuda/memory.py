"""GPU memory accounting for the CUDA backend."""

from __future__ import annotations

from typing import Any

__all__ = ["gpu_memory_snapshot", "free_memory_bytes", "total_memory_bytes"]


def _torch():
    try:
        import torch

        return torch
    except Exception:  # pragma: no cover - torch optional
        return None


def free_memory_bytes(device_index: int = 0) -> int:
    """Return the free memory of a CUDA device."""

    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free)
    except Exception:  # pragma: no cover
        return 0


def total_memory_bytes(device_index: int = 0) -> int:
    """Return the total memory of a CUDA device."""

    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.get_device_properties(device_index).total_memory)
    except Exception:  # pragma: no cover
        return 0


def gpu_memory_snapshot(device_index: int = 0) -> dict[str, Any]:
    """Return a snapshot of the GPU memory usage."""

    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {
            "available": False,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
        }
    try:
        free, total = torch.cuda.mem_get_info(device_index)
        return {
            "available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}
