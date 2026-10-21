"""Reproducibility helpers: seeding and RNG state capture/restore."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["seed_everything", "capture_rng_state", "ReproducibilityState"]


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (when installed) PyTorch RNGs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch optional at import time
        pass


@dataclass
class ReproducibilityState:
    """Snapshot of the RNG state required to resume training bit-for-bit."""

    seed: int = 0
    python_state: Any = field(default=None)
    numpy_state: Any = field(default=None)
    torch_state: Any = field(default=None)
    torch_cuda_state: Any = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "python_state": self.python_state,
            "numpy_state": self.numpy_state,
            "torch_state": self.torch_state,
            "torch_cuda_state": self.torch_cuda_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReproducibilityState":
        return cls(
            seed=int(payload.get("seed", 0)),
            python_state=payload.get("python_state"),
            numpy_state=payload.get("numpy_state"),
            torch_state=payload.get("torch_state"),
            torch_cuda_state=payload.get("torch_cuda_state"),
        )


def capture_rng_state(seed: int = 0) -> ReproducibilityState:
    """Capture the current RNG state of Python, NumPy and PyTorch."""

    state = ReproducibilityState(seed=seed)
    state.python_state = random.getstate()
    try:
        import numpy as np

        state.numpy_state = np.random.get_state()
    except Exception:  # pragma: no cover
        state.numpy_state = None
    try:
        import torch

        state.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            state.torch_cuda_state = torch.cuda.get_rng_state_all()
    except Exception:  # pragma: no cover
        state.torch_state = None
    return state
