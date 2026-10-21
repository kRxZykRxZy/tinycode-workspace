"""Output head for the numpy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["project_logits"]


def project_logits(hidden: Any, weight: Any, bias: Any = None) -> Any:
    """Project hidden states onto the vocabulary."""

    logits = np.matmul(hidden, weight.T)
    if bias is not None:
        logits = logits + bias
    return logits.astype(np.float32, copy=False)
