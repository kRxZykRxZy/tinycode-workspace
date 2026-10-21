"""Greedy decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["select_greedy"]


def select_greedy(logits: Any) -> int:
    """Return the token with the highest logit."""

    values = np.asarray(logits, dtype=np.float32)
    return int(np.argmax(values))
