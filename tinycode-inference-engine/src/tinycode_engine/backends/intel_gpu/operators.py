"""OpenVINO based operators for the Intel graphics backend.

The Intel backend executes an OpenVINO IR model.  The operators of the
TinyCode architecture are therefore provided by the compiled graph; this
module only exposes the small helpers that the runtime needs.
"""

from __future__ import annotations

from typing import Any

from ...runtime.operators import OperatorSet

__all__ = ["OpenVinoOperators"]


class OpenVinoOperators(OperatorSet):
    """Operators backed by an OpenVINO compiled model."""

    name = "intel_gpu"

    def __init__(self, compiled_model: Any = None, tokenizer: Any = None) -> None:
        self.compiled_model = compiled_model
        self.tokenizer = tokenizer

    def supports(self, operator: str) -> bool:
        """Report which operators the compiled graph implements."""

        if self.compiled_model is None:
            return False
        try:
            names = {
                str(node.get_type_name())
                for node in self.compiled_model.get_ops()
            }
        except Exception:  # pragma: no cover - provider dependent
            return False
        mapping = {
            "embedding": "gather",
            "linear": "matmul",
            "layernorm": "normalize",
            "rmsnorm": "normalize",
            "causal_attention": "scaled_dot_product_attention",
            "softmax": "softmax",
            "silu": "silu",
            "gelu": "gelu",
            "sampling": "multinomial",
        }
        target = mapping.get(operator)
        if target is None:
            return False
        return any(target in name for name in names)
