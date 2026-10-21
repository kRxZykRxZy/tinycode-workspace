"""Capabilities of the CPU backend."""

from __future__ import annotations

from ..interface import BackendCapabilities
from .vectorization import VectorizationInfo, detect_vectorization

__all__ = ["cpu_capabilities"]


def cpu_capabilities() -> BackendCapabilities:
    """Return the capabilities of the CPU backend."""

    vectorization: VectorizationInfo = detect_vectorization()
    notes = [f"SIMD: {', '.join(vectorization.features) or 'none detected'}"]
    if vectorization.has_neon:
        notes.append("NEON is available on this CPU.")
    return BackendCapabilities(
        name="cpu",
        supported_operators=frozenset(
            {
                "embedding",
                "linear",
                "layernorm",
                "rmsnorm",
                "causal_attention",
                "softmax",
                "silu",
                "gelu",
                "sampling",
            }
        ),
        supported_precisions=frozenset({"float32"}),
        supports_batching=False,
        supports_kv_cache=True,
        device_memory_bytes=0,
        is_accelerated=False,
        execution_provider="numpy",
        notes="; ".join(notes),
    )
