"""The runtime layer of the inference engine.

``EngineConfig`` and ``InferenceEngine`` are exposed lazily: the engine
module imports the artifact loader, which imports the model adapter, which
imports the operators of this package.  Importing them eagerly would
create a cycle.
"""

from __future__ import annotations

from typing import Any

from .errors import BackendError, EngineError, KernelUnavailableError, ModelError
from .execution_graph import ExecutionGraph, GraphNode, build_execution_graph
from .kernel_dispatch import KernelDispatcher, NativeLibrary
from .memory import Arena, MemoryAccountant, MemorySnapshot
from .operators import NumpyOperators, OperatorSet
from .precision import Precision, cast_weights, resolve_precision
from .quantization import dequantize_state_dict
from .synchronization import CountingEvent, SynchronizationContext
from .tensor import TensorInfo, flatten_weights

__all__ = [
    "Arena",
    "BackendError",
    "CountingEvent",
    "EngineConfig",
    "EngineError",
    "ExecutionGraph",
    "GraphNode",
    "InferenceEngine",
    "KernelDispatcher",
    "KernelUnavailableError",
    "MemoryAccountant",
    "MemorySnapshot",
    "ModelError",
    "NativeLibrary",
    "NumpyOperators",
    "OperatorSet",
    "Precision",
    "SynchronizationContext",
    "TensorInfo",
    "build_execution_graph",
    "cast_weights",
    "dequantize_state_dict",
    "flatten_weights",
    "resolve_precision",
]


def __getattr__(name: str) -> Any:
    """Expose the engine classes without importing them eagerly."""

    if name == "EngineConfig":
        from .engine import EngineConfig

        return EngineConfig
    if name == "InferenceEngine":
        from .engine import InferenceEngine

        return InferenceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
