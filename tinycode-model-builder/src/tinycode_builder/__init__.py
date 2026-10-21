"""TinyCode Model Builder.

Builds, trains, evaluates, checkpoints and exports the TinyCode causal
language model together with its optional trainable adaptive-budget
controller.

This package intentionally contains no serving runtime.  The separate
``tinycode-inference-engine`` repository consumes the artifacts produced
here through a versioned manifest contract.
"""

from __future__ import annotations

__all__ = ["__version__", "CONTRACT_VERSION", "CONTEXT_LENGTH"]

__version__ = "0.1.0"

#: Version of the shared artifact contract this builder emits.
CONTRACT_VERSION = "1.0.0"

#: The permanently hardcoded context length.  See
#: ``tinycode_builder.configuration.immutable_constraints``.
CONTEXT_LENGTH = 1024
