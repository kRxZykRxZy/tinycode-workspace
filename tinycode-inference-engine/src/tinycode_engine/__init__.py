"""TinyCode Inference Engine.

A custom inference runtime with an OpenAI compatible HTTP API.  The engine
loads TinyCode artifacts (as produced by the separate model builder
repository) and serves them locally.

The engine does not depend on the model builder and does not import any
training code.
"""

from __future__ import annotations

__all__ = ["__version__", "CONTRACT_VERSION", "CONTEXT_LENGTH"]

__version__ = "0.1.0"

#: Version of the artifact contract this engine understands.
CONTRACT_VERSION = "1.0.0"

#: The permanently hardcoded context length.
CONTEXT_LENGTH = 1024
