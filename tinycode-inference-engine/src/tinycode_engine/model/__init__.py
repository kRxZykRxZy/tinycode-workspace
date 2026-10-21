"""Model handling inside the inference engine."""

from __future__ import annotations

from .interface import ModelCapabilities, ModelInterface
from .loader import load_safetensors, load_torch_state_dict, load_weights
from .model_adapter import TinyCodeModel
from .tokenizer_adapter import TokenizerAdapter, chat_prompt

__all__ = [
    "ModelCapabilities",
    "ModelInterface",
    "TinyCodeModel",
    "TokenizerAdapter",
    "chat_prompt",
    "load_safetensors",
    "load_torch_state_dict",
    "load_weights",
]
