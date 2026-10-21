"""Cache configuration and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .eviction import EvictionPolicy

__all__ = ["CachePolicy"]


@dataclass
class CachePolicy:
    """The configuration of the runtime caches."""

    enabled: bool = True
    prefix_cache_entries: int = 8
    prefix_cache_min_tokens: int = 16
    token_cache_entries: int = 64
    eviction: str = EvictionPolicy.LRU
    #: The cache key always includes these components so that state is
    #: never reused across incompatible models or tokenizers.
    key_components: tuple[str, ...] = ("model_id", "model_version", "tokenizer_fingerprint")

    def validate(self) -> "CachePolicy":
        self.eviction = EvictionPolicy.validate(self.eviction)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "prefix_cache_entries": self.prefix_cache_entries,
            "prefix_cache_min_tokens": self.prefix_cache_min_tokens,
            "token_cache_entries": self.token_cache_entries,
            "eviction": self.eviction,
            "key_components": list(self.key_components),
        }
