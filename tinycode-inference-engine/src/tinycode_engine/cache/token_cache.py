"""Tokenisation cache.

The cache key contains the model identity and the tokenizer fingerprint so
that tokens are never reused across incompatible tokenizers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .cache_policy import CachePolicy
from .eviction import LruCache

__all__ = ["TokenCache", "cache_key"]


def cache_key(model_id: str, model_version: str, fingerprint: str, text: str) -> str:
    """Build the cache key of a tokenised text."""

    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"{model_id}|{model_version}|{fingerprint}|{digest}"


@dataclass
class TokenCache:
    """Caches the token ids of prompts."""

    policy: CachePolicy = None  # type: ignore[assignment]
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = CachePolicy()
        self._entries = LruCache(self.policy.token_cache_entries, self.policy.eviction)

    def get(self, model_id: str, model_version: str, fingerprint: str, text: str) -> list[int] | None:
        if not self.policy.enabled:
            return None
        value = self._entries.get(cache_key(model_id, model_version, fingerprint, text))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(
        self,
        model_id: str,
        model_version: str,
        fingerprint: str,
        text: str,
        token_ids: list[int],
    ) -> None:
        if not self.policy.enabled:
            return
        self._entries.put(cache_key(model_id, model_version, fingerprint, text), list(token_ids))

    def clear(self) -> None:
        self._entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 6),
            "entries": len(self._entries),
        }
