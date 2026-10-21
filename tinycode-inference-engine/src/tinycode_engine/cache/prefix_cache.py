"""Prefix cache.

The prefix cache stores the KV state of a prompt prefix so that repeated
prefixes do not have to be recomputed.  Correctness is preserved by
including the model identity, the model version and the tokenizer
fingerprint in the cache key: state is never reused across incompatible
models or tokenizers.

The identity is also stored *inside* the entry so that a lookup can find
the longest cached prefix of a prompt without hashing every candidate
length.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .cache_policy import CachePolicy
from .eviction import LruCache

__all__ = ["PrefixEntry", "PrefixCache", "prefix_key"]


def prefix_key(
    model_id: str,
    model_version: str,
    fingerprint: str,
    token_ids: list[int],
) -> str:
    """Build the cache key of a token prefix."""

    digest = hashlib.sha256()
    digest.update(f"{model_id}|{model_version}|{fingerprint}|".encode("utf-8"))
    digest.update(",".join(str(int(token)) for token in token_ids).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class PrefixEntry:
    """A cached prefix."""

    token_ids: list[int]
    length: int
    payload: Any = None
    model_id: str = ""
    model_version: str = ""
    fingerprint: str = ""

    def matches(self, prompt_ids: list[int]) -> bool:
        """Return whether the entry is a prefix of ``prompt_ids``."""

        if len(self.token_ids) > len(prompt_ids):
            return False
        return list(prompt_ids[: len(self.token_ids)]) == list(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "tokens": len(self.token_ids),
            "model_id": self.model_id,
            "has_state": self.payload is not None,
        }


@dataclass
class PrefixCache:
    """Caches the KV state of prompt prefixes."""

    policy: CachePolicy = None  # type: ignore[assignment]
    hits: int = 0
    misses: int = 0
    entries: dict[str, PrefixEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = CachePolicy()
        self._entries = LruCache(self.policy.prefix_cache_entries, self.policy.eviction)

    # ------------------------------------------------------------------
    def get(
        self,
        model_id: str,
        model_version: str,
        fingerprint: str,
        token_ids: list[int],
    ) -> PrefixEntry | None:
        if not self.policy.enabled:
            return None
        key = prefix_key(model_id, model_version, fingerprint, token_ids)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(
        self,
        model_id: str,
        model_version: str,
        fingerprint: str,
        token_ids: list[int],
        payload: Any = None,
    ) -> None:
        if not self.policy.enabled:
            return
        if len(token_ids) < self.policy.prefix_cache_min_tokens:
            return
        key = prefix_key(model_id, model_version, fingerprint, token_ids)
        entry = PrefixEntry(
            token_ids=list(token_ids),
            length=len(token_ids),
            payload=payload,
            model_id=model_id,
            model_version=model_version,
            fingerprint=fingerprint,
        )
        self._entries.put(key, entry)

    def find_prefix(
        self,
        model_id: str,
        model_version: str,
        fingerprint: str,
        prompt_ids: list[int],
    ) -> PrefixEntry | None:
        """Return the longest cached prefix of ``prompt_ids``.

        Only the entries of the requested model and tokenizer are
        considered, so the state of an incompatible model is never reused.
        """

        if not self.policy.enabled:
            return None
        best: PrefixEntry | None = None
        for _key, entry in self._entries.items():
            if entry.model_id != model_id:
                continue
            if entry.model_version != model_version:
                continue
            if entry.fingerprint != fingerprint:
                continue
            if not entry.matches(prompt_ids):
                continue
            if best is None or entry.length > best.length:
                best = entry
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
        return best

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
            "max_entries": self.policy.prefix_cache_entries,
        }
