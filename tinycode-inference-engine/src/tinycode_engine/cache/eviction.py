"""LRU eviction for the caches."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Generic, TypeVar

__all__ = ["LruCache", "EvictionPolicy"]

T = TypeVar("T")


class EvictionPolicy:
    """The eviction strategies supported by the caches."""

    LRU = "lru"
    FIFO = "fifo"

    @staticmethod
    def validate(name: str) -> str:
        if name not in (EvictionPolicy.LRU, EvictionPolicy.FIFO):
            raise ValueError(f"Unknown eviction policy: {name!r}")
        return name


class LruCache(Generic[T]):
    """A bounded cache with LRU eviction."""

    def __init__(self, max_entries: int = 32, policy: str = EvictionPolicy.LRU) -> None:
        self.max_entries = max(1, int(max_entries))
        self.policy = EvictionPolicy.validate(policy)
        self._entries: "OrderedDict[str, T]" = OrderedDict()

    def get(self, key: str) -> T | None:
        """Return an entry and mark it as recently used."""

        if key not in self._entries:
            return None
        if self.policy == EvictionPolicy.LRU:
            self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: T) -> None:
        """Insert an entry, evicting the least recently used one if needed."""

        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        while len(self._entries) > self.max_entries:
            evicted, _value = self._entries.popitem(last=False)
            self.on_evict(evicted)

    def on_evict(self, key: str) -> None:
        """Hook that is called when an entry is evicted."""

    def remove(self, key: str) -> bool:
        return self._entries.pop(key, None) is not None

    def items(self) -> list[tuple[str, T]]:
        """Return the entries from the least to the most recently used."""

        return list(self._entries.items())

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._entries), "max_entries": self.max_entries, "policy": self.policy}
