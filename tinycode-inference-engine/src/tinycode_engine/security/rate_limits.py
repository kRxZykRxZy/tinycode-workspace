"""Rate limiting.

A simple, per-process token bucket.  The limit is configurable and can be
disabled; the server never rate limits the health endpoints.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..protocol.errors import APIError

__all__ = ["RateLimiter", "RateLimitResult"]


@dataclass
class RateLimitResult:
    """The outcome of a rate limit check."""

    allowed: bool = True
    remaining: int = 0
    retry_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "retry_after": round(self.retry_after, 3),
        }


@dataclass
class RateLimiter:
    """A per-client token bucket."""

    requests_per_minute: int = 0
    burst: int = 0
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, client_id: str = "default") -> RateLimitResult:
        """Check whether a request may proceed."""

        if not self.enabled:
            return RateLimitResult(allowed=True, remaining=-1)

        now = time.monotonic()
        capacity = self.burst or max(1, self.requests_per_minute)
        refill_per_second = self.requests_per_minute / 60.0
        with self._lock:
            tokens, last = self._buckets.get(client_id, (float(capacity), now))
            tokens = min(float(capacity), tokens + (now - last) * refill_per_second)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[client_id] = (tokens, now)
                return RateLimitResult(allowed=True, remaining=int(tokens))
            retry_after = (1.0 - tokens) / refill_per_second if refill_per_second else 1.0
            self._buckets[client_id] = (tokens, now)
            return RateLimitResult(allowed=False, remaining=0, retry_after=retry_after)

    def enforce(self, client_id: str = "default") -> None:
        """Raise an ``APIError`` when the limit is exceeded."""

        result = self.check(client_id)
        if not result.allowed:
            raise APIError(
                "Rate limit exceeded. Retry later.",
                "rate_limit_error",
                status_code=429,
            )

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
