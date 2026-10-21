"""API key management and verification.

The server binds to 127.0.0.1 by default and requires an API key.  The
comparison is constant time and the keys are never logged.
"""

from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ApiKeyStore",
    "extract_bearer_token",
    "generate_api_key",
    "load_api_keys",
    "write_api_key",
]


def generate_api_key(length: int = 32) -> str:
    """Generate a random API key."""

    return "tc-" + secrets.token_urlsafe(length)


def extract_bearer_token(authorization: str | None) -> str | None:
    """Extract the token of an ``Authorization: Bearer <token>`` header."""

    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def load_api_keys(
    config_keys: list[str] | None = None, env_name: str = "TINYCODE_API_KEYS"
) -> list[str]:
    """Load the accepted API keys from the configuration and the environment."""

    keys = [str(key).strip() for key in (config_keys or []) if str(key).strip()]
    env_value = os.environ.get(env_name)
    if env_value:
        keys.extend(key.strip() for key in env_value.split(",") if key.strip())
    # Keep the order but remove the duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def write_api_key(path: str | Path, key: str | None = None) -> Path:
    """Write an API key to a file (used for the local development default)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = key or generate_api_key()
    target.write_text(value + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return target


@dataclass
class ApiKeyStore:
    """Holds the accepted API keys."""

    keys: list[str] = field(default_factory=list)
    required: bool = True

    @property
    def enabled(self) -> bool:
        return self.required and bool(self.keys)

    def verify(self, candidate: str | None) -> bool:
        """Verify a candidate key in constant time."""

        if not self.enabled:
            return True
        if not candidate:
            return False
        for key in self.keys:
            if hmac.compare_digest(str(candidate), str(key)):
                return True
        return False

    def authenticate(self, authorization: str | None) -> None:
        """Raise an ``APIError`` when the request is not authorised."""

        if not self.enabled:
            return
        token = extract_bearer_token(authorization)
        if not self.verify(token):
            from ..protocol.errors import APIError

            raise APIError(
                "Invalid API key. Provide the key as 'Authorization: Bearer <key>'.",
                "authentication_error",
                status_code=401,
            )

    @classmethod
    def from_config(cls, payload: dict[str, Any] | None = None) -> "ApiKeyStore":
        """Create the store from the security configuration."""

        payload = dict(payload or {})
        return cls(
            keys=load_api_keys(list(payload.get("api_keys", []) or [])),
            required=bool(payload.get("require_api_key", True)),
        )
