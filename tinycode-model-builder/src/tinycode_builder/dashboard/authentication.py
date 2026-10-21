"""Authentication for the training dashboard.

The dashboard is always authenticated, independently of how it is
exposed.  Tokens are compared in constant time and never written to the
logs.
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

__all__ = ["DashboardAuth", "generate_token", "read_or_create_token", "verify_token"]


def generate_token(length: int = 32) -> str:
    """Generate a random, URL safe dashboard token."""

    return secrets.token_urlsafe(length)


def read_or_create_token(path: str | Path) -> str:
    """Read the dashboard token, creating one when it does not exist yet."""

    token_path = Path(path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = generate_token()
    token_path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(token_path, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return token


def verify_token(candidate: str | None, expected: str | None) -> bool:
    """Compare two tokens in constant time."""

    if not candidate or not expected:
        return False
    return hmac.compare_digest(str(candidate), str(expected))


class DashboardAuth:
    """Holds the expected dashboard token."""

    def __init__(self, token: str | None, required: bool = True) -> None:
        self.token = token
        self.required = required

    @property
    def enabled(self) -> bool:
        return bool(self.token) and self.required

    def check(self, candidate: str | None) -> bool:
        """Return whether ``candidate`` is authorised."""

        if not self.enabled:
            return True
        return verify_token(candidate, self.token)

    def check_header(self, authorization: str | None) -> bool:
        """Check an ``Authorization: Bearer <token>`` header."""

        if not self.enabled:
            return True
        if not authorization:
            return False
        parts = authorization.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return verify_token(parts[1].strip(), self.token)
