"""OpenAI compatible error payloads."""

from __future__ import annotations

from typing import Any

from .openai_compat import ERROR_TYPES

__all__ = ["APIError", "error_payload", "error_response"]


class APIError(Exception):
    """An error that is rendered as an OpenAI compatible error body."""

    def __init__(
        self,
        message: str,
        error_type: str = "invalid_request_error",
        status_code: int | None = None,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code or ERROR_TYPES.get(error_type, 400)
        self.param = param
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return error_payload(
            self.message, self.error_type, param=self.param, code=self.code
        )


def error_payload(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI compatible error body."""

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def error_response(
    message: str,
    error_type: str = "invalid_request_error",
    status_code: int | None = None,
    param: str | None = None,
    code: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Return the ``(body, status_code)`` pair of an error response."""

    return (
        error_payload(message, error_type, param=param, code=code),
        status_code or ERROR_TYPES.get(error_type, 400),
    )
