"""Middleware of the API layer."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..protocol.errors import APIError
from ..security.request_limits import RequestLimits, check_request_size

__all__ = ["RequestContextMiddleware", "install_middleware"]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Adds a request id, the duration and the request size limit."""

    def __init__(self, app: Any, limits: RequestLimits | None = None) -> None:
        super().__init__(app)
        self.limits = limits or RequestLimits()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            check_request_size(request.headers.get("content-length"), self.limits)
            response = await call_next(request)
        except APIError as exc:
            from fastapi.responses import JSONResponse

            response = JSONResponse(status_code=exc.status_code, content=exc.to_dict())
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        response.headers["x-request-duration-ms"] = f"{duration_ms:.3f}"
        return response


def install_middleware(app: Any, limits: RequestLimits | None = None) -> None:
    """Install the middleware of the application."""

    app.add_middleware(RequestContextMiddleware, limits=limits or RequestLimits())
