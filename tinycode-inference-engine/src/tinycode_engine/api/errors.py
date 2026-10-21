"""Error handling of the API layer."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..protocol.errors import APIError
from ..utils.errors import EngineError, RequestError

__all__ = ["install_error_handlers"]


def install_error_handlers(app: FastAPI) -> None:
    """Install the OpenAI compatible error handlers."""

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestError)
    async def _request_error(request: Request, exc: RequestError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", [])[1:]) or None
        message = first.get("msg", "the request is invalid")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": field,
                    "code": None,
                }
            },
        )

    @app.exception_handler(EngineError)
    async def _engine_error(request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "The server could not complete the request.",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )
