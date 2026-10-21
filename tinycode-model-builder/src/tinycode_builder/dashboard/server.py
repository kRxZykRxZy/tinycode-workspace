"""The training dashboard HTTP server.

The dashboard is authenticated independently of the tunnel URL and never
exposes arbitrary shell execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..telemetry.counters import Counters
from .authentication import DashboardAuth
from .controls import ControlBridge
from .metrics_api import build_metrics_payload, build_prometheus_payload
from .state import DashboardState

__all__ = ["create_dashboard_app", "DashboardServer"]


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def create_dashboard_app(
    state: DashboardState | None = None,
    controls: ControlBridge | None = None,
    auth: DashboardAuth | None = None,
    counters: Counters | None = None,
) -> FastAPI:
    """Create the FastAPI application that serves the dashboard."""

    state = state or DashboardState()
    controls = controls or ControlBridge(state)
    auth = auth or DashboardAuth(None, required=False)

    app = FastAPI(title="TinyCode Training Dashboard", version="0.1.0")
    app.state.dashboard_state = state
    app.state.dashboard_controls = controls
    app.state.dashboard_auth = auth
    app.state.dashboard_counters = counters or Counters()

    def _require_auth(authorization: str | None, token: str | None) -> None:
        if not auth.enabled:
            return
        if auth.check_header(authorization) or (token is not None and auth.check(token)):
            return
        raise HTTPException(status_code=401, detail="Invalid dashboard credentials.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "running": state.get().running}

    @app.get("/api/state")
    def api_state(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _require_auth(authorization, token)
        return JSONResponse(state.get().to_dict())

    @app.get("/api/metrics")
    def api_metrics(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _require_auth(authorization, token)
        return JSONResponse(build_metrics_payload(state, app.state.dashboard_counters))

    @app.get("/metrics")
    def prometheus_metrics(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> PlainTextResponse:
        _require_auth(authorization, token)
        return PlainTextResponse(
            build_prometheus_payload(state, app.state.dashboard_counters),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/api/logs")
    def api_logs(
        limit: int = Query(default=100, ge=1, le=1000),
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _require_auth(authorization, token)
        return JSONResponse({"logs": state.recent_logs(limit)})

    @app.post("/api/control/{action}")
    def api_control(
        action: str,
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _require_auth(authorization, token)
        actions = {
            "checkpoint": controls.request_checkpoint,
            "pause": controls.request_pause,
            "resume": controls.request_resume,
            "stop": controls.request_stop,
        }
        handler = actions.get(action)
        if handler is None:
            raise HTTPException(status_code=404, detail=f"Unknown control action: {action}")
        result = handler()
        return JSONResponse(result.to_dict(), status_code=200 if result.ok else 500)

    @app.get("/", response_class=HTMLResponse)
    def index(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> HTMLResponse:
        if auth.enabled:
            _require_auth(authorization, token)
        index_file = _static_dir() / "index.html"
        if not index_file.is_file():  # pragma: no cover - packaging issue
            return HTMLResponse("<html><body><h1>Dashboard assets are missing.</h1></body></html>")
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    static_path = _static_dir()
    if static_path.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app


class DashboardServer:
    """Runs the dashboard with uvicorn."""

    def __init__(
        self,
        app: FastAPI,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self._server: Any = None

    def start_background(self) -> Any:
        """Start uvicorn in a background thread."""

        import threading

        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        thread = threading.Thread(target=self._server.run, name="tinycode-dashboard", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:  # pragma: no cover - blocking
        """Run the dashboard in the foreground."""

        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
