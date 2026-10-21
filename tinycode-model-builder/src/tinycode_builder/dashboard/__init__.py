"""Training dashboard for the model builder."""

from __future__ import annotations

from .authentication import DashboardAuth, generate_token, read_or_create_token, verify_token
from .controls import ControlBridge, ControlResult
from .metrics_api import build_metrics_payload, build_prometheus_payload
from .server import DashboardServer, create_dashboard_app
from .state import DashboardState, TrainingSnapshot
from .tunnel import CloudflareTunnel, is_cloudflared_available

__all__ = [
    "CloudflareTunnel",
    "ControlBridge",
    "ControlResult",
    "DashboardAuth",
    "DashboardServer",
    "DashboardState",
    "TrainingSnapshot",
    "build_metrics_payload",
    "build_prometheus_payload",
    "create_dashboard_app",
    "generate_token",
    "is_cloudflared_available",
    "read_or_create_token",
    "verify_token",
]
