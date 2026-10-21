"""The lifecycle helpers of the application layer."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

__all__ = ["service_context", "graceful_shutdown"]


@contextlib.contextmanager
def service_context(service: Any) -> Iterator[Any]:
    """Build the service, load the models and release the resources on exit."""

    service.build()
    report = service.start_models()
    if report is not None and report.errors:
        for error in report.errors:
            print(f"error: {error}")
    try:
        yield service
    finally:
        service.shutdown()


def graceful_shutdown(service: Any, timeout: float = 10.0) -> None:
    """Stop the service and release its resources."""

    from ..server.shutdown import shutdown_engine

    engine = getattr(service, "engine", None)
    if engine is not None:
        shutdown_engine(engine, timeout=timeout)
