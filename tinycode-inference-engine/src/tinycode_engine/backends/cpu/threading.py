"""Thread configuration for the CPU backend."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ...utils.platform import cpu_count

__all__ = ["ThreadConfig", "configure_threads", "current_thread_settings"]


@dataclass
class ThreadConfig:
    """How many threads the CPU backend may use."""

    threads: int = 1
    source: str = "default"

    def to_dict(self) -> dict[str, object]:
        return {"threads": self.threads, "source": self.source}


def configure_threads(requested: int | None = None) -> ThreadConfig:
    """Configure the BLAS/OpenMP thread count.

    The environment variables are only set when the caller explicitly asks
    for a thread count; otherwise the existing settings are respected.
    """

    if requested is None:
        return ThreadConfig(threads=cpu_count(), source="system")
    threads = max(1, int(requested))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(threads)
    try:
        import numpy as np

        # numpy exposes the OpenBLAS thread count on most platforms.
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            threadpool_limits(threads)
        except Exception:
            pass
    except Exception:  # pragma: no cover
        pass
    return ThreadConfig(threads=threads, source="configured")


def current_thread_settings() -> dict[str, object]:
    """Return the thread related environment settings."""

    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "cpu_count": cpu_count(),
    }
