"""Resource cleanup helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["cleanup_resources", "release_caches"]


def cleanup_resources() -> None:
    """Release the caches and the temporary buffers of the process."""

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_caches(engine: Any) -> None:
    """Release the caches of the engine."""

    token_cache = getattr(engine, "token_cache", None)
    if token_cache is not None:
        token_cache.clear()
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None:
        for request in list(scheduler.active.values()):
            try:
                request.session.release()
            except Exception:  # pragma: no cover
                continue
        scheduler.active.clear()
