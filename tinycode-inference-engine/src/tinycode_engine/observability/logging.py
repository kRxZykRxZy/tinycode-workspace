"""Logging configuration of the inference engine."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger."""

    global _CONFIGURED
    logger = logging.getLogger("tinycode")
    if _CONFIGURED and not force:
        return logger

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "tinycode") -> logging.Logger:
    """Return a namespaced logger."""

    if not _CONFIGURED:
        configure_logging()
    if name.startswith("tinycode"):
        return logging.getLogger(name)
    return logging.getLogger(f"tinycode.{name}")
