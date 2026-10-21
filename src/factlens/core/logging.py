"""Logging configuration built on loguru."""

from __future__ import annotations

import sys

from loguru import logger

from factlens.core.settings import Settings

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
_configured = False


def configure_logging(settings: Settings) -> None:
    """Configure console (and optional file) logging. Safe to call more than once."""
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level.upper(), format=_LOG_FORMAT)
    if settings.log_file is not None:
        logger.add(
            settings.log_file,
            level=settings.log_level.upper(),
            format=_LOG_FORMAT,
            rotation="10 MB",
            retention="30 days",
        )
    _configured = True
