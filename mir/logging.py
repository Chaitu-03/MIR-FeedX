"""
mir/logging.py
Structured logging config — Prompt 15.

JSON renderer for production (log_json=True), ConsoleRenderer otherwise.
Call configure_logging() once at process startup (FastAPI lifespan, Celery setup).
"""
from __future__ import annotations

import logging
import sys

import structlog

from mir.config import settings


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    lvl = (level or settings.log_level).upper()
    json_logs = settings.log_json if json_logs is None else json_logs

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=lvl,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, lvl, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()
