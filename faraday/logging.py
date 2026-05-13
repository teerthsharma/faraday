# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.logging — Structured logging setup for the faraday package.

Uses structlog for structured log output with both console (human-readable)
and JSON (machine-readable) modes.

Usage
-----
    from faraday.logging import get_logger

    log = get_logger(__name__)
    log.info("solving_cavity", geometry=(2.0, 1.0), nx=60, ny=60)
    log.warning("mode_count_low", expected=12, found=3)
    log.debug("iteration", iter=42, delta=1.23e-4)

Console output (default)::

    2026-05-03 20:55:01 [info] solving_cavity geometry=(2.0, 1.0) nx=60 ny=60

JSON output (when FARADAY_LOG_FORMAT=json)::

    {"event": "solving_cavity", "geometry": [2.0, 1.0], "nx": 60, "ny": 60, "level": "info", "timestamp": "..."}
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

# --------------------------------------------------------------------
# Log format selection
# --------------------------------------------------------------------
_FARADAY_LOG_FORMAT = os.environ.get("FARADAY_LOG_FORMAT", "console").lower()
_IS_JSON = _FARADAY_LOG_FORMAT == "json"
_IS_VERBOSE = os.environ.get("FARADAY_VERBOSE", "0") == "1"
_IS_QUIET = os.environ.get("FARADAY_QUIET", "0") == "1"

# --------------------------------------------------------------------
# Configure structlog
# --------------------------------------------------------------------


def _configure_structlog() -> None:
    """Configure structlog processors and loggers."""
    processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if _IS_JSON:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_configure_structlog()

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger for ``name``.

    The logger is configured once at import time via ``_configure_structlog()``.
    All messages are written to stderr and rendered as:

    - JSON (when ``FARADAY_LOG_FORMAT=json``)
    - Console output (otherwise)

    Log methods accept arbitrary keyword arguments that become structured
    fields.  Use ``.error()``, ``.debug()`` etc. with keyword arguments for
    structured fields.

    Example
    -------
    >>> from faraday.logging import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("batch_processed", n=42, elapsed_s=1.23)
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def set_log_level(level: str) -> None:
    """
    Set the minimum log level across all faraday loggers.

    Parameters
    ----------
    level : str
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        Case-insensitive.

    Notes
    -----
    This sets the level on the underlying Python ``logging`` root logger,
    which structlog proxies through.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("faraday").setLevel(numeric)
