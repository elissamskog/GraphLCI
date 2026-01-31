"""
Centralized logging configuration for KnowledgeAudit.
Location: config/logging_config.py
Owner: ROOT Instance

Provides ``setup_logging`` to configure root logger with console + optional
file handlers, and ``log_file_from_output_path`` to derive a ``.log`` path
from an output file path.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"

_configured = False


def setup_logging(
    log_file: str | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Configure the root logger with console and optional file handlers.

    Idempotent: subsequent calls are no-ops once handlers are attached.

    Parameters
    ----------
    log_file:
        Path to the log file.  ``None`` means console-only.
    console_level:
        Logging level for console output (default ``"INFO"``).
    file_level:
        Logging level for the file handler (default ``"DEBUG"``).
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress verbose third-party HTTP logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_file_from_output_path(output_path: str) -> str:
    """Derive a ``.log`` path from an output file path.

    Example::

        >>> log_file_from_output_path("output_5claims.md")
        'output_5claims.log'
    """
    base, _ = os.path.splitext(output_path)
    return base + ".log"
