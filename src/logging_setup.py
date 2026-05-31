"""
Structured logging setup for production.
JSON format for log aggregation pipelines (Datadog, Splunk, CloudWatch).
Text format for local development.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from pythonjsonlogger import jsonlogger  # python-json-logger


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_dir: Optional[Path] = None,
    name: str = "fcps",  # Financial Crisis Prediction System
) -> logging.Logger:
    """
    Configure root logger with console + optional rotating file handler.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        fmt: "json" for structured production logs, "text" for human-readable.
        log_dir: If given, writes rotating file logs here.
        name: Logger name (defaults to package root).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    # ── Console handler ───────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    if fmt == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(
            log_dir / "fcps.log",
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=10,
            encoding="utf-8",
        )
        fh.setLevel(log_level)
        fh.setFormatter(
            jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("yfinance", "urllib3", "matplotlib", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'fcps' namespace."""
    return logging.getLogger(f"fcps.{name}")
