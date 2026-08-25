"""Console and file logging shared by C3 entry points."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_LOG_DIR = Path("outputs/logs/c3")
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return an idempotent logger writing to console and ``outputs/logs/c3``."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("logger name must be a non-empty string")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if getattr(logger, "_c3_configured", False):
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    formatter = logging.Formatter(_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        _LOG_DIR / f"{safe_name}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger._c3_configured = True  # type: ignore[attr-defined]
    return logger
