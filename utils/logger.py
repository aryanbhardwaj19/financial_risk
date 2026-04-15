"""
Application-wide logging configuration.

Usage
-----
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from config import settings

# ── Formatting ──────────────────────────────────────────────────────
_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# We configure the root logger exactly once – subsequent calls to
# get_logger simply return a child, inheriting this setup.
_ROOT_CONFIGURED = False


def _configure_root() -> None:
    """Set up the root logger with a stdout handler and formatter."""
    global _ROOT_CONFIGURED  # noqa: PLW0603
    if _ROOT_CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)
    _ROOT_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger (creates the root setup on first call).

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.  If *None* the
        root logger is returned.
    """
    _configure_root()
    return logging.getLogger(name)
