"""Logging configuration helpers for the BTC CPI Backtest package."""

from __future__ import annotations

import logging
from typing import Mapping

_LOG_LEVELS: Mapping[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str | int = logging.INFO) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level:
        Either a logging level name (``"INFO"``) or numeric level (``20``).
    """

    resolved_level = _resolve_level(level)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    upper = level.upper()
    if upper not in _LOG_LEVELS:
        raise ValueError(
            f"Unsupported log level '{level}'. Choose from {', '.join(_LOG_LEVELS)}."
        )
    return _LOG_LEVELS[upper]
