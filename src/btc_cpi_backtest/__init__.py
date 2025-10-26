"""BTC CPI Backtest package."""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("btc-cpi-backtest")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.1.0"

__all__ = ["__version__"]
