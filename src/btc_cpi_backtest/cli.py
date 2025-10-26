"""Command line interface for the BTC CPI Backtest package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from .logging_config import configure_logging
from .settings import Settings, get_settings

app = typer.Typer(
    add_completion=True,
    help="Utilities for exploring Bitcoin performance relative to CPI benchmarks.",
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA_PATH = Path("data") / "btc_market_data.csv"


@app.callback()
def configure_cli(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging verbosity. One of: CRITICAL, ERROR, WARNING, INFO, DEBUG.",
        show_default=True,
    ),
) -> None:
    """Configure application-wide dependencies for CLI commands."""

    ctx.ensure_object(dict)
    try:
        configure_logging(log_level)
    except ValueError as exc:  # pragma: no cover - defensive, exercised via CLI
        raise typer.BadParameter(str(exc)) from exc

    settings = get_settings()
    ctx.obj["settings"] = settings
    if settings.ccxt_api_key:
        _LOGGER.debug("CCXT API credentials loaded from environment variables.")
    else:
        _LOGGER.debug("No CCXT API credentials provided; using public endpoints if available.")


@app.command("fetch-data")
def fetch_data(
    ctx: typer.Context,
    exchange: str = typer.Option(
        "binance",
        "--exchange",
        "-e",
        help="Name of the exchange supported by CCXT.",
        show_default=True,
    ),
    symbol: str = typer.Option(
        "BTC/USDT",
        "--symbol",
        "-s",
        help="Trading pair symbol to pull data for.",
    ),
    timeframe: str = typer.Option(
        "1d",
        "--timeframe",
        "-t",
        help="CCXT timeframe for the OHLCV data.",
        show_default=True,
    ),
    limit: int = typer.Option(
        365,
        "--limit",
        "-n",
        help="Number of candles to download from the exchange.",
        show_default=True,
    ),
    destination: Path = typer.Option(
        _DEFAULT_DATA_PATH,
        "--destination",
        "-d",
        help="Where to save the downloaded market data.",
        show_default=True,
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox/--no-sandbox",
        help="Whether to use the exchange sandbox/testnet if available.",
        show_default=True,
    ),
) -> None:
    """Fetch market data using CCXT (placeholder implementation)."""

    settings_obj = ctx.obj.get("settings")
    if isinstance(settings_obj, Settings):
        settings = settings_obj
    else:  # pragma: no cover - should not happen when callback executes
        settings = get_settings()
        ctx.obj["settings"] = settings

    _LOGGER.info(
        "Fetching %s candles for %s on %s (sandbox=%s) -> %s",
        limit,
        symbol,
        exchange,
        sandbox,
        destination,
    )
    if settings.ccxt_api_key:
        _LOGGER.debug("CCXT API key detected; authenticated requests are available.")
    typer.echo(
        "Download not yet implemented. This command will fetch data in a future iteration."
    )


@app.command()
def analyze(
    _ctx: typer.Context,
    strategy: str = typer.Option(
        "buy-and-hold",
        "--strategy",
        "-s",
        help="Strategy name to evaluate.",
        show_default=True,
    ),
    lookback_days: int = typer.Option(
        365,
        "--lookback-days",
        "-n",
        help="Historical window to evaluate performance over.",
        show_default=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to export analysis results.",
    ),
) -> None:
    """Analyze CPI-relative performance for a placeholder trading strategy."""

    _LOGGER.info(
        "Running analysis for strategy '%s' over %s days (output=%s)",
        strategy,
        lookback_days,
        output or "stdout",
    )
    typer.echo(
        "Analysis not yet implemented. This command will evaluate strategies in a later release."
    )

def main() -> None:
    """Invoke the Typer application."""

    app()
