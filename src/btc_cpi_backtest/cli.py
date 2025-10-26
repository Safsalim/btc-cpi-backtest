"""Command line interface for the BTC CPI Backtest package."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from .cpi_loader import CPIColumnConfig, load_cpi_from_csv, load_sample_cpi_data
from .logging_config import configure_logging
from .market_data import download_cpi_market_data, timeframe_to_seconds
from .settings import Settings, get_settings

app = typer.Typer(
    add_completion=True,
    help="Utilities for exploring Bitcoin performance relative to CPI benchmarks.",
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_CACHE_DIR = Path("data") / "btc_cpi_cache"


def _normalize_optional_column(column_name: Optional[str]) -> Optional[str]:
    if column_name is None:
        return None
    trimmed = column_name.strip()
    if not trimmed or trimmed.lower() == "none":
        return None
    return trimmed


def _parse_datetime_option(value: Optional[str], option_name: str) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - validated via CLI
        raise typer.BadParameter(f"Invalid ISO datetime for {option_name}: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
        "1m",
        "--timeframe",
        "-t",
        help="CCXT timeframe for the OHLCV data.",
        show_default=True,
    ),
    cpi_source: str = typer.Option(
        "sample",
        "--cpi-source",
        "-c",
        help="Path to a CPI CSV file or 'sample' to use the bundled dataset.",
        show_default=True,
    ),
    cache_dir: Path = typer.Option(
        _DEFAULT_CACHE_DIR,
        "--cache-dir",
        help="Directory where candle data should be cached.",
        show_default=True,
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox/--no-sandbox",
        help="Whether to use the exchange sandbox/testnet if available.",
        show_default=True,
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Optional ISO-8601 timestamp (UTC) for the earliest CPI release to include.",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="Optional ISO-8601 timestamp (UTC) for the latest CPI release to include.",
    ),
    lookback_years: int = typer.Option(
        3,
        "--years",
        min=1,
        help="Number of years to look back when no --start value is provided.",
        show_default=True,
    ),
) -> None:
    """Fetch BTC candles around CPI releases with caching."""

    settings_obj = ctx.obj.get("settings")
    if isinstance(settings_obj, Settings):
        settings = settings_obj
    else:  # pragma: no cover - should not happen when callback executes
        settings = get_settings()
        ctx.obj["settings"] = settings

    try:
        timeframe_to_seconds(timeframe)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    start_dt = _parse_datetime_option(start, "--start")
    end_dt = _parse_datetime_option(end, "--end")
    if start_dt and end_dt and start_dt >= end_dt:
        raise typer.BadParameter("--start must be earlier than --end")

    now_utc = datetime.now(timezone.utc)
    if start_dt is None:
        start_dt = now_utc - timedelta(days=lookback_years * 365)
    if end_dt is None:
        end_dt = now_utc

    cache_dir = cache_dir.expanduser()

    columns = CPIColumnConfig()
    try:
        if cpi_source.lower() == "sample":
            releases = load_sample_cpi_data(columns=columns, input_timezone=timezone.utc)
            source_label = "bundled sample dataset"
        else:
            path = Path(cpi_source)
            releases = load_cpi_from_csv(path, columns=columns, input_timezone=timezone.utc)
            source_label = str(path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    exchange_name = exchange.strip().lower()
    if not exchange_name:
        raise typer.BadParameter("Exchange name must not be empty.")

    _LOGGER.info(
        "Starting CPI candle download: exchange=%s symbol=%s timeframe=%s cache=%s",
        exchange_name,
        symbol,
        timeframe,
        cache_dir,
    )
    _LOGGER.info(
        "CPI releases considered from %s to %s using source %s",
        start_dt.isoformat(),
        end_dt.isoformat(),
        source_label,
    )

    summary = download_cpi_market_data(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe=timeframe,
        releases=releases,
        cache_dir=cache_dir,
        settings=settings,
        sandbox=sandbox,
        start=start_dt,
        end=end_dt,
    )

    typer.echo(
        f"Processed {summary.total_releases} CPI releases from {source_label}."
    )
    typer.echo(
        f"Downloaded: {summary.downloaded}, Reused cache: {summary.reused}, Candles: {summary.total_candles}"
    )
    typer.echo(f"Cache directory: {cache_dir}")


@app.command("cpi-summary")
def cpi_summary(
    source: str = typer.Option(
        "sample",
        "--source",
        "-s",
        help="Path to a CPI CSV file or 'sample' to use the bundled dataset.",
        show_default=True,
    ),
    timestamp_column: str = typer.Option(
        "release_time",
        "--timestamp-column",
        help="Column name containing CPI release timestamps.",
        show_default=True,
    ),
    actual_column: str = typer.Option(
        "actual",
        "--actual-column",
        help="Column name containing the actual CPI value.",
        show_default=True,
    ),
    expected_column: str = typer.Option(
        "expected",
        "--expected-column",
        help="Column name containing the expected CPI value.",
        show_default=True,
    ),
    previous_column: Optional[str] = typer.Option(
        "previous",
        "--previous-column",
        help="Column name for the previous CPI value (set to 'none' if unavailable).",
        show_default=True,
    ),
    surprise_column: Optional[str] = typer.Option(
        "surprise",
        "--surprise-column",
        help="Column name for the CPI surprise value (set to 'none' if unavailable).",
        show_default=True,
    ),
    input_timezone: str = typer.Option(
        "UTC",
        "--timezone",
        "-t",
        help="Timezone to assume for naive CPI timestamps (use 'none' to leave unspecified).",
        show_default=True,
    ),
) -> None:
    """Load CPI data from CSV and print a release summary."""

    columns = CPIColumnConfig(
        timestamp=timestamp_column,
        actual=actual_column,
        expected=expected_column,
        previous=_normalize_optional_column(previous_column),
        surprise=_normalize_optional_column(surprise_column),
    )

    timezone_arg: Optional[str]
    if input_timezone.strip().lower() == "none":
        timezone_arg = None
    else:
        timezone_arg = input_timezone

    try:
        if source.lower() == "sample":
            releases = load_sample_cpi_data(columns=columns, input_timezone=timezone_arg)
            source_label = "bundled sample dataset"
        else:
            path = Path(source)
            releases = load_cpi_from_csv(path, columns=columns, input_timezone=timezone_arg)
            source_label = str(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not releases:
        typer.echo(f"No CPI releases found in {source_label}.")
        return

    first_release = releases[0].release_datetime.isoformat()
    last_release = releases[-1].release_datetime.isoformat()
    typer.echo(f"Loaded {len(releases)} CPI releases from {source_label}.")
    typer.echo(f"Date range: {first_release} to {last_release}")


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
