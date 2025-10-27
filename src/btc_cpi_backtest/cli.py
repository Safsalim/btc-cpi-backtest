"""Command line interface for the BTC CPI Backtest package."""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import typer

from .analysis import FakeoutConfig, FakeoutStats, FakeoutSummary, analyze_fakeouts
from .cpi_loader import CPIColumnConfig, CPIRelease, load_cpi_from_csv, load_sample_cpi_data
from .dashboard import DEFAULT_DASHBOARD_PATH, render_dashboard_html
from .logging_config import configure_logging
from .price_loader import load_price_series_from_csv, load_sample_price_series
from .settings import Settings, get_settings

app = typer.Typer(
    add_completion=True,
    help="Utilities for exploring Bitcoin performance relative to CPI benchmarks.",
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA_PATH = Path("data") / "btc_market_data.csv"
_DEFAULT_BTC_DATA_PATH = Path("btcusd_1-min_data.csv")
_DEFAULT_CPI_DATA_PATH = Path("data") / "cpi_releases.csv"
_DEFAULT_OUTPUT_PATH = Path("data") / "fakeout_analysis.csv"
_DEFAULT_ANALYSIS_CONFIG = FakeoutConfig()


def _normalize_optional_column(column_name: Optional[str]) -> Optional[str]:
    if column_name is None:
        return None
    trimmed = column_name.strip()
    if not trimmed or trimmed.lower() == "none":
        return None
    return trimmed


def _normalize_timezone_option(value: str) -> Optional[str]:
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _normalize_html_output_option(value: str) -> Optional[Path]:
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return Path(normalized).expanduser()


def _parse_window_expression(value: str) -> tuple[str, timedelta]:
    original = value.strip()
    if not original:
        raise ValueError("Window duration cannot be empty")

    normalized = original.lower()
    if normalized.endswith("m"):
        unit = "m"
        magnitude_str = normalized[:-1]
    elif normalized.endswith("h"):
        unit = "h"
        magnitude_str = normalized[:-1]
    else:
        raise ValueError("Window definitions must end with 'm' or 'h'")

    try:
        magnitude = int(magnitude_str)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid window magnitude: {original}") from exc

    if magnitude <= 0:
        raise ValueError("Window duration must be positive")

    if unit == "m":
        delta = timedelta(minutes=magnitude)
    else:
        delta = timedelta(hours=magnitude)

    label = f"{magnitude}{unit}"
    return label, delta


def _coerce_windows(
    defaults: Sequence[tuple[str, timedelta]],
    overrides: List[str],
) -> tuple[tuple[str, timedelta], ...]:
    if not overrides:
        return tuple(defaults)

    parsed: dict[str, timedelta] = {}
    for value in overrides:
        label, delta = _parse_window_expression(value)
        parsed[label] = delta
    return tuple((label, parsed[label]) for label in parsed)


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(value):
            return "n/a"
    except TypeError:  # pragma: no cover - defensive guard
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_ratio(stats: Optional[FakeoutStats]) -> str:
    if stats is None:
        return "n/a"
    if stats.total == 0:
        return "n/a"
    ratio = stats.fake_ratio
    if ratio is None:
        return "n/a"
    return f"{stats.fake_count}/{stats.total} ({ratio * 100:.1f}%)"


@lru_cache(maxsize=8)
def _cached_price_series(
    source: str,
    timestamp_column: str,
    close_column: str,
    timezone_value: Optional[str],
) -> pd.Series:
    if source == "__sample__":
        return load_sample_price_series(
            timestamp_column=timestamp_column,
            close_column=close_column,
            input_timezone=timezone_value,
        )
    return load_price_series_from_csv(
        source,
        timestamp_column=timestamp_column,
        close_column=close_column,
        input_timezone=timezone_value,
    )


def _print_summary(summary: FakeoutSummary, config: FakeoutConfig) -> None:
    typer.echo(f"Analysis complete for {summary.event_count} CPI releases.")
    typer.echo("Fake-out rates:")
    for reaction_label, _ in config.reaction_windows:
        for evaluation_label, _ in config.evaluation_windows:
            stats = summary.fake_out_stats.get(reaction_label, {}).get(evaluation_label)
            typer.echo(f"  {reaction_label} -> {evaluation_label}: {_format_ratio(stats)}")

    typer.echo("Average returns (%):")
    typer.echo("  Reaction windows:")
    for reaction_label, _ in config.reaction_windows:
        value = summary.average_returns.get("reaction", {}).get(reaction_label)
        typer.echo(f"    {reaction_label}: {_format_percentage(value)}")

    typer.echo("  Evaluation windows:")
    for evaluation_label, _ in config.evaluation_windows:
        value = summary.average_returns.get("evaluation", {}).get(evaluation_label)
        typer.echo(f"    {evaluation_label}: {_format_percentage(value)}")

    typer.echo("Correlation with CPI surprise:")
    for evaluation_label, _ in config.evaluation_windows:
        value = summary.surprise_correlations.get(evaluation_label)
        if value is None or math.isnan(value):
            typer.echo(f"  {evaluation_label}: n/a")
        else:
            typer.echo(f"  {evaluation_label}: {value:.3f}")


def _export_report(events: pd.DataFrame, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        events.to_csv(destination, index=False)
    elif fmt == "json":
        events.to_json(destination, orient="records", date_format="iso")
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"Unsupported output format: {fmt}")


def _render_plot(events: pd.DataFrame, config: FakeoutConfig) -> None:
    if events.empty:
        typer.echo("No analysis events available to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        typer.echo("Matplotlib is not installed; skipping plot generation.")
        return

    time_values = events["release_datetime"]
    for label, _ in config.evaluation_windows:
        series = events[f"return_{label}"] * 100.0
        plt.plot(time_values, series, marker="o", label=f"{label} return")

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Return (%)")
    plt.title("BTC performance after CPI releases")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _warn_on_price_coverage(
    price_series: pd.Series,
    releases: Sequence[CPIRelease],
    evaluation_windows: Sequence[tuple[str, timedelta]],
) -> None:
    if price_series.empty or not releases:
        return

    first_timestamp = price_series.index.min()
    missing_before = sum(
        1 for release in releases if release.release_datetime < first_timestamp
    )
    if missing_before:
        _LOGGER.warning(
            "Price data begins at %s; %d CPI releases occur before this time and may lack price coverage.",
            first_timestamp.isoformat(),
            missing_before,
        )

    if not evaluation_windows:
        return

    max_eval_delta = max(delta for _, delta in evaluation_windows)
    last_timestamp = price_series.index.max()
    missing_after = sum(
        1
        for release in releases
        if release.release_datetime + max_eval_delta > last_timestamp
    )
    if missing_after:
        _LOGGER.warning(
            "Price data ends at %s; %d CPI releases extend beyond the available candles. Evaluation windows may be incomplete.",
            last_timestamp.isoformat(),
            missing_after,
        )


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


@app.command("cpi-summary")
def cpi_summary(
    source: str = typer.Option(
        "sample",
        "--source",
        "-s",
        help="Path to a CPI CSV file or 'sample' to use the bundled dataset.",
        show_default=True,
    ),
    timestamp_column: Optional[str] = typer.Option(
        None,
        "--timestamp-column",
        help="Column name containing CPI release timestamps (set to 'none' when using separate date and time columns).",
    ),
    date_column: Optional[str] = typer.Option(
        "Release Date",
        "--date-column",
        help="Column name containing the CPI release date (set to 'none' when a combined timestamp column is provided).",
        show_default=True,
    ),
    time_column: Optional[str] = typer.Option(
        "Time",
        "--time-column",
        help="Column name containing the CPI release time.",
        show_default=True,
    ),
    actual_column: str = typer.Option(
        "Actual",
        "--actual-column",
        help="Column name containing the actual CPI value.",
        show_default=True,
    ),
    expected_column: str = typer.Option(
        "Forecast",
        "--expected-column",
        help="Column name containing the expected CPI value.",
        show_default=True,
    ),
    previous_column: Optional[str] = typer.Option(
        "Previous",
        "--previous-column",
        help="Column name for the previous CPI value (set to 'none' if unavailable).",
        show_default=True,
    ),
    surprise_column: Optional[str] = typer.Option(
        None,
        "--surprise-column",
        help="Column name for the CPI surprise value (set to 'none' if unavailable).",
    ),
    input_timezone: str = typer.Option(
        "US/Eastern",
        "--timezone",
        "-t",
        help="Timezone to assume for naive CPI timestamps (use 'none' to leave unspecified).",
        show_default=True,
    ),
) -> None:
    """Load CPI data from CSV and print a release summary."""

    columns = CPIColumnConfig(
        timestamp=_normalize_optional_column(timestamp_column),
        date=_normalize_optional_column(date_column),
        time=_normalize_optional_column(time_column),
        actual=actual_column.strip(),
        expected=expected_column.strip(),
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
    ctx: typer.Context,
    strategy: str = typer.Option(
        "fake-moving-average",
        "--strategy",
        "-s",
        help="Label for the analysis run (for logging purposes).",
        show_default=True,
    ),
    lookback_days: int = typer.Option(
        365,
        "--lookback-days",
        "-n",
        help="How many days prior to the first CPI release to retain from price data.",
        show_default=True,
    ),
    btc_data: Path = typer.Option(
        _DEFAULT_BTC_DATA_PATH,
        "--btc-data",
        "-b",
        help="Path to a BTC OHLCV CSV file.",
        show_default=True,
    ),
    cpi_data: Path = typer.Option(
        _DEFAULT_CPI_DATA_PATH,
        "--cpi-data",
        "-c",
        help="Path to a CPI release CSV file.",
        show_default=True,
    ),
    cpi_timestamp_column: Optional[str] = typer.Option(
        None,
        "--cpi-timestamp-column",
        help="CPI column containing release timestamps (set to 'none' when using separate date and time columns).",
    ),
    cpi_date_column: Optional[str] = typer.Option(
        "Release Date",
        "--cpi-date-column",
        help="CPI column containing release dates (set to 'none' when a combined timestamp column is provided).",
        show_default=True,
    ),
    cpi_time_column: Optional[str] = typer.Option(
        "Time",
        "--cpi-time-column",
        help="CPI column containing release times.",
        show_default=True,
    ),
    cpi_actual_column: str = typer.Option(
        "Actual",
        "--cpi-actual-column",
        help="CPI column containing actual CPI values.",
        show_default=True,
    ),
    cpi_expected_column: str = typer.Option(
        "Forecast",
        "--cpi-expected-column",
        help="CPI column containing expected CPI values.",
        show_default=True,
    ),
    cpi_previous_column: Optional[str] = typer.Option(
        "Previous",
        "--cpi-previous-column",
        help="CPI column containing previous CPI values (set to 'none' if unavailable).",
        show_default=True,
    ),
    cpi_surprise_column: Optional[str] = typer.Option(
        None,
        "--cpi-surprise-column",
        help="CPI column containing surprise values (set to 'none' if unavailable).",
    ),
    cpi_timezone: str = typer.Option(
        "US/Eastern",
        "--cpi-timezone",
        help="Timezone for naive CPI timestamps (use 'none' to leave unspecified).",
        show_default=True,
    ),
    btc_timestamp_column: str = typer.Option(
        "Timestamp",
        "--btc-timestamp-column",
        help="BTC column containing candle timestamps.",
        show_default=True,
    ),
    btc_close_column: str = typer.Option(
        "Close",
        "--btc-close-column",
        help="BTC column containing close values.",
        show_default=True,
    ),
    btc_timezone: str = typer.Option(
        "UTC",
        "--btc-timezone",
        help="Timezone for naive BTC timestamps (use 'none' to leave unspecified).",
        show_default=True,
    ),
    reaction_window: List[str] = typer.Option(
        [],
        "--reaction-window",
        help="Override reaction window durations (e.g., '--reaction-window 10m').",
    ),
    evaluation_window: List[str] = typer.Option(
        [],
        "--evaluation-window",
        help="Override evaluation window durations (e.g., '--evaluation-window 90m').",
    ),
    output: Path = typer.Option(
        _DEFAULT_OUTPUT_PATH,
        "--output",
        "-o",
        help="Destination for the detailed per-release analysis report.",
        show_default=True,
    ),
    output_format: str = typer.Option(
        "csv",
        "--output-format",
        "-f",
        help="Format for the detailed report (csv or json).",
        show_default=True,
    ),
    html_output: str = typer.Option(
        str(DEFAULT_DASHBOARD_PATH),
        "--html-output",
        help="Write interactive HTML dashboard to this path ('none' to skip).",
        show_default=True,
    ),
    open_html: bool = typer.Option(
        False,
        "--open-html/--no-open-html",
        help="Open the generated HTML dashboard in a browser.",
        show_default=False,
    ),
    plot: bool = typer.Option(
        False,
        "--plot/--no-plot",
        help="Generate a matplotlib plot showing evaluation window returns.",
        show_default=False,
    ),
) -> None:
    """Evaluate BTC reactions to CPI releases and highlight potential fake moves."""

    _LOGGER.info(
        "Running fake move analysis (strategy=%s, cpi_data=%s, btc_data=%s, output=%s)",
        strategy,
        cpi_data,
        btc_data,
        output,
    )

    columns = CPIColumnConfig(
        timestamp=_normalize_optional_column(cpi_timestamp_column),
        date=_normalize_optional_column(cpi_date_column),
        time=_normalize_optional_column(cpi_time_column),
        actual=cpi_actual_column.strip(),
        expected=cpi_expected_column.strip(),
        previous=_normalize_optional_column(cpi_previous_column),
        surprise=_normalize_optional_column(cpi_surprise_column),
    )

    cpi_timezone_arg = _normalize_timezone_option(cpi_timezone)

    cpi_label: str
    cpi_data_str = str(cpi_data)
    try:
        if cpi_data_str.lower() == "sample":
            releases = load_sample_cpi_data(columns=columns, input_timezone=cpi_timezone_arg)
            cpi_label = "bundled sample CPI dataset"
        else:
            cpi_path = Path(cpi_data)
            if not cpi_path.exists() and cpi_path.parent == Path("."):
                candidate_path = Path("data") / cpi_path.name
                if candidate_path.exists():
                    cpi_path = candidate_path
            releases = load_cpi_from_csv(cpi_path, columns=columns, input_timezone=cpi_timezone_arg)
            cpi_label = str(cpi_path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not releases:
        typer.echo(f"No CPI releases found in {cpi_label}.")
        return

    try:
        reaction_windows = _coerce_windows(_DEFAULT_ANALYSIS_CONFIG.reaction_windows, reaction_window)
        evaluation_windows = _coerce_windows(
            _DEFAULT_ANALYSIS_CONFIG.evaluation_windows,
            evaluation_window,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    analysis_config = FakeoutConfig(
        reaction_windows=reaction_windows,
        evaluation_windows=evaluation_windows,
        tolerance=_DEFAULT_ANALYSIS_CONFIG.tolerance,
    )

    btc_timezone_arg = _normalize_timezone_option(btc_timezone)

    btc_label: str
    btc_data_str = str(btc_data)
    try:
        if btc_data_str.lower() == "sample":
            price_series = _cached_price_series(
                "__sample__",
                btc_timestamp_column,
                btc_close_column,
                btc_timezone_arg,
            ).copy()
            btc_label = "bundled sample price dataset"
        else:
            btc_path = Path(btc_data)
            if not btc_path.exists() and btc_path.parent == Path("."):
                candidate_path = Path("data") / btc_path.name
                if candidate_path.exists():
                    btc_path = candidate_path
            price_series = _cached_price_series(
                str(btc_path.resolve()),
                btc_timestamp_column,
                btc_close_column,
                btc_timezone_arg,
            ).copy()
            btc_label = str(btc_path)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    earliest_release = min(release.release_datetime for release in releases)
    start_time = earliest_release - timedelta(days=lookback_days)
    price_series = price_series.loc[start_time:]

    if analysis_config.evaluation_windows:
        max_eval_delta = max(delta for _, delta in analysis_config.evaluation_windows)
    else:
        max_eval_delta = timedelta(0)
    latest_release = max(release.release_datetime for release in releases)
    end_time = latest_release + max_eval_delta
    price_series = price_series.loc[: end_time + timedelta(minutes=5)]

    if price_series.empty:
        typer.echo("Price dataset does not cover the requested analysis period.")
        return

    _warn_on_price_coverage(price_series, releases, analysis_config.evaluation_windows)

    typer.echo(f"Loaded {len(releases)} CPI releases from {cpi_label}.")
    typer.echo(f"Using {len(price_series)} BTC price points from {btc_label}.")

    result = analyze_fakeouts(releases, price_series, config=analysis_config)

    _print_summary(result.summary, analysis_config)

    fmt = output_format.strip().lower()
    if fmt not in {"csv", "json"}:
        raise typer.BadParameter("output-format must be either 'csv' or 'json'")

    _export_report(result.events, output, fmt)
    typer.echo(f"Saved detailed report to {output.resolve()}")

    html_output_path = _normalize_html_output_option(html_output)
    if html_output_path is None:
        if open_html:
            typer.echo("HTML dashboard generation disabled; ignoring --open-html flag.")
    else:
        try:
            destination = render_dashboard_html(
                result=result,
                releases=releases,
                price_series=price_series,
                config=analysis_config,
                output_path=html_output_path,
                open_browser=open_html,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise typer.BadParameter(
                "Plotly is required to generate the HTML dashboard. Install with 'pip install plotly'."
            ) from exc
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        else:
            typer.echo(f"Saved interactive dashboard to {destination.resolve()}")

    if plot:
        _render_plot(result.events, analysis_config)


def main() -> None:
    """Invoke the Typer application."""

    app()
