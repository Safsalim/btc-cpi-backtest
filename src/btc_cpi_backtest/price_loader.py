"""Utilities for loading BTC price data for analysis."""

from __future__ import annotations

from datetime import timezone, tzinfo
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "load_price_series_from_csv",
    "load_sample_price_series",
]


def _resolve_timezone(value: str | tzinfo | None) -> tzinfo | None:
    if value is None:
        return None
    if isinstance(value, tzinfo):
        return value
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - depends on system tz database
        raise ValueError(f"Unknown timezone: {value}") from exc


def load_price_series_from_csv(
    csv_path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    close_column: str = "close",
    input_timezone: str | tzinfo | None = "UTC",
) -> pd.Series:
    """Load BTC close prices from a CSV file into a time-indexed Series."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Price data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Price dataset {path} is empty")

    missing = [col for col in (timestamp_column, close_column) if col not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required column(s) in price data: {missing_list}")

    timestamp_series = pd.to_datetime(df[timestamp_column], errors="coerce", utc=False)
    if timestamp_series.isna().any():
        raise ValueError("Invalid or missing timestamps detected in price dataset")

    tzinfo_value = _resolve_timezone(input_timezone)
    if timestamp_series.dt.tz is None:
        tz_to_use = tzinfo_value or timezone.utc
        timestamp_series = timestamp_series.dt.tz_localize(tz_to_use)
    elif tzinfo_value is not None:
        timestamp_series = timestamp_series.dt.tz_convert(tzinfo_value)

    timestamp_series = timestamp_series.dt.tz_convert(timezone.utc)

    close_series = pd.to_numeric(df[close_column], errors="coerce")
    if close_series.isna().any():
        raise ValueError("Close prices must be numeric")

    result = pd.Series(close_series.to_numpy(), index=timestamp_series)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result.name = close_column
    return result


_SAMPLE_FILENAME = "btc_price_sample.csv"


def load_sample_price_series(
    *,
    timestamp_column: str = "timestamp",
    close_column: str = "close",
    input_timezone: str | tzinfo | None = "UTC",
) -> pd.Series:
    """Load a bundled sample of BTC close prices for demonstration purposes."""

    base_path = Path(__file__).resolve().parents[2]
    sample_path = base_path / "data" / _SAMPLE_FILENAME
    return load_price_series_from_csv(
        sample_path,
        timestamp_column=timestamp_column,
        close_column=close_column,
        input_timezone=input_timezone,
    )
