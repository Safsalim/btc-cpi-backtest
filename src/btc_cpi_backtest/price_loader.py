"""Utilities for loading BTC price data for analysis."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timezone, tzinfo
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
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


def _resolve_column_name(df: pd.DataFrame, candidates: Sequence[str | None]) -> str:
    available = {column.lower(): column for column in df.columns}
    seen: set[str] = set()
    display_candidates: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = candidate.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        display_candidates.append(normalized)
        if lower in available:
            return available[lower]

    formatted = ", ".join(display_candidates) if display_candidates else "no candidates provided"
    raise ValueError(
        f"Missing required column(s) in price data. Expected one of: {formatted}"
    )


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

    timestamp_candidates: tuple[str | None, ...] = (
        timestamp_column,
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "date",
        "Date",
        "datetime",
        "Datetime",
    )
    timestamp_column_name = _resolve_column_name(df, timestamp_candidates)

    close_candidates: tuple[str | None, ...] = (
        close_column,
        "close",
        "Close",
        "closing_price",
        "Closing Price",
    )
    close_column_name = _resolve_column_name(df, close_candidates)

    timestamp_raw = df[timestamp_column_name]
    if is_numeric_dtype(timestamp_raw):
        timestamp_values = pd.to_datetime(timestamp_raw, unit="s", errors="coerce", utc=True)
    else:
        numeric_timestamp = pd.to_numeric(timestamp_raw, errors="coerce")
        if numeric_timestamp.notna().all():
            timestamp_values = pd.to_datetime(
                numeric_timestamp,
                unit="s",
                errors="coerce",
                utc=True,
            )
        else:
            timestamp_values = pd.to_datetime(timestamp_raw, errors="coerce", utc=False)

    if isinstance(timestamp_values, pd.Series):
        timestamp_series = timestamp_values.copy()
    else:
        timestamp_series = pd.Series(timestamp_values, index=df.index)
    if timestamp_series.isna().any():
        raise ValueError("Invalid or missing timestamps detected in price dataset")

    tzinfo_value = _resolve_timezone(input_timezone)
    if timestamp_series.dt.tz is None:
        tz_to_use = tzinfo_value or timezone.utc
        timestamp_series = timestamp_series.dt.tz_localize(tz_to_use)
    elif tzinfo_value is not None:
        timestamp_series = timestamp_series.dt.tz_convert(tzinfo_value)

    timestamp_series = timestamp_series.dt.tz_convert(timezone.utc)

    close_series = pd.to_numeric(df[close_column_name], errors="coerce")
    if close_series.isna().any():
        raise ValueError("Close prices must be numeric")

    result = pd.Series(close_series.to_numpy(), index=timestamp_series)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result.name = close_column_name
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
