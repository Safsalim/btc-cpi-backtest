from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import re

import pandas as pd

__all__ = [
    "CPIColumnConfig",
    "CPIRelease",
    "load_cpi_from_csv",
    "load_sample_cpi_data",
]


@dataclass(frozen=True)
class CPIColumnConfig:
    """Configuration describing the columns available in a CPI dataset."""

    timestamp: Optional[str] = "release_time"
    date: Optional[str] = None
    time: Optional[str] = None
    actual: str = "actual"
    expected: str = "expected"
    previous: Optional[str] = "previous"
    surprise: Optional[str] = "surprise"

    def required_columns(self) -> tuple[str, ...]:
        required: list[str] = [self.actual, self.expected]
        if self.timestamp:
            required.append(self.timestamp)
        elif self.date and self.time:
            required.extend([self.date, self.time])
        else:
            raise ValueError(
                "CPIColumnConfig must specify either a timestamp column or both date and time columns"
            )
        return tuple(column for column in required if column)


@dataclass(frozen=True)
class CPIRelease:
    """Normalized CPI release information."""

    release_datetime: datetime
    actual: float
    expected: float
    previous: Optional[float] = None
    surprise: Optional[float] = None

    def __post_init__(self) -> None:
        if self.release_datetime.tzinfo is None or self.release_datetime.utcoffset() is None:
            raise ValueError("release_datetime must be timezone-aware")
        object.__setattr__(self, "release_datetime", self.release_datetime.astimezone(timezone.utc))


def _clean_string_series(series: pd.Series) -> pd.Series:
    """Strip whitespace and normalize empty/placeholder values to NaN."""

    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": pd.NA})
    cleaned = cleaned.replace(to_replace=r"(?i)^(nan|none)$", value=pd.NA, regex=True)
    return cleaned


def _resolve_column_name(df: pd.DataFrame, candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    normalized = candidate.strip()
    if not normalized:
        return None
    if normalized in df.columns:
        return normalized
    lower_map = {column.lower(): column for column in df.columns}
    return lower_map.get(normalized.lower())


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = _clean_string_series(series)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _build_timestamp_series(
    df: pd.DataFrame,
    *,
    timestamp_col: Optional[str],
    date_col: Optional[str],
    time_col: Optional[str],
) -> pd.Series:
    if timestamp_col:
        return pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)

    if date_col and time_col:
        date_series = _clean_string_series(df[date_col])
        time_series = _clean_string_series(df[time_col])

        combined: list[Optional[str]] = []
        for date_value, time_value in zip(date_series, time_series):
            if pd.isna(date_value) or pd.isna(time_value):
                combined.append(None)
                continue
            cleaned_date = re.sub(r"\s*\(.*?\)", "", str(date_value)).strip()
            combined.append(f"{cleaned_date} {time_value}")

        combined_series = pd.Series(combined, index=df.index)
        return pd.to_datetime(combined_series, errors="coerce", format="%b %d, %Y %H:%M")

    raise ValueError("Missing columns to construct CPI release timestamps")


def _resolve_timezone(value: str | tzinfo | None) -> tzinfo | None:
    if value is None:
        return None
    if isinstance(value, tzinfo):
        return value
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - depends on system tz database
        raise ValueError(f"Unknown timezone: {value}") from exc


def _optional_numeric_series(df: pd.DataFrame, column_name: Optional[str]) -> pd.Series:
    if column_name:
        return _parse_numeric_series(df[column_name])
    return pd.Series([float("nan")] * len(df), index=df.index)


def load_cpi_from_csv(
    csv_path: str | Path,
    *,
    columns: CPIColumnConfig | None = None,
    input_timezone: str | tzinfo | None = "UTC",
) -> List[CPIRelease]:
    """Load CPI release data from a CSV file and normalize to UTC."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CPI data file not found: {path}")

    config = columns or CPIColumnConfig()

    df = pd.read_csv(path)
    if df.empty:
        return []

    df.columns = [str(column).strip() for column in df.columns]

    resolved_required: dict[str, str] = {}
    missing: list[str] = []
    for column_name in config.required_columns():
        resolved = _resolve_column_name(df, column_name)
        if resolved is None:
            missing.append(column_name)
        else:
            resolved_required[column_name] = resolved

    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_list}")

    timestamp_series = _build_timestamp_series(
        df,
        timestamp_col=resolved_required.get(config.timestamp) if config.timestamp else None,
        date_col=resolved_required.get(config.date) if config.date else None,
        time_col=resolved_required.get(config.time) if config.time else None,
    )
    if timestamp_series.isna().any():
        raise ValueError("Invalid or missing release timestamps detected in CPI dataset")

    tzinfo_value = _resolve_timezone(input_timezone)
    if timestamp_series.dt.tz is None:
        tz_to_use = tzinfo_value or timezone.utc
        timestamp_series = timestamp_series.dt.tz_localize(
            tz_to_use, ambiguous="infer", nonexistent="raise"
        )

    timestamp_series = timestamp_series.dt.tz_convert(timezone.utc)

    actual_column = resolved_required[config.actual]
    expected_column = resolved_required[config.expected]
    actual_series = _parse_numeric_series(df[actual_column])
    expected_series = _parse_numeric_series(df[expected_column])
    if actual_series.isna().any() or expected_series.isna().any():
        raise ValueError("Actual and expected CPI values must be numeric")

    previous_column = _resolve_column_name(df, config.previous)
    surprise_column = _resolve_column_name(df, config.surprise)
    previous_series = _optional_numeric_series(df, previous_column)
    surprise_series = _optional_numeric_series(df, surprise_column)

    releases: list[CPIRelease] = []
    for idx in range(len(df)):
        dt_value = timestamp_series.iloc[idx].to_pydatetime().astimezone(timezone.utc)
        actual_value = float(actual_series.iloc[idx])
        expected_value = float(expected_series.iloc[idx])

        previous_value = previous_series.iloc[idx]
        surprise_value = surprise_series.iloc[idx]

        releases.append(
            CPIRelease(
                release_datetime=dt_value,
                actual=actual_value,
                expected=expected_value,
                previous=None if pd.isna(previous_value) else float(previous_value),
                surprise=None if pd.isna(surprise_value) else float(surprise_value),
            )
        )

    releases.sort(key=lambda release: release.release_datetime)
    return releases


_SAMPLE_FILENAME = "cpi_sample.csv"


def load_sample_cpi_data(
    *, columns: CPIColumnConfig | None = None, input_timezone: str | tzinfo | None = "US/Eastern"
) -> List[CPIRelease]:
    """Load the bundled CPI sample dataset for quick experimentation."""

    base_path = Path(__file__).resolve().parents[2]
    sample_path = base_path / "data" / _SAMPLE_FILENAME

    sample_columns = columns or CPIColumnConfig(
        timestamp=None,
        date="Release Date",
        time="Time",
        actual="Actual",
        expected="Forecast",
        previous="Previous",
        surprise=None,
    )

    return load_cpi_from_csv(sample_path, columns=sample_columns, input_timezone=input_timezone)
