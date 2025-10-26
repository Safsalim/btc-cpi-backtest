from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

    timestamp: str = "release_time"
    actual: str = "actual"
    expected: str = "expected"
    previous: Optional[str] = "previous"
    surprise: Optional[str] = "surprise"

    def required_columns(self) -> tuple[str, ...]:
        return (self.timestamp, self.actual, self.expected)


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
    if column_name and column_name in df.columns:
        return pd.to_numeric(df[column_name], errors="coerce")
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

    missing = [column for column in config.required_columns() if column not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_list}")

    timestamp_series = pd.to_datetime(df[config.timestamp], errors="coerce", utc=False)
    if timestamp_series.isna().any():
        raise ValueError("Invalid or missing release timestamps detected in CPI dataset")

    tzinfo_value = _resolve_timezone(input_timezone)
    if timestamp_series.dt.tz is None:
        tz_to_use = tzinfo_value or timezone.utc
        timestamp_series = timestamp_series.dt.tz_localize(tz_to_use)

    timestamp_series = timestamp_series.dt.tz_convert(timezone.utc)

    actual_series = pd.to_numeric(df[config.actual], errors="coerce")
    expected_series = pd.to_numeric(df[config.expected], errors="coerce")
    if actual_series.isna().any() or expected_series.isna().any():
        raise ValueError("Actual and expected CPI values must be numeric")

    previous_series = _optional_numeric_series(df, config.previous)
    surprise_series = _optional_numeric_series(df, config.surprise)

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
    *, columns: CPIColumnConfig | None = None, input_timezone: str | tzinfo | None = "UTC"
) -> List[CPIRelease]:
    """Load the bundled CPI sample dataset for quick experimentation."""

    base_path = Path(__file__).resolve().parents[2]
    sample_path = base_path / "data" / _SAMPLE_FILENAME
    return load_cpi_from_csv(sample_path, columns=columns, input_timezone=input_timezone)
