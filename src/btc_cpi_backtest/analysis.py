"""Analysis utilities for classifying CPI-driven price moves."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Sequence

import math
import pandas as pd

from .cpi_loader import CPIRelease

__all__ = [
    "FakeoutConfig",
    "FakeoutStats",
    "FakeoutSummary",
    "FakeoutAnalysisResult",
    "analyze_fakeouts",
]


@dataclass(frozen=True)
class FakeoutConfig:
    """Configuration payload describing the analysis windows."""

    reaction_windows: tuple[tuple[str, timedelta], ...] = (
        ("5m", timedelta(minutes=5)),
        ("15m", timedelta(minutes=15)),
        ("30m", timedelta(minutes=30)),
    )
    evaluation_windows: tuple[tuple[str, timedelta], ...] = (
        ("1h", timedelta(hours=1)),
        ("2h", timedelta(hours=2)),
        ("4h", timedelta(hours=4)),
    )
    tolerance: float = 1e-6


@dataclass(frozen=True)
class FakeoutStats:
    """Fake-out statistics for a specific reaction/evaluation pairing."""

    reaction_window: str
    evaluation_window: str
    fake_count: int
    total: int

    @property
    def fake_ratio(self) -> float | None:
        if self.total == 0:
            return None
        return self.fake_count / self.total


@dataclass(frozen=True)
class FakeoutSummary:
    """Aggregate statistics summarizing the fake-out analysis."""

    event_count: int
    fake_out_stats: dict[str, dict[str, FakeoutStats]]
    average_returns: dict[str, dict[str, float]]
    surprise_correlations: dict[str, float]


@dataclass(frozen=True)
class FakeoutAnalysisResult:
    """Container for raw per-event data and summary statistics."""

    events: pd.DataFrame
    summary: FakeoutSummary


def analyze_fakeouts(
    releases: Sequence[CPIRelease],
    price_series: pd.Series,
    *,
    config: FakeoutConfig | None = None,
) -> FakeoutAnalysisResult:
    """Analyze CPI releases to identify fake moves in BTC price action.

    Parameters
    ----------
    releases:
        Ordered CPI release metadata.
    price_series:
        Time-indexed close prices for BTC, with a timezone-aware index.
    config:
        Analysis configuration. Defaults to :class:`FakeoutConfig` if omitted.

    Returns
    -------
    FakeoutAnalysisResult
        Container with per-release metrics and aggregate statistics.
    """

    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series")

    price_series = price_series.sort_index()
    if price_series.empty:
        raise ValueError("Price series is empty; cannot perform analysis")

    if price_series.index.tz is None:
        raise ValueError("Price series index must be timezone-aware")

    active_config = config or FakeoutConfig()
    reaction_windows = active_config.reaction_windows
    evaluation_windows = active_config.evaluation_windows
    tolerance = active_config.tolerance

    rows: list[dict[str, object]] = []
    for release in releases:
        release_dt = release.release_datetime
        base_price = _price_at(price_series, release_dt)

        surprise = release.surprise
        if surprise is None:
            surprise = release.actual - release.expected

        row: dict[str, object] = {
            "release_datetime": release_dt,
            "cpi_actual": release.actual,
            "cpi_expected": release.expected,
            "cpi_surprise": surprise,
            "base_price": base_price,
        }

        reaction_returns: dict[str, float | None] = {}
        for label, delta in reaction_windows:
            end_price = _price_at(price_series, release_dt + delta)
            reaction_returns[label] = _compute_return(base_price, end_price)
            row[f"return_{label}"] = reaction_returns[label]

        evaluation_returns: dict[str, float | None] = {}
        for label, delta in evaluation_windows:
            end_price = _price_at(price_series, release_dt + delta)
            evaluation_returns[label] = _compute_return(base_price, end_price)
            row[f"return_{label}"] = evaluation_returns[label]

        for reaction_label, reaction_return in reaction_returns.items():
            for evaluation_label, evaluation_return in evaluation_returns.items():
                fake_col = f"fake_{reaction_label}_{evaluation_label}"
                row[fake_col] = _classify_fakeout(
                    reaction_return,
                    evaluation_return,
                    tolerance=tolerance,
                )

        rows.append(row)

    events_df = pd.DataFrame(rows)
    if not events_df.empty:
        events_df.sort_values("release_datetime", inplace=True)
        events_df.reset_index(drop=True, inplace=True)

    summary = _summarize(events_df, reaction_windows, evaluation_windows, tolerance)
    return FakeoutAnalysisResult(events=events_df, summary=summary)


def _price_at(series: pd.Series, timestamp: pd.Timestamp | None) -> float | None:
    if timestamp is None:
        return None

    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(series.index.tz)

    try:
        value = series.asof(timestamp)
    except AttributeError:  # pragma: no cover - for older pandas versions
        filtered = series.loc[:timestamp]
        value = filtered.iloc[-1] if not filtered.empty else math.nan

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _compute_return(base_price: float | None, target_price: float | None) -> float | None:
    if base_price is None or target_price is None:
        return None
    if base_price == 0:
        return None
    return (target_price / base_price) - 1.0


def _classify_fakeout(
    reaction_return: float | None,
    evaluation_return: float | None,
    *,
    tolerance: float,
) -> bool | None:
    if reaction_return is None or evaluation_return is None:
        return None
    if abs(reaction_return) <= tolerance or abs(evaluation_return) <= tolerance:
        return False
    return math.copysign(1.0, reaction_return) != math.copysign(1.0, evaluation_return)


def _summarize(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
    tolerance: float,
) -> FakeoutSummary:
    fake_out_stats: dict[str, dict[str, FakeoutStats]] = {}
    for reaction_label, _ in reaction_windows:
        fake_out_stats[reaction_label] = {}
        reaction_col = f"return_{reaction_label}"
        for evaluation_label, _ in evaluation_windows:
            evaluation_col = f"return_{evaluation_label}"
            fake_col = f"fake_{reaction_label}_{evaluation_label}"

            if events.empty:
                stats = FakeoutStats(
                    reaction_window=reaction_label,
                    evaluation_window=evaluation_label,
                    fake_count=0,
                    total=0,
                )
                fake_out_stats[reaction_label][evaluation_label] = stats
                continue

            mask = (
                events[reaction_col].notna()
                & events[evaluation_col].notna()
                & (events[reaction_col].abs() > tolerance)
                & (events[evaluation_col].abs() > tolerance)
            )
            total = int(mask.sum())
            fake_series = events.loc[mask, fake_col].fillna(False)
            fake_count = int(fake_series.sum())
            stats = FakeoutStats(
                reaction_window=reaction_label,
                evaluation_window=evaluation_label,
                fake_count=fake_count,
                total=total,
            )
            fake_out_stats[reaction_label][evaluation_label] = stats

    average_returns: dict[str, dict[str, float]] = {"reaction": {}, "evaluation": {}}
    for label, _ in reaction_windows:
        col = f"return_{label}"
        if events.empty:
            average_returns["reaction"][label] = math.nan
        else:
            average_returns["reaction"][label] = float(events[col].dropna().mean())

    for label, _ in evaluation_windows:
        col = f"return_{label}"
        if events.empty:
            average_returns["evaluation"][label] = math.nan
        else:
            average_returns["evaluation"][label] = float(events[col].dropna().mean())

    surprise_correlations: dict[str, float] = {}
    if events.empty:
        for label, _ in evaluation_windows:
            surprise_correlations[label] = math.nan
    else:
        for label, _ in evaluation_windows:
            col = f"return_{label}"
            subset = events[["cpi_surprise", col]].dropna()
            if len(subset) >= 2:
                surprise_correlations[label] = float(subset["cpi_surprise"].corr(subset[col]))
            else:
                surprise_correlations[label] = math.nan

    summary = FakeoutSummary(
        event_count=len(events),
        fake_out_stats=fake_out_stats,
        average_returns=average_returns,
        surprise_correlations=surprise_correlations,
    )
    return summary
