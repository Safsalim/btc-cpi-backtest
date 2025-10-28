"""Analysis utilities for classifying CPI-driven price moves."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Sequence

import logging
import math

import numpy as np
import pandas as pd

from .cpi_loader import CPIRelease

__all__ = [
    "FakeoutConfig",
    "FakeoutStats",
    "FakeoutSummary",
    "FakeoutAnalysisResult",
    "analyze_fakeouts",
]

logger = logging.getLogger(__name__)


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
    fake_probability_by_surprise: list[dict[str, float]] = field(default_factory=list)
    fake_duration_stats: dict[str, float] = field(default_factory=dict)


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
    price_series = price_series[~price_series.index.duplicated(keep="last")]
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
        initial_reaction_label: str | None = None
        initial_reaction_return: float | None = None
        for label, delta in reaction_windows:
            end_price = _price_at(price_series, release_dt + delta)
            computed_return = _compute_return(base_price, end_price)
            reaction_returns[label] = computed_return
            row[f"return_{label}"] = computed_return
            if (
                initial_reaction_return is None
                and computed_return is not None
                and abs(computed_return) > tolerance
            ):
                initial_reaction_label = label
                initial_reaction_return = computed_return

        evaluation_returns: dict[str, float | None] = {}
        for label, delta in evaluation_windows:
            end_price = _price_at(price_series, release_dt + delta)
            computed_return = _compute_return(base_price, end_price)
            evaluation_returns[label] = computed_return
            row[f"return_{label}"] = computed_return

        fake_any: bool | None = None
        has_false = False
        for reaction_label, reaction_return in reaction_returns.items():
            for evaluation_label, evaluation_return in evaluation_returns.items():
                fake_col = f"fake_{reaction_label}_{evaluation_label}"
                classification = _classify_fakeout(
                    reaction_return,
                    evaluation_return,
                    tolerance=tolerance,
                )
                row[fake_col] = classification
                if classification is True:
                    fake_any = True
                elif classification is False and fake_any is not True:
                    has_false = True
        if fake_any is not True and has_false:
            fake_any = False

        row["initial_reaction_window"] = initial_reaction_label
        row["initial_reaction_return"] = initial_reaction_return
        row["initial_reaction_return_pct"] = (
            initial_reaction_return * 100.0 if initial_reaction_return is not None else None
        )
        row["fake_any"] = fake_any

        rows.append(row)

    events_df = pd.DataFrame(rows)
    if events_df.empty:
        summary = _summarize(events_df, reaction_windows, evaluation_windows, tolerance)
        return FakeoutAnalysisResult(events=events_df, summary=summary)

    events_df.sort_values("release_datetime", inplace=True)
    events_df.reset_index(drop=True, inplace=True)

    events_df["release_datetime"] = pd.to_datetime(events_df["release_datetime"], utc=True)
    events_df["cpi_surprise"] = pd.to_numeric(events_df["cpi_surprise"], errors="coerce")
    events_df["initial_reaction_return"] = pd.to_numeric(
        events_df["initial_reaction_return"], errors="coerce"
    )
    events_df["initial_reaction_return_pct"] = events_df["initial_reaction_return"] * 100.0

    if "fake_any" in events_df.columns:
        events_df["fake_any"] = pd.Series(events_df["fake_any"], dtype="boolean")
    else:
        events_df["fake_any"] = pd.Series(pd.array([pd.NA] * len(events_df), dtype="boolean"))

    if not price_series.empty:
        duration_series = _compute_fake_durations(
            events_df,
            price_series,
            evaluation_windows,
            [label for label, _ in reaction_windows],
            tolerance=tolerance,
        )
        events_df["fake_duration_minutes"] = duration_series
    else:
        events_df["fake_duration_minutes"] = pd.Series(dtype="Float64")

    for label, _ in reaction_windows + evaluation_windows:
        column_name = f"return_{label}"
        if column_name in events_df.columns:
            events_df[column_name] = pd.to_numeric(events_df[column_name], errors="coerce")

    summary = _summarize(events_df, reaction_windows, evaluation_windows, tolerance)
    logger.debug(
        "Completed fake-out analysis for %d releases (fake_any true=%d, false=%d)",
        len(events_df),
        int((events_df["fake_any"] == True).sum()),  # noqa: E712
        int((events_df["fake_any"] == False).sum()),  # noqa: E712
    )
    return FakeoutAnalysisResult(events=events_df, summary=summary)


def _price_at(series: pd.Series, timestamp: pd.Timestamp | None) -> float | None:
    if timestamp is None:
        return None

    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(series.index.tz)
    else:
        timestamp = timestamp.tz_convert(series.index.tz)

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


def _compute_fake_durations(
    events: pd.DataFrame,
    price_series: pd.Series,
    evaluation_windows: Sequence[tuple[str, timedelta]],
    reaction_labels: Sequence[str],
    *,
    tolerance: float,
) -> pd.Series:
    if events.empty:
        return pd.Series(dtype="Float64")

    max_eval_delta = max((delta for _, delta in evaluation_windows), default=timedelta(0))
    durations: list[float | None] = []

    series_tz = price_series.index.tz

    for idx, row in events.iterrows():
        fake_any = row.get("fake_any")
        if pd.isna(fake_any) or not bool(fake_any):
            durations.append(math.nan)
            continue

        release_value = row.get("release_datetime")
        if pd.isna(release_value):
            durations.append(math.nan)
            continue

        base_price = row.get("base_price")
        if base_price is None or (isinstance(base_price, float) and math.isnan(base_price)):
            durations.append(math.nan)
            continue
        base_price = float(base_price)
        if base_price == 0.0:
            durations.append(math.nan)
            continue

        release_ts = pd.Timestamp(release_value)
        if release_ts.tzinfo is None:
            release_ts = release_ts.tz_localize(series_tz)
        else:
            release_ts = release_ts.tz_convert(series_tz)

        initial_return = row.get("initial_reaction_return")
        if initial_return is None or (isinstance(initial_return, float) and math.isnan(initial_return)):
            for label in reaction_labels:
                value = row.get(f"return_{label}")
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                if abs(float(value)) <= tolerance:
                    continue
                initial_return = float(value)
                break

        if initial_return is None:
            durations.append(math.nan)
            continue

        search_end = release_ts + max_eval_delta
        window_slice = price_series.loc[release_ts:search_end]
        if release_ts not in window_slice.index:
            base_at_release = _price_at(price_series, release_ts)
            if base_at_release is not None:
                window_slice.loc[release_ts] = base_at_release
                window_slice = window_slice.sort_index()
        window_slice = window_slice[window_slice.index > release_ts]
        if window_slice.empty:
            durations.append(math.nan)
            continue

        relative_returns = (window_slice / base_price) - 1.0
        if initial_return > 0:
            reversal_points = relative_returns[relative_returns <= 0]
        else:
            reversal_points = relative_returns[relative_returns >= 0]

        if reversal_points.empty:
            durations.append(math.nan)
            continue

        reversal_ts = reversal_points.index[0]
        duration_minutes = max(0.0, (reversal_ts - release_ts).total_seconds() / 60.0)
        durations.append(duration_minutes)

    duration_series = pd.Series(durations, index=events.index, dtype="Float64")
    logger.debug(
        "Computed fake move durations for %d releases", int(duration_series.notna().sum())
    )
    return duration_series


def _compute_fake_probability_by_surprise(events: pd.DataFrame) -> list[dict[str, float]]:
    if events.empty:
        return []
    if "cpi_surprise" not in events.columns or "fake_any" not in events.columns:
        return []

    candidate = events.dropna(subset=["cpi_surprise", "fake_any"])
    if candidate.empty:
        return []

    surprise_values = candidate["cpi_surprise"].astype(float).to_numpy()
    if surprise_values.size == 0:
        return []

    min_surprise = float(np.min(surprise_values))
    max_surprise = float(np.max(surprise_values))

    if math.isclose(min_surprise, max_surprise):
        bins = np.array([min_surprise - 0.05, max_surprise + 0.05])
    else:
        bin_count = min(12, max(4, int(math.sqrt(candidate.shape[0]))))
        bins = np.linspace(min_surprise, max_surprise, bin_count + 1)
        bins = np.unique(bins)
        if bins.size < 2:
            bins = np.array([min_surprise - 0.05, max_surprise + 0.05])

    candidate = candidate.copy()
    candidate["surprise_bin"] = pd.cut(candidate["cpi_surprise"], bins=bins, include_lowest=True)
    candidate["fake_flag"] = candidate["fake_any"].astype(int)

    grouped = (
        candidate.groupby("surprise_bin", observed=True)
        .agg(
            fake_rate=("fake_flag", "mean"),
            sample_size=("fake_flag", "size"),
            avg_surprise=("cpi_surprise", "mean"),
        )
        .dropna()
    )

    if grouped.empty:
        return []

    records: list[dict[str, float]] = []
    for interval, row in grouped.iterrows():
        label = str(interval)
        left = float(interval.left) if getattr(interval, "left", None) is not None else float("nan")
        right = float(interval.right) if getattr(interval, "right", None) is not None else float("nan")
        records.append(
            {
                "label": label,
                "lower": left,
                "upper": right,
                "avg_surprise": float(row["avg_surprise"]),
                "fake_rate": float(row["fake_rate"]),
                "fake_rate_pct": float(row["fake_rate"] * 100.0),
                "sample_size": int(row["sample_size"]),
            }
        )

    logger.debug(
        "Computed fake probability across %d surprise bins", len(records)
    )
    return records


def _compute_duration_stats(events: pd.DataFrame) -> dict[str, float]:
    if "fake_duration_minutes" not in events.columns:
        return {}
    series = events["fake_duration_minutes"].dropna()
    if series.empty:
        return {}
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


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

            if events.empty or fake_col not in events.columns:
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

    fake_probability = _compute_fake_probability_by_surprise(events)
    duration_stats = _compute_duration_stats(events)

    summary = FakeoutSummary(
        event_count=len(events),
        fake_out_stats=fake_out_stats,
        average_returns=average_returns,
        surprise_correlations=surprise_correlations,
        fake_probability_by_surprise=fake_probability,
        fake_duration_stats=duration_stats,
    )
    return summary
