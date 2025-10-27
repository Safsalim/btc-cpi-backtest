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
    "FakeoutTimingStats",
    "SurpriseCategorySummary",
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
class FakeoutTimingStats:
    """Timing metrics associated with fake moves."""

    reaction_window: str
    evaluation_window: str
    count: int
    average_reversal_minutes: float | None
    distribution: dict[str, int]
    average_peak_return: float | None


@dataclass(frozen=True)
class SurpriseCategorySummary:
    """Aggregate metrics segmented by surprise type."""

    surprise_type: str
    count: int
    direction_counts: dict[str, int]
    direction_percentages: dict[str, float]
    average_initial_move: float | None
    fake_out_stats: dict[str, dict[str, FakeoutStats]]
    average_returns: dict[str, dict[str, float]]


@dataclass(frozen=True)
class FakeoutSummary:
    """Aggregate statistics summarizing the fake-out analysis."""

    event_count: int
    fake_out_stats: dict[str, dict[str, FakeoutStats]]
    average_returns: dict[str, dict[str, float]]
    surprise_correlations: dict[str, float]
    surprise_breakdown: dict[str, SurpriseCategorySummary]
    fakeout_timing: dict[str, dict[str, FakeoutTimingStats]]
    surprise_fakeout_correlations: dict[str, float]


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

    reaction_window_map = {label: delta for label, delta in reaction_windows}
    evaluation_window_map = {label: delta for label, delta in evaluation_windows}
    primary_reaction_label = reaction_windows[0][0] if reaction_windows else None

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

        surprise_type = _categorize_surprise(surprise, tolerance=tolerance)
        row["surprise_type"] = surprise_type

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

        if primary_reaction_label is not None:
            primary_return = reaction_returns.get(primary_reaction_label)
            row["initial_return_window"] = primary_reaction_label
            row["initial_direction"] = _determine_direction(primary_return, tolerance=tolerance)
            row["initial_move_magnitude"] = (
                float(abs(primary_return)) if primary_return is not None else math.nan
            )
        else:
            row["initial_return_window"] = None
            row["initial_direction"] = "unknown"
            row["initial_move_magnitude"] = math.nan

        for reaction_label, reaction_return in reaction_returns.items():
            reaction_delta = reaction_window_map.get(reaction_label)
            for evaluation_label, evaluation_return in evaluation_returns.items():
                evaluation_delta = evaluation_window_map.get(evaluation_label)
                fake_col = f"fake_{reaction_label}_{evaluation_label}"
                reversal_col = f"fake_reversal_minutes_{reaction_label}_{evaluation_label}"
                peak_col = f"fake_peak_return_{reaction_label}_{evaluation_label}"

                is_fake = _classify_fakeout(
                    reaction_return,
                    evaluation_return,
                    tolerance=tolerance,
                )
                row[fake_col] = is_fake

                row[reversal_col] = math.nan
                row[peak_col] = math.nan
                if is_fake and reaction_delta is not None and evaluation_delta is not None:
                    reversal_minutes, peak_return = _compute_fake_reversal_metrics(
                        price_series,
                        release_dt,
                        base_price,
                        reaction_return,
                        evaluation_delta,
                        tolerance=tolerance,
                    )
                    if not math.isnan(reversal_minutes):
                        row[reversal_col] = reversal_minutes
                    if not math.isnan(peak_return):
                        row[peak_col] = peak_return

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


def _categorize_surprise(surprise: float | None, *, tolerance: float) -> str | None:
    if surprise is None or (isinstance(surprise, float) and math.isnan(surprise)):
        return None
    if surprise > tolerance:
        return "positive"
    if surprise < -tolerance:
        return "negative"
    return "met"


def _determine_direction(value: float | None, *, tolerance: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    if abs(value) <= tolerance:
        return "flat"
    if value > 0:
        return "up"
    return "down"


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


def _compute_fake_reversal_metrics(
    price_series: pd.Series,
    release_dt: pd.Timestamp,
    base_price: float | None,
    reaction_return: float | None,
    evaluation_delta: timedelta,
    *,
    tolerance: float,
) -> tuple[float, float]:
    if (
        base_price is None
        or reaction_return is None
        or abs(reaction_return) <= tolerance
        or isinstance(reaction_return, float)
        and math.isnan(reaction_return)
    ):
        return math.nan, math.nan

    direction_sign = 1.0 if reaction_return > 0 else -1.0
    end_time = release_dt + evaluation_delta
    window = price_series.loc[release_dt:end_time]

    peak_magnitude = abs(reaction_return)
    reversal_timestamp: pd.Timestamp | None = None

    for timestamp, price in window.items():
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        relative_return = _compute_return(base_price, float(price))
        if relative_return is None or (isinstance(relative_return, float) and math.isnan(relative_return)):
            continue

        if direction_sign > 0:
            if relative_return > peak_magnitude:
                peak_magnitude = relative_return
            if relative_return <= tolerance:
                reversal_timestamp = timestamp
                break
        else:
            magnitude = abs(relative_return)
            if relative_return < -tolerance and magnitude > peak_magnitude:
                peak_magnitude = magnitude
            if relative_return >= -tolerance:
                reversal_timestamp = timestamp
                break

    if reversal_timestamp is None:
        return math.nan, peak_magnitude if peak_magnitude > 0 else math.nan

    duration_minutes = (reversal_timestamp - release_dt).total_seconds() / 60.0
    if duration_minutes < 0:
        duration_minutes = 0.0
    return float(duration_minutes), float(peak_magnitude)


def _summarize(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
    tolerance: float,
) -> FakeoutSummary:
    fake_out_stats = _aggregate_fakeout_stats(events, reaction_windows, evaluation_windows, tolerance)
    average_returns = _aggregate_average_returns(events, reaction_windows, evaluation_windows)
    surprise_correlations = _compute_return_correlations(events, evaluation_windows)
    timing_summary = _compute_fake_timing_summary(events, reaction_windows, evaluation_windows)
    surprise_breakdown = _compute_surprise_breakdown(
        events,
        reaction_windows,
        evaluation_windows,
        tolerance,
    )
    fakeout_probability_correlations = _compute_surprise_fakeout_correlations(
        events,
        reaction_windows,
        evaluation_windows,
    )

    summary = FakeoutSummary(
        event_count=len(events),
        fake_out_stats=fake_out_stats,
        average_returns=average_returns,
        surprise_correlations=surprise_correlations,
        surprise_breakdown=surprise_breakdown,
        fakeout_timing=timing_summary,
        surprise_fakeout_correlations=fakeout_probability_correlations,
    )
    return summary


def _aggregate_fakeout_stats(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
    tolerance: float,
) -> dict[str, dict[str, FakeoutStats]]:
    stats: dict[str, dict[str, FakeoutStats]] = {}
    for reaction_label, _ in reaction_windows:
        stats[reaction_label] = {}
        reaction_col = f"return_{reaction_label}"
        for evaluation_label, _ in evaluation_windows:
            evaluation_col = f"return_{evaluation_label}"
            fake_col = f"fake_{reaction_label}_{evaluation_label}"

            if events.empty or reaction_col not in events or evaluation_col not in events:
                stats[reaction_label][evaluation_label] = FakeoutStats(
                    reaction_window=reaction_label,
                    evaluation_window=evaluation_label,
                    fake_count=0,
                    total=0,
                )
                continue

            mask = (
                events[reaction_col].notna()
                & events[evaluation_col].notna()
                & (events[reaction_col].abs() > tolerance)
                & (events[evaluation_col].abs() > tolerance)
            )
            total = int(mask.sum())
            fake_count = 0
            if total:
                if fake_col in events:
                    fake_series = events.loc[mask, fake_col].fillna(False)
                    fake_count = int(fake_series.sum())
            stats[reaction_label][evaluation_label] = FakeoutStats(
                reaction_window=reaction_label,
                evaluation_window=evaluation_label,
                fake_count=fake_count,
                total=total,
            )
    return stats


def _aggregate_average_returns(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
) -> dict[str, dict[str, float]]:
    average_returns: dict[str, dict[str, float]] = {"reaction": {}, "evaluation": {}}
    for label, _ in reaction_windows:
        col = f"return_{label}"
        if events.empty or col not in events:
            average_returns["reaction"][label] = math.nan
        else:
            series = events[col].dropna()
            average_returns["reaction"][label] = float(series.mean()) if not series.empty else math.nan

    for label, _ in evaluation_windows:
        col = f"return_{label}"
        if events.empty or col not in events:
            average_returns["evaluation"][label] = math.nan
        else:
            series = events[col].dropna()
            average_returns["evaluation"][label] = float(series.mean()) if not series.empty else math.nan

    return average_returns


def _compute_return_correlations(
    events: pd.DataFrame,
    evaluation_windows: Iterable[tuple[str, timedelta]],
) -> dict[str, float]:
    surprise_correlations: dict[str, float] = {}
    if events.empty or "cpi_surprise" not in events:
        for label, _ in evaluation_windows:
            surprise_correlations[label] = math.nan
        return surprise_correlations

    for label, _ in evaluation_windows:
        col = f"return_{label}"
        if col not in events:
            surprise_correlations[label] = math.nan
            continue
        subset = events[["cpi_surprise", col]].dropna()
        if len(subset) >= 2:
            surprise_correlations[label] = float(subset["cpi_surprise"].corr(subset[col]))
        else:
            surprise_correlations[label] = math.nan
    return surprise_correlations


def _compute_fake_timing_summary(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
) -> dict[str, dict[str, FakeoutTimingStats]]:
    summary: dict[str, dict[str, FakeoutTimingStats]] = {}
    for reaction_label, _ in reaction_windows:
        summary[reaction_label] = {}
        for evaluation_label, _ in evaluation_windows:
            minutes_col = f"fake_reversal_minutes_{reaction_label}_{evaluation_label}"
            magnitude_col = f"fake_peak_return_{reaction_label}_{evaluation_label}"

            if events.empty or minutes_col not in events:
                distribution = _build_reversal_distribution([])
                summary[reaction_label][evaluation_label] = FakeoutTimingStats(
                    reaction_window=reaction_label,
                    evaluation_window=evaluation_label,
                    count=0,
                    average_reversal_minutes=math.nan,
                    distribution=distribution,
                    average_peak_return=math.nan,
                )
                continue

            mask = events[minutes_col].notna()
            durations = events.loc[mask, minutes_col].astype(float).tolist()
            magnitudes = []
            if magnitude_col in events:
                magnitudes = events.loc[mask & events[magnitude_col].notna(), magnitude_col].astype(float).tolist()

            count = len(durations)
            if count:
                avg_duration = float(sum(durations) / count)
                distribution = _build_reversal_distribution(durations)
            else:
                avg_duration = math.nan
                distribution = _build_reversal_distribution([])

            if magnitudes:
                avg_magnitude = float(sum(magnitudes) / len(magnitudes))
            else:
                avg_magnitude = math.nan

            summary[reaction_label][evaluation_label] = FakeoutTimingStats(
                reaction_window=reaction_label,
                evaluation_window=evaluation_label,
                count=count,
                average_reversal_minutes=avg_duration,
                distribution=distribution,
                average_peak_return=avg_magnitude,
            )
    return summary


def _build_reversal_distribution(durations: Sequence[float]) -> dict[str, int]:
    thresholds = (15.0, 30.0, 60.0)
    distribution = {f"<= {int(threshold)}m": 0 for threshold in thresholds}
    distribution["> 60m"] = 0

    for duration in durations:
        if duration <= thresholds[0]:
            distribution[f"<= {int(thresholds[0])}m"] += 1
        if duration <= thresholds[1]:
            distribution[f"<= {int(thresholds[1])}m"] += 1
        if duration <= thresholds[2]:
            distribution[f"<= {int(thresholds[2])}m"] += 1
        if duration > thresholds[2]:
            distribution["> 60m"] += 1
    return distribution


def _compute_surprise_breakdown(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
    tolerance: float,
) -> dict[str, SurpriseCategorySummary]:
    breakdown: dict[str, SurpriseCategorySummary] = {}
    if events.empty or "surprise_type" not in events:
        for label in ("positive", "negative", "met"):
            breakdown[label] = SurpriseCategorySummary(
                surprise_type=label,
                count=0,
                direction_counts={"up": 0, "down": 0, "flat": 0, "unknown": 0},
                direction_percentages={"up": math.nan, "down": math.nan, "flat": math.nan, "unknown": math.nan},
                average_initial_move=math.nan,
                fake_out_stats=_aggregate_fakeout_stats(
                    events.head(0), reaction_windows, evaluation_windows, tolerance
                ),
                average_returns=_aggregate_average_returns(events.head(0), reaction_windows, evaluation_windows),
            )
        return breakdown

    categories = ("positive", "negative", "met")
    for category in categories:
        subset = events.loc[events["surprise_type"] == category]
        count = len(subset)

        direction_counts = {key: 0 for key in ("up", "down", "flat", "unknown")}
        direction_percentages = {key: math.nan for key in direction_counts}
        average_initial_move = math.nan

        if count:
            for key in direction_counts:
                direction_counts[key] = int((subset["initial_direction"] == key).sum())
            direction_percentages = {
                key: direction_counts[key] / count if count else math.nan for key in direction_counts
            }
            magnitudes = subset["initial_move_magnitude"].dropna()
            average_initial_move = float(magnitudes.mean()) if not magnitudes.empty else math.nan

        fake_stats = _aggregate_fakeout_stats(subset, reaction_windows, evaluation_windows, tolerance)
        averages = _aggregate_average_returns(subset, reaction_windows, evaluation_windows)
        breakdown[category] = SurpriseCategorySummary(
            surprise_type=category,
            count=count,
            direction_counts=direction_counts,
            direction_percentages=direction_percentages,
            average_initial_move=average_initial_move,
            fake_out_stats=fake_stats,
            average_returns=averages,
        )

    return breakdown


def _compute_surprise_fakeout_correlations(
    events: pd.DataFrame,
    reaction_windows: Iterable[tuple[str, timedelta]],
    evaluation_windows: Iterable[tuple[str, timedelta]],
) -> dict[str, float]:
    correlations: dict[str, float] = {}
    if events.empty or "cpi_surprise" not in events:
        for _, label in enumerate(evaluation_windows):
            correlations[label[0]] = math.nan
        return correlations

    if not reaction_windows:
        return correlations

    primary_reaction_label = reaction_windows[0][0]
    for evaluation_label, _ in evaluation_windows:
        fake_col = f"fake_{primary_reaction_label}_{evaluation_label}"
        if fake_col not in events:
            correlations[evaluation_label] = math.nan
            continue

        subset = events[["cpi_surprise", fake_col]].dropna()
        if subset.empty:
            correlations[evaluation_label] = math.nan
            continue

        subset = subset.copy()
        subset = subset[subset[fake_col].isin([True, False])]
        if subset.empty:
            correlations[evaluation_label] = math.nan
            continue

        subset["abs_surprise"] = subset["cpi_surprise"].abs()
        subset["fake_numeric"] = subset[fake_col].astype(int)
        if subset["fake_numeric"].nunique() <= 1:
            correlations[evaluation_label] = math.nan
            continue

        correlations[evaluation_label] = float(subset["abs_surprise"].corr(subset["fake_numeric"]))

    return correlations
