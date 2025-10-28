from __future__ import annotations

import html
import json
import logging
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

from .analysis import (
    FakeoutAnalysisResult,
    FakeoutConfig,
    _classify_fakeout,
    _compute_fake_probability_by_surprise,
)
from .cpi_loader import CPIRelease

DEFAULT_DASHBOARD_PATH = Path("data") / "analysis_dashboard.html"

__all__ = ["DEFAULT_DASHBOARD_PATH", "render_dashboard_html"]

logger = logging.getLogger(__name__)


def _aggregate_flags(values: Iterable[Any]) -> bool | None:
    has_false = False
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if bool(value):
            return True
        has_false = True
    if has_false:
        return False
    return None


def _categorize_surprise(value: float | None, *, neutral_band: float) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return "Unknown"
    if value > neutral_band:
        return "Positive"
    if value < -neutral_band:
        return "Negative"
    return "Inline"


def _timedelta_minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)


def _clean_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64", index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": np.nan}, regex=False)
    cleaned = cleaned.replace(
        to_replace=r"(?i)^(nan|none|n/?a|n\\.a\\.?|--|—)$",
        value=np.nan,
        regex=True,
    )
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_boolean_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(pd.array([], dtype="boolean"), index=series.index)

    values: list[object] = []
    for value in series:
        if pd.isna(value):
            values.append(pd.NA)
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "t", "1", "yes"}:
                values.append(True)
                continue
            if normalized in {"false", "f", "0", "no"}:
                values.append(False)
                continue
        try:
            values.append(bool(value))
        except Exception:  # pragma: no cover - defensive guard
            values.append(pd.NA)
    return pd.Series(pd.array(values, dtype="boolean"), index=series.index)


def _price_at(series: pd.Series, timestamp: pd.Timestamp | None) -> float | None:
    if timestamp is None:
        return None

    ts = pd.Timestamp(timestamp)
    index_tz = getattr(series.index, "tz", None)

    if ts.tzinfo is None:
        if index_tz is not None:
            ts = ts.tz_localize(index_tz)
    else:
        if index_tz is not None:
            ts = ts.tz_convert(index_tz)
        else:
            ts = ts.tz_localize(None)

    try:
        value = series.asof(ts)
    except AttributeError:  # pragma: no cover - for older pandas versions
        filtered = series.loc[:ts]
        value = filtered.iloc[-1] if not filtered.empty else math.nan

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


class DashboardBuilder:
    """Constructs an interactive Plotly dashboard for fake-out analysis."""

    def __init__(
        self,
        *,
        result: FakeoutAnalysisResult,
        releases: Sequence[CPIRelease],
        price_series: pd.Series,
        config: FakeoutConfig,
        neutral_surprise_band: float = 0.05,
        price_window_before: timedelta = timedelta(hours=6),
        price_window_after: timedelta = timedelta(hours=12),
        plotly_js_mode: Literal["cdn", "inline"] = "cdn",
        plotly_cdn_url: str | None = None,
    ) -> None:
        self._result = result
        self._releases = list(releases)
        self._price_series = price_series
        self._config = config
        self._neutral_surprise_band = neutral_surprise_band
        self._price_window_before = price_window_before
        self._price_window_after = price_window_after

        normalized_mode = plotly_js_mode.lower()
        if normalized_mode not in {"cdn", "inline"}:
            raise ValueError(f"Unsupported Plotly JS mode: {plotly_js_mode!r}")
        self._plotly_js_mode = normalized_mode
        self._plotly_cdn_url = plotly_cdn_url

        self._events = result.events.copy()
        self._reaction_windows = list(config.reaction_windows)
        self._evaluation_windows = list(config.evaluation_windows)
        self._reaction_labels = [label for label, _ in self._reaction_windows]
        self._evaluation_labels = [label for label, _ in self._evaluation_windows]
        self._primary_reaction = self._reaction_labels[0] if self._reaction_labels else None
        self._primary_evaluation = self._evaluation_labels[0] if self._evaluation_labels else None

        self._processed = False
        self._figures: Dict[str, dict[str, Any]] = {}
        self._figure_messages: Dict[str, str] = {}
        self._table_rows: list[dict[str, Any]] = []
        self._price_windows: dict[int, dict[str, Any]] = {}
        self._summary_cards: list[dict[str, str]] = []

    def build_html(self) -> str:
        if not self._processed:
            self._prepare_events()
            self._build_summary_cards()
            self._build_figures()
            self._prepare_table_and_price_windows()
            self._processed = True
        return self._render_template()

    def _serialize_figure(self, figure: go.Figure) -> dict[str, Any]:
        try:
            serialized = json.loads(pio.to_json(figure, validate=False))
        except Exception as exc:  # pragma: no cover - defensive guard around Plotly internals
            raise RuntimeError("Failed to serialize Plotly figure; try switching to the mpld3 backend or inline Plotly mode.") from exc
        return serialized

    def _prepare_events(self) -> None:
        events = self._events
        logger.debug("Preparing dashboard events dataset with %d rows", len(events))
        if not events.empty:
            events = events.copy()

        if "release_datetime" in events.columns:
            events["release_datetime"] = pd.to_datetime(events["release_datetime"], utc=True)
            events.sort_values("release_datetime", inplace=True)
            events.reset_index(drop=True, inplace=True)

        price_series = self._price_series
        if not price_series.empty:
            price_series = price_series.sort_index()
            price_series = price_series[~price_series.index.duplicated(keep="last")]
        self._price_series = price_series
        logger.debug(
            "Price series prepared with %d points (tz=%s)",
            len(price_series),
            getattr(price_series.index, "tz", None),
        )

        if "cpi_surprise" in events.columns:
            events["cpi_surprise"] = _normalize_numeric_series(events["cpi_surprise"])
        else:
            events["cpi_surprise"] = pd.Series(
                [math.nan] * len(events),
                index=events.index,
                dtype="float64",
            )

        if "cpi_actual" in events.columns:
            events["cpi_actual"] = _normalize_numeric_series(events["cpi_actual"])
        if "cpi_expected" in events.columns:
            events["cpi_expected"] = _normalize_numeric_series(events["cpi_expected"])
        if "cpi_actual" in events.columns and "cpi_expected" in events.columns:
            actual_series = events["cpi_actual"]
            expected_series = events["cpi_expected"]
            fallback_surprise = actual_series - expected_series
            missing_mask = events["cpi_surprise"].isna()
            if missing_mask.all() and len(events) > 0:
                events["cpi_surprise"] = fallback_surprise
                logger.debug(
                    "Filled CPI surprise for all %d releases using actual minus expected values",
                    len(events),
                )
            elif missing_mask.any():
                events.loc[missing_mask, "cpi_surprise"] = fallback_surprise.loc[missing_mask]
                logger.debug(
                    "Filled CPI surprise for %d releases using actual minus expected values",
                    int(missing_mask.sum()),
                )

        if "base_price" in events.columns:
            events["base_price"] = _normalize_numeric_series(events["base_price"])
        else:
            events["base_price"] = pd.Series(
                [math.nan] * len(events),
                index=events.index,
                dtype="float64",
            )

        if "fake_duration_minutes" in events.columns:
            events["fake_duration_minutes"] = _normalize_numeric_series(
                events["fake_duration_minutes"]
            ).astype("Float64")
        else:
            events["fake_duration_minutes"] = pd.Series(
                [math.nan] * len(events),
                index=events.index,
                dtype="Float64",
            )

        if "initial_reaction_return" in events.columns:
            events["initial_reaction_return"] = _normalize_numeric_series(
                events["initial_reaction_return"]
            ).astype("Float64")
        else:
            events["initial_reaction_return"] = pd.Series(
                [math.nan] * len(events),
                index=events.index,
                dtype="Float64",
            )

        if "initial_reaction_return_pct" in events.columns:
            events["initial_reaction_return_pct"] = _normalize_numeric_series(
                events["initial_reaction_return_pct"]
            ).astype("Float64")
        else:
            fallback_pct: pd.Series | None = None
            if self._reaction_labels:
                primary_reaction = self._reaction_labels[0]
                pct_candidate = f"return_{primary_reaction}_pct"
                base_candidate = f"return_{primary_reaction}"
                if pct_candidate in events.columns:
                    fallback_pct = _normalize_numeric_series(events[pct_candidate])
                elif base_candidate in events.columns:
                    fallback_pct = _normalize_numeric_series(events[base_candidate]) * 100.0
            if fallback_pct is not None:
                events["initial_reaction_return_pct"] = fallback_pct.astype("Float64")
            else:
                events["initial_reaction_return_pct"] = pd.Series(
                    [math.nan] * len(events),
                    index=events.index,
                    dtype="Float64",
                )

        combined_windows = self._reaction_windows + self._evaluation_windows
        for label, _ in combined_windows:
            column_name = f"return_{label}"
            if column_name in events.columns:
                events[column_name] = _normalize_numeric_series(events[column_name]).astype("Float64")
            else:
                events[column_name] = pd.Series(
                    [math.nan] * len(events),
                    index=events.index,
                    dtype="Float64",
                )

        tolerance = self._config.tolerance
        if not price_series.empty and "release_datetime" in events.columns:
            for idx in range(len(events)):
                release_dt = events.at[idx, "release_datetime"]
                if pd.isna(release_dt):
                    continue
                release_dt = pd.Timestamp(release_dt)
                if release_dt.tzinfo is None:
                    release_dt = release_dt.tz_localize("UTC")
                else:
                    release_dt = release_dt.tz_convert("UTC")
                base_price = events.at[idx, "base_price"]
                if pd.isna(base_price):
                    base_price = _price_at(price_series, release_dt)
                    if base_price is None:
                        logger.debug("Missing base price for release at %s; skipping price backfill", release_dt)
                        continue
                    events.at[idx, "base_price"] = base_price
                else:
                    base_price = float(base_price)
                if base_price is None or base_price == 0.0:
                    continue
                for label, delta in combined_windows:
                    column_name = f"return_{label}"
                    value = events.at[idx, column_name]
                    if pd.isna(value):
                        end_price = _price_at(price_series, release_dt + delta)
                        if end_price is None:
                            continue
                        events.at[idx, column_name] = (end_price / base_price) - 1.0

        for reaction_label in self._reaction_labels:
            for evaluation_label in self._evaluation_labels:
                fake_column = f"fake_{reaction_label}_{evaluation_label}"
                if fake_column in events.columns:
                    events[fake_column] = _normalize_boolean_series(events[fake_column])
                else:
                    events[fake_column] = pd.Series(
                        pd.array([pd.NA] * len(events), dtype="boolean"),
                        index=events.index,
                    )

        if combined_windows and not events.empty:
            for idx in range(len(events)):
                for reaction_label in self._reaction_labels:
                    reaction_column = f"return_{reaction_label}"
                    reaction_value = events.at[idx, reaction_column]
                    if pd.isna(reaction_value):
                        continue
                    for evaluation_label in self._evaluation_labels:
                        evaluation_column = f"return_{evaluation_label}"
                        evaluation_value = events.at[idx, evaluation_column]
                        if pd.isna(evaluation_value):
                            continue
                        fake_column = f"fake_{reaction_label}_{evaluation_label}"
                        if pd.isna(events.at[idx, fake_column]):
                            events.at[idx, fake_column] = _classify_fakeout(
                                float(reaction_value),
                                float(evaluation_value),
                                tolerance=tolerance,
                            )

        for label in self._reaction_labels + self._evaluation_labels:
            base_column = f"return_{label}"
            pct_column = f"return_{label}_pct"
            if base_column in events.columns:
                events[pct_column] = events[base_column] * 100.0

        fake_column_map: dict[tuple[str, str], str] = {}
        for reaction_label in self._reaction_labels:
            for evaluation_label in self._evaluation_labels:
                column_name = f"fake_{reaction_label}_{evaluation_label}"
                if column_name in events.columns:
                    fake_column_map[(reaction_label, evaluation_label)] = column_name

        for evaluation_label in self._evaluation_labels:
            columns = [
                fake_column_map[(reaction_label, evaluation_label)]
                for reaction_label in self._reaction_labels
                if (reaction_label, evaluation_label) in fake_column_map
            ]
            if not columns:
                continue
            aggregated = events[columns].apply(_aggregate_flags, axis=1)
            events[f"fake_by_evaluation_{evaluation_label}"] = pd.Series(
                pd.array(aggregated, dtype="boolean"),
                index=events.index,
            )

        for reaction_label in self._reaction_labels:
            columns = [
                fake_column_map[(reaction_label, evaluation_label)]
                for evaluation_label in self._evaluation_labels
                if (reaction_label, evaluation_label) in fake_column_map
            ]
            if not columns:
                continue
            aggregated = events[columns].apply(_aggregate_flags, axis=1)
            events[f"fake_by_reaction_{reaction_label}"] = pd.Series(
                pd.array(aggregated, dtype="boolean"),
                index=events.index,
            )

        fake_columns = list(fake_column_map.values())
        if fake_columns:
            aggregated = events[fake_columns].apply(_aggregate_flags, axis=1)
            events["fake_any"] = pd.Series(
                pd.array(aggregated, dtype="boolean"),
                index=events.index,
            )
        else:
            events["fake_any"] = pd.Series(
                pd.array([pd.NA] * len(events), dtype="boolean"),
                index=events.index,
            )

        recompute_durations = (
            "fake_duration_minutes" not in events.columns
            or events["fake_duration_minutes"].dropna().empty
        )
        if recompute_durations:
            logger.debug("Recomputing fake-out durations for %d releases", len(events))
            durations: list[int | None] = []
            if not price_series.empty and "release_datetime" in events.columns:
                max_eval_delta = max((delta for _, delta in self._evaluation_windows), default=timedelta(0))
                search_delta = max(self._price_window_after, max_eval_delta)
                for idx in range(len(events)):
                    fake_any_value = events.at[idx, "fake_any"]
                    if pd.isna(fake_any_value) or not bool(fake_any_value):
                        durations.append(None)
                        continue
                    release_dt = events.at[idx, "release_datetime"]
                    if pd.isna(release_dt):
                        durations.append(None)
                        continue
                    release_dt = pd.Timestamp(release_dt)
                    if release_dt.tzinfo is None:
                        release_dt = release_dt.tz_localize("UTC")
                    else:
                        release_dt = release_dt.tz_convert("UTC")
                    base_price = events.at[idx, "base_price"]
                    if pd.isna(base_price):
                        durations.append(None)
                        continue
                    base_price = float(base_price)
                    initial_return = None
                    for reaction_label in self._reaction_labels:
                        reaction_value = events.at[idx, f"return_{reaction_label}"]
                        if pd.isna(reaction_value):
                            continue
                        if abs(float(reaction_value)) <= tolerance:
                            continue
                        initial_return = float(reaction_value)
                        break
                    if initial_return is None:
                        durations.append(None)
                        continue
                    window_end = release_dt + search_delta
                    window_slice = price_series.loc[release_dt:window_end]
                    window_slice = window_slice[window_slice.index > release_dt]
                    if window_slice.empty:
                        durations.append(None)
                        continue
                    relative_returns = (window_slice / base_price) - 1.0
                    if initial_return > 0:
                        reversal_points = relative_returns[relative_returns <= 0]
                    else:
                        reversal_points = relative_returns[relative_returns >= 0]
                    if reversal_points.empty:
                        durations.append(None)
                        continue
                    reversal_ts = reversal_points.index[0]
                    minutes = max(0, _timedelta_minutes(reversal_ts - release_dt))
                    durations.append(minutes)
            if durations:
                duration_series = pd.Series(durations, index=events.index, dtype="Float64")
                events["fake_duration_minutes"] = duration_series
                logger.debug(
                    "Computed price-based fake durations for %d of %d releases",
                    int(duration_series.notna().sum()),
                    len(events),
                )
            else:
                events["fake_duration_minutes"] = pd.Series(dtype="Float64")
                logger.debug("No reversal durations could be derived from price history")
        else:
            existing = events["fake_duration_minutes"].dropna()
            logger.debug(
                "Using precomputed fake-out durations for %d releases",
                int(existing.size),
            )

        events["surprise_type"] = events["cpi_surprise"].apply(
            lambda value: _categorize_surprise(value, neutral_band=self._neutral_surprise_band)
        )

        columns_list = list(events.columns)
        logger.debug("Dashboard events columns after preparation: %s", columns_list)

        def _extract_sample(series: pd.Series | None) -> list[Any]:
            if series is None:
                return []
            values: list[Any] = []
            for raw_value in series.head(5):
                if pd.isna(raw_value):
                    values.append(None)
                    continue
                if isinstance(raw_value, (np.bool_, bool)):
                    values.append(bool(raw_value))
                    continue
                if isinstance(raw_value, (int, float, np.number)):
                    values.append(float(raw_value))
                    continue
                values.append(raw_value)
            return values

        reaction_series: pd.Series | None = None
        if "initial_reaction_return_pct" in events.columns:
            reaction_series = events["initial_reaction_return_pct"]
        elif self._primary_reaction:
            candidate_column = f"return_{self._primary_reaction}_pct"
            if candidate_column in events.columns:
                reaction_series = events[candidate_column]

        sample_surprise = _extract_sample(events.get("cpi_surprise"))
        sample_reaction = _extract_sample(reaction_series)
        sample_fake_any = _extract_sample(events.get("fake_any"))
        sample_duration = _extract_sample(events.get("fake_duration_minutes"))

        logger.debug(
            "Sample metrics for dashboard charts (surprise=%s, reaction=%s, fake_any=%s, fake_duration=%s)",
            sample_surprise,
            sample_reaction,
            sample_fake_any,
            sample_duration,
        )

        self._events = events

    def _build_summary_cards(self) -> None:
        events = self._events
        cards: list[dict[str, str]] = []
        cards.append({
            "label": "CPI Releases",
            "value": f"{len(events)}",
            "description": "Total releases analyzed",
        })

        if not events.empty and "fake_any" in events.columns:
            valid = events["fake_any"].dropna()
            if not valid.empty:
                rate = float(valid.mean())
                cards.append({
                    "label": "Overall Fake-out Rate",
                    "value": f"{rate * 100:.1f}%",
                    "description": "Share of releases reversing direction",
                })

        if self._primary_reaction:
            column = f"return_{self._primary_reaction}_pct"
            if column in events.columns and not events[column].dropna().empty:
                avg_reaction = float(events[column].dropna().mean())
                cards.append({
                    "label": f"Avg {self._primary_reaction} Return",
                    "value": f"{avg_reaction:.2f}%",
                    "description": "Mean immediate move after release",
                })

        if self._primary_evaluation:
            column = f"return_{self._primary_evaluation}_pct"
            if column in events.columns and not events[column].dropna().empty:
                avg_eval = float(events[column].dropna().mean())
                cards.append({
                    "label": f"Avg {self._primary_evaluation} Return",
                    "value": f"{avg_eval:.2f}%",
                    "description": "Mean subsequent move in evaluation window",
                })

        if "fake_duration_minutes" in events.columns:
            duration_series = events["fake_duration_minutes"].dropna()
            if not duration_series.empty:
                median_duration = float(duration_series.median())
                cards.append({
                    "label": "Median Reversal Time",
                    "value": f"{median_duration:.0f} min",
                    "description": "Typical time to fake-out completion",
                })

        mean_surprise = events["cpi_surprise"].dropna()
        if not mean_surprise.empty:
            avg_surprise = float(mean_surprise.mean())
            cards.append({
                "label": "Avg CPI Surprise",
                "value": f"{avg_surprise:+.2f}",
                "description": "Actual minus expected (%)",
            })

        self._summary_cards = cards

    def _build_figures(self) -> None:
        events = self._events
        figures: Dict[str, dict[str, Any]] = {}
        messages: Dict[str, str] = {}

        logger.debug("Building dashboard figures from %d events", len(events))

        if events.empty:
            logger.debug("No events available; skipping chart generation")
            self._figures = figures
            self._figure_messages = messages
            return

        def register(chart_id: str, builder: Callable[[], go.Figure | None], empty_message: str) -> None:
            try:
                figure = builder()
            except Exception:
                logger.exception("Failed to build chart '%s'", chart_id)
                messages[chart_id] = "Unable to render chart due to an internal error."
                return
            if figure is None:
                messages[chart_id] = empty_message
                logger.debug("Chart '%s' unavailable: %s", chart_id, empty_message)
                return
            trace_count = len(getattr(figure, "data", []) or [])
            if trace_count == 0:
                logger.warning("Chart '%s' produced a figure with zero traces", chart_id)
            layout_obj = getattr(figure, "layout", None)
            annotations = getattr(layout_obj, "annotations", None) if layout_obj is not None else None
            if isinstance(annotations, (list, tuple)):
                annotation_count = len(annotations)
            elif annotations:
                annotation_count = 1
            else:
                annotation_count = 0
            preview_payload: dict[str, Any] = {}
            try:
                preview_payload = figure.to_plotly_json()
            except Exception as payload_exc:  # pragma: no cover - defensive guard around Plotly internals
                logger.debug(
                    "Chart '%s' payload serialization skipped due to: %s",
                    chart_id,
                    payload_exc,
                )
            data_preview = preview_payload.get("data", []) if preview_payload else []
            if isinstance(data_preview, list) and data_preview:
                try:
                    preview_text = json.dumps(data_preview[:1], default=str)[:500]
                except TypeError:
                    preview_text = str(data_preview[:1])[:500]
            else:
                preview_text = "[]"
            logger.debug(
                "Chart '%s' generated with %d traces (annotations=%d) data_preview=%s",
                chart_id,
                trace_count,
                annotation_count,
                preview_text,
            )
            figures[chart_id] = self._serialize_figure(figure)

        def build_fake_by_window() -> go.Figure | None:
            records: list[dict[str, float]] = []
            for evaluation_label in self._evaluation_labels:
                column = f"fake_by_evaluation_{evaluation_label}"
                if column not in events.columns:
                    continue
                valid = events[column].dropna()
                if valid.empty:
                    continue
                records.append(
                    {
                        "evaluation_label": evaluation_label,
                        "fake_rate": float(valid.mean()),
                        "count": int(valid.size),
                    }
                )
            if not records:
                return None
            df = pd.DataFrame(records)
            logger.debug(
                "Fake-out by evaluation window chart using %d evaluation windows", len(df)
            )
            df["fake_rate_pct"] = df["fake_rate"] * 100.0
            df["label"] = df["fake_rate_pct"].apply(
                lambda value: "n/a" if pd.isna(value) else f"{value:.1f}%"
            )
            fig = px.bar(
                df,
                x="evaluation_label",
                y="fake_rate_pct",
                text="label",
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(title="Fake-out rate (%)")
            fig.update_xaxes(title="Evaluation window")
            fig.update_layout(title="Fake-out rates by evaluation window")
            return fig

        def build_fake_by_surprise_type() -> go.Figure | None:
            records: list[dict[str, float]] = []
            for evaluation_label in self._evaluation_labels:
                column = f"fake_by_evaluation_{evaluation_label}"
                if column not in events.columns:
                    continue
                for surprise_type, group in events.groupby("surprise_type", dropna=False):
                    valid = group[column].dropna()
                    if valid.empty:
                        continue
                    records.append(
                        {
                            "evaluation_label": evaluation_label,
                            "surprise_type": surprise_type,
                            "fake_rate": float(valid.mean()),
                            "count": int(valid.size),
                        }
                    )
            if not records:
                return None
            df = pd.DataFrame(records)
            logger.debug(
                "Fake-out by surprise type chart using %d grouped rows", len(df)
            )
            df["fake_rate_pct"] = df["fake_rate"] * 100.0
            fig = px.bar(
                df,
                x="surprise_type",
                y="fake_rate_pct",
                color="evaluation_label",
                barmode="group",
            )
            fig.update_yaxes(title="Fake-out rate (%)")
            fig.update_xaxes(title="Surprise type")
            fig.update_layout(title="Fake-out rates by surprise type")
            return fig

        def build_timeline() -> go.Figure | None:
            if not self._primary_evaluation:
                return None
            column = f"fake_by_evaluation_{self._primary_evaluation}"
            if column not in events.columns:
                return None
            timeline_df = events.copy()
            values = timeline_df.get(column, pd.Series(dtype="boolean"))
            outcome_labels = []
            for value in values:
                if pd.isna(value):
                    outcome_labels.append("Incomplete")
                elif bool(value):
                    outcome_labels.append("Fake-out")
                else:
                    outcome_labels.append("Sustained")
            timeline_df["outcome"] = outcome_labels
            return_column = f"return_{self._primary_evaluation}_pct"
            if return_column not in timeline_df.columns:
                return None
            logger.debug(
                "Timeline chart prepared with %d releases", len(timeline_df)
            )
            fig = px.scatter(
                timeline_df,
                x="release_datetime",
                y=return_column,
                color="outcome",
                hover_data={
                    "cpi_actual": True,
                    "cpi_expected": True,
                    "cpi_surprise": True,
                    return_column: True,
                },
            )
            fig.update_traces(marker=dict(size=11), selector=dict(mode="markers"))
            fig.update_layout(
                title=f"Timeline of CPI releases ({self._primary_evaluation} outcome)",
                yaxis_title=f"Return {self._primary_evaluation} (%)",
                xaxis_title="Release datetime",
            )
            return fig

        def build_surprise_distribution() -> go.Figure | None:
            if "cpi_surprise" not in events.columns:
                logger.debug("CPI surprise distribution chart aborted: missing 'cpi_surprise' column")
                return None
            surprises = pd.to_numeric(events["cpi_surprise"], errors="coerce").dropna()
            if surprises.empty:
                logger.debug("CPI surprise distribution chart aborted: no non-null surprises available")
                return None
            sample_count = int(surprises.size)
            min_surprise = float(surprises.min())
            max_surprise = float(surprises.max())
            range_min = min(-0.3, min_surprise)
            range_max = max(0.6, max_surprise)
            logger.debug(
                "CPI surprise distribution dataset: count=%d, min=%.3f, max=%.3f, range=[%.3f, %.3f]",
                sample_count,
                min_surprise,
                max_surprise,
                range_min,
                range_max,
            )
            nbins = min(30, max(10, int(math.sqrt(sample_count))))
            logger.debug(
                "CPI surprise distribution sample length=%d (nbins=%d)",
                sample_count,
                nbins,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=surprises.tolist(),
                    nbinsx=nbins,
                    name="CPI Surprises",
                    marker=dict(color="#636EFA"),
                    opacity=0.85,
                )
            )
            fig.update_traces(showlegend=False)
            fig.add_vline(x=0, line_dash="dash", line_color="#EF4444")
            fig.update_xaxes(range=[range_min, range_max])
            fig.update_layout(
                title="Distribution of CPI surprises",
                xaxis_title="Surprise (actual - expected, %)",
                yaxis_title="Count",
            )
            trace_points = len(getattr(fig.data[0], "x", []) or []) if fig.data else 0
            logger.debug(
                "CPI surprise distribution traces generated: %d (first trace points=%d)",
                len(fig.data),
                trace_points,
            )
            return fig

        def build_reaction_vs_surprise() -> go.Figure | None:
            required_columns = {"cpi_surprise", "initial_reaction_return_pct"}
            missing_columns = [column for column in required_columns if column not in events.columns]
            if missing_columns:
                logger.debug(
                    "Reaction vs surprise chart aborted: missing columns %s",
                    missing_columns,
                )
                return None
            working_columns = [
                "cpi_surprise",
                "initial_reaction_return_pct",
                "fake_any",
                "release_datetime",
            ]
            available_columns = [column for column in working_columns if column in events.columns]
            df = events[available_columns].copy()
            df["cpi_surprise"] = pd.to_numeric(df["cpi_surprise"], errors="coerce")
            df["initial_reaction_return_pct"] = pd.to_numeric(
                df["initial_reaction_return_pct"], errors="coerce"
            )
            df = df.dropna(subset=["cpi_surprise", "initial_reaction_return_pct"])
            if df.empty:
                logger.debug("Reaction vs surprise chart aborted: no complete surprise/reaction pairs")
                return None
            logger.debug("Reaction vs surprise sample length prior to plotting: %d", len(df))

            if "fake_any" in df.columns:
                def _normalize_flag(value: Any) -> object:
                    if pd.isna(value):
                        return pd.NA
                    if isinstance(value, str):
                        normalized = value.strip().lower()
                        if normalized in {"true", "t", "1", "yes"}:
                            return True
                        if normalized in {"false", "f", "0", "no"}:
                            return False
                        if normalized in {"", "nan", "n/a", "na", "none", "null", "--", "—"}:
                            return pd.NA
                    try:
                        return bool(value)
                    except Exception:  # pragma: no cover - defensive guard
                        return pd.NA

                df["fake_flag"] = df["fake_any"].apply(_normalize_flag)
                df["outcome"] = df["fake_flag"].map({True: "Fake move", False: "Sustained move"}).fillna(
                    "Incomplete"
                )
            else:
                df["outcome"] = "Unknown"
            outcome_counts = df["outcome"].value_counts(dropna=False).to_dict()
            surprise_min = float(df["cpi_surprise"].min())
            surprise_max = float(df["cpi_surprise"].max())
            reaction_min = float(df["initial_reaction_return_pct"].min())
            reaction_max = float(df["initial_reaction_return_pct"].max())
            logger.debug(
                "Reaction vs surprise dataset: rows=%d, outcome_counts=%s, surprise_range=(%.3f, %.3f), reaction_range=(%.3f, %.3f)",
                len(df),
                outcome_counts,
                surprise_min,
                surprise_max,
                reaction_min,
                reaction_max,
            )
            if "release_datetime" in df.columns:
                df["release_label"] = (
                    pd.to_datetime(df["release_datetime"], utc=True)
                    .dt.tz_convert("UTC")
                    .dt.strftime("%Y-%m-%d %H:%M UTC")
                )
            color_map = {
                "Fake move": "#EF553B",
                "Sustained move": "#636EFA",
                "Incomplete": "#94a3b8",
                "Unknown": "#a855f7",
            }
            outcome_order = ["Fake move", "Sustained move", "Incomplete", "Unknown"]
            fig = go.Figure()
            for outcome in outcome_order:
                subset = df[df["outcome"] == outcome]
                if subset.empty:
                    continue
                customdata = None
                hovertemplate = (
                    "Surprise %{x:.2f}<br>Reaction %{y:.2f}%<extra>"
                    f"{outcome}</extra>"
                )
                if "release_label" in subset.columns:
                    release_labels = subset["release_label"].astype(str).tolist()
                    if release_labels:
                        customdata = np.array(release_labels, dtype=object).reshape(-1, 1)
                        hovertemplate = (
                            "Release %{customdata[0]}<br>Surprise %{x:.2f}<br>Reaction %{y:.2f}%<extra>"
                            f"{outcome}</extra>"
                        )
                fig.add_trace(
                    go.Scatter(
                        x=subset["cpi_surprise"].astype(float),
                        y=subset["initial_reaction_return_pct"].astype(float),
                        mode="markers",
                        name=outcome,
                        marker=dict(
                            color=color_map.get(outcome, "#636EFA"),
                            size=10,
                            opacity=0.85,
                        ),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )
            if len(df) >= 2:
                x_vals = df["cpi_surprise"].astype(float).to_numpy()
                y_vals = df["initial_reaction_return_pct"].astype(float).to_numpy()
                try:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover - defensive guard
                    logger.debug("Reaction vs surprise trendline skipped: %s", exc)
                else:
                    x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=slope * x_range + intercept,
                            mode="lines",
                            name="Trendline",
                            line=dict(color="#EF553B", width=2),
                            hovertemplate="Surprise %{x:.2f}<br>Trend %{y:.2f}%<extra>Trendline</extra>",
                        )
                    )
            fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            fig.update_layout(
                title="Price reaction vs CPI surprise",
                xaxis_title="CPI surprise (%)",
                yaxis_title="Initial reaction return (%)",
            )
            point_counts = [len(getattr(trace, "x", []) or []) for trace in fig.data]
            logger.debug(
                "Reaction vs surprise traces generated: %d (point_counts=%s)",
                len(fig.data),
                point_counts,
            )
            return fig

        def build_fake_probability() -> go.Figure | None:
            required_columns = {"cpi_surprise", "fake_any"}
            missing_columns = [column for column in required_columns if column not in events.columns]
            if missing_columns:
                logger.debug(
                    "Fake probability chart aborted: missing columns %s",
                    missing_columns,
                )
                return None
            probability_records = _compute_fake_probability_by_surprise(events)
            if not probability_records:
                logger.debug("Fake probability chart aborted: insufficient data to build bins")
                return None
            df = pd.DataFrame(probability_records)
            if df.empty:
                logger.debug("Fake probability chart aborted: computed dataset is empty")
                return None
            df = df[df["sample_size"] > 0].sort_values("avg_surprise")
            if df.empty:
                logger.debug("Fake probability chart aborted: no bins with samples")
                return None
            total_samples = int(df["sample_size"].sum())
            logger.debug(
                "Fake probability vs surprise dataset: bins=%d, total_samples=%d, fake_rate_mean=%.3f",
                len(df),
                total_samples,
                float(df["fake_rate"].mean()),
            )
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df["label"],
                    y=df["fake_rate_pct"],
                    name="Fake rate (%)",
                    marker_color="#636EFA",
                    customdata=[(float(avg), int(size)) for avg, size in zip(df["avg_surprise"], df["sample_size"])],
                    hovertemplate=(
                        "Surprise bin %{x}<br>Avg surprise %{customdata[0]:.2f}"
                        "<br>Fake rate %{y:.1f}%<br>Samples %{customdata[1]}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["label"],
                    y=df["sample_size"],
                    name="Sample size",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(color="#EF553B", width=2),
                    marker=dict(size=8),
                    hovertemplate="Surprise bin %{x}<br>Sample size %{y}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Fake-out probability vs surprise",
                xaxis_title="CPI surprise bin",
                yaxis=dict(title="Fake-out rate (%)"),
                yaxis2=dict(
                    title="Sample size",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                bargap=0.15,
            )
            return fig

        def build_fake_duration_distribution() -> go.Figure | None:
            required_columns = {"fake_duration_minutes", "fake_any"}
            missing_columns = [column for column in required_columns if column not in events.columns]
            if missing_columns:
                logger.debug(
                    "Fake duration chart aborted: missing columns %s",
                    missing_columns,
                )
                return None
            fake_flags = _normalize_boolean_series(events["fake_any"])
            fake_mask = fake_flags.fillna(False).astype(bool)
            series = pd.to_numeric(
                events.loc[fake_mask, "fake_duration_minutes"],
                errors="coerce",
            ).dropna()
            if series.empty:
                logger.debug("Fake duration chart aborted: no durations for confirmed fake moves")
                return None
            sample_count = int(series.size)
            mean_value = float(series.mean())
            median_value = float(series.median())
            min_value = float(series.min())
            max_value = float(series.max())
            logger.debug(
                "Fake duration dataset: count=%d, mean=%.2f, median=%.2f, range=[%.2f, %.2f]",
                sample_count,
                mean_value,
                median_value,
                min_value,
                max_value,
            )
            nbins = min(30, max(6, int(sample_count ** 0.5)))
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=series,
                    nbinsx=nbins,
                    marker_color="#636EFA",
                    opacity=0.85,
                    name="Durations",
                )
            )
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color="#2563eb",
                annotation_text=f"Mean {mean_value:.0f}m",
                annotation_position="top right",
            )
            fig.add_vline(
                x=median_value,
                line_dash="dot",
                line_color="#ef4444",
                annotation_text=f"Median {median_value:.0f}m",
                annotation_position="top left",
            )
            fig.update_layout(
                title="Distribution of fake move durations",
                xaxis_title="Minutes",
                yaxis_title="Count",
                bargap=0.05,
                showlegend=False,
            )
            trace_points = len(getattr(fig.data[0], "x", []) or []) if fig.data else 0
            logger.debug(
                "Fake duration histogram traces generated: %d (first trace points=%d)",
                len(fig.data),
                trace_points,
            )
            return fig

        def build_duration_by_surprise() -> go.Figure | None:
            if "fake_duration_minutes" not in events.columns:
                return None
            df = events.dropna(subset=["fake_duration_minutes"])
            if df.empty:
                return None
            grouped = (
                df.groupby("surprise_type", dropna=False)["fake_duration_minutes"].mean().dropna()
            )
            if grouped.empty:
                return None
            records = grouped.reset_index(name="avg_minutes")
            records["avg_minutes"] = records["avg_minutes"].astype(float)
            logger.debug(
                "Duration by surprise chart using %d groups", len(records)
            )
            fig = px.bar(
                records,
                x="surprise_type",
                y="avg_minutes",
            )
            fig.update_layout(
                title="Average reversal time by surprise type",
                xaxis_title="Surprise type",
                yaxis_title="Average minutes",
            )
            return fig

        def build_price_trajectories() -> go.Figure | None:
            if self._price_series.empty:
                logger.debug("Price trajectory chart aborted: price series is empty")
                return None
            if "fake_any" not in events.columns:
                logger.debug("Price trajectory chart aborted: missing 'fake_any' column")
                return None
            categories = {"Fake move": 0, "Sustained move": 0}
            max_per_category = 5
            records: list[dict[str, float]] = []
            price_series = self._price_series.sort_index()
            price_tz = price_series.index.tz
            for idx, row in events.iterrows():
                fake_value = row.get("fake_any")
                if pd.isna(fake_value):
                    continue
                category = "Fake move" if bool(fake_value) else "Sustained move"
                if categories.get(category, 0) >= max_per_category:
                    continue
                release_value = row.get("release_datetime")
                base_price = row.get("base_price")
                if pd.isna(release_value):
                    continue
                release_ts = pd.Timestamp(release_value)
                if release_ts.tzinfo is None:
                    release_ts = release_ts.tz_localize(price_tz)
                else:
                    release_ts = release_ts.tz_convert(price_tz)
                start = release_ts - self._price_window_before
                end = release_ts + self._price_window_after
                window = price_series.loc[start:end].dropna().copy()
                if window.empty:
                    continue
                base_value: float | None
                if base_price is None or (isinstance(base_price, float) and math.isnan(base_price)):
                    base_value = _price_at(price_series, release_ts)
                else:
                    base_value = float(base_price)
                if base_value is None or base_value == 0.0:
                    continue
                if release_ts not in window.index:
                    window.loc[release_ts] = base_value
                    window.sort_index(inplace=True)
                normalized = (window / base_value) * 100.0
                for ts, value in normalized.items():
                    minutes = (ts - release_ts).total_seconds() / 60.0
                    records.append(
                        {
                            "event_id": idx,
                            "category": category,
                            "minutes": minutes,
                            "normalized_price": float(value),
                            "release": release_ts.isoformat(),
                        }
                    )
                categories[category] = categories.get(category, 0) + 1
            logger.debug("Price trajectory sampling counts by category: %s", categories)
            if not records:
                logger.debug("Price trajectory chart aborted: no qualifying price windows found")
                return None
            df = pd.DataFrame(records)
            if df.empty:
                logger.debug("Price trajectory chart aborted: normalized dataset is empty")
                return None
            df.sort_values(["category", "event_id", "minutes"], inplace=True)
            unique_events = df["event_id"].nunique()
            logger.debug(
                "Price trajectory chart using %d points from %d releases (categories=%s)",
                len(df),
                unique_events,
                categories,
            )
            fig = go.Figure()
            color_map = {"Fake move": "#EF553B", "Sustained move": "#636EFA"}
            legend_tracker: set[str] = set()
            for (event_id, category), group in df.groupby(["event_id", "category"]):
                color = color_map.get(category, "#636EFA")
                legendgroup = category
                showlegend = category not in legend_tracker
                release_labels = group["release"].astype(str).tolist() if "release" in group.columns else []
                customdata = (
                    np.array(release_labels, dtype=object).reshape(-1, 1)
                    if release_labels
                    else None
                )
                if customdata is not None:
                    hovertemplate = (
                        "Release %{customdata[0]}<br>Minutes %{x:.0f}<br>Normalized price %{y:.2f}<extra>"
                        f"{category}</extra>"
                    )
                else:
                    hovertemplate = (
                        "Minutes %{x:.0f}<br>Normalized price %{y:.2f}<extra>"
                        f"{category}</extra>"
                    )
                fig.add_trace(
                    go.Scatter(
                        x=group["minutes"].astype(float),
                        y=group["normalized_price"].astype(float),
                        mode="lines",
                        name=category,
                        legendgroup=legendgroup,
                        showlegend=showlegend,
                        line=dict(color=color, width=1.5),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )
                legend_tracker.add(category)
            fig.add_vline(x=0, line_dash="dash", line_color="#EF4444")
            fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8")
            fig.update_layout(
                title="Normalized BTC price trajectories",
                xaxis_title="Minutes from release",
                yaxis_title="Normalized price (release = 100)",
            )
            point_counts = [len(getattr(trace, "x", []) or []) for trace in fig.data]
            logger.debug(
                "Price trajectory traces generated: %d (point_counts=%s)",
                len(fig.data),
                point_counts,
            )
            return fig

        register(
            "fake_by_window",
            build_fake_by_window,
            "No evaluation windows contain fake-out data.",
        )
        register(
            "fake_by_surprise",
            build_fake_by_surprise_type,
            "No fake-out data available by surprise category.",
        )
        register(
            "timeline",
            build_timeline,
            "Evaluation window returns are required to plot the timeline.",
        )
        register(
            "surprise_distribution",
            build_surprise_distribution,
            "CPI surprise data is unavailable.",
        )
        register(
            "reaction_vs_surprise",
            build_reaction_vs_surprise,
            "Need CPI surprise values and reaction returns to build this chart.",
        )
        register(
            "fake_probability",
            build_fake_probability,
            "Need fake-out classifications and CPI surprises to estimate probabilities.",
        )
        register(
            "fake_durations",
            build_fake_duration_distribution,
            "Fake-out durations are unavailable for this dataset.",
        )
        register(
            "duration_by_surprise",
            build_duration_by_surprise,
            "Fake-out durations by surprise type are unavailable.",
        )
        register(
            "price_trajectories",
            build_price_trajectories,
            "Price history around CPI releases is required to show trajectories.",
        )

        self._figures = figures
        self._figure_messages = messages
        logger.debug("Generated %d Plotly figures", len(figures))

    def _prepare_table_and_price_windows(self) -> None:
        events = self._events
        table_rows: list[dict[str, Any]] = []
        price_windows: dict[int, dict[str, Any]] = {}

        if events.empty:
            self._table_rows = table_rows
            self._price_windows = price_windows
            return

        reaction_columns = [f"return_{label}_pct" for label in self._reaction_labels if f"return_{label}_pct" in events.columns]
        evaluation_columns = [f"return_{label}_pct" for label in self._evaluation_labels if f"return_{label}_pct" in events.columns]

        fake_status_column = None
        if self._primary_evaluation:
            candidate = f"fake_by_evaluation_{self._primary_evaluation}"
            if candidate in events.columns:
                fake_status_column = candidate

        for idx, (release, row) in enumerate(zip(self._releases, events.itertuples(index=False))):
            row_dict = row._asdict()
            release_dt = pd.Timestamp(row_dict.get("release_datetime"))
            if release_dt.tzinfo is None:
                release_dt = release_dt.tz_localize("UTC")
            release_label = release_dt.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
            surprise_value = row_dict.get("cpi_surprise")
            fake_any_value = row_dict.get("fake_any")
            outcome = "Incomplete"
            if fake_any_value is True:
                outcome = "Fake"
            elif fake_any_value is False:
                outcome = "Sustained"

            if fake_status_column is not None:
                status_value = row_dict.get(fake_status_column)
                if status_value is True:
                    outcome = "Fake"
                elif status_value is False:
                    outcome = "Sustained"
                elif status_value is None or (isinstance(status_value, float) and math.isnan(status_value)):
                    outcome = "Incomplete"

            duration_value = row_dict.get("fake_duration_minutes")
            if duration_value is None or pd.isna(duration_value):
                cleaned_duration: int | None = None
            else:
                cleaned_duration = int(round(float(duration_value)))

            table_row = {
                "id": idx,
                "releaseLabel": release_label,
                "releaseIso": release_dt.isoformat(),
                "actual": _clean_numeric(row_dict.get("cpi_actual")),
                "expected": _clean_numeric(row_dict.get("cpi_expected")),
                "surprise": _clean_numeric(surprise_value),
                "surpriseType": row_dict.get("surprise_type", "Unknown"),
                "basePrice": _clean_numeric(row_dict.get("base_price")),
                "outcome": outcome,
                "fakeDurationMinutes": cleaned_duration,
            }

            reactions_payload: dict[str, float | None] = {}
            for column in reaction_columns:
                label = column.replace("return_", "").replace("_pct", "")
                value = row_dict.get(column)
                reactions_payload[label] = None if value is None or pd.isna(value) else float(value)
            table_row["reactionReturns"] = reactions_payload

            evaluation_payload: dict[str, float | None] = {}
            for column in evaluation_columns:
                label = column.replace("return_", "").replace("_pct", "")
                value = row_dict.get(column)
                evaluation_payload[label] = None if value is None or pd.isna(value) else float(value)
            table_row["evaluationReturns"] = evaluation_payload

            fake_map: dict[str, Any] = {}
            for evaluation_label in self._evaluation_labels:
                column = f"fake_by_evaluation_{evaluation_label}"
                if column in events.columns:
                    value = row_dict.get(column)
                    fake_map[evaluation_label] = None if value is None or pd.isna(value) else bool(value)
            table_row["fakeByEvaluation"] = fake_map

            table_rows.append(table_row)

            # prepare price window data
            release_time = release.release_datetime
            start_time = release_time - self._price_window_before
            end_time = release_time + self._price_window_after
            window_series = self._price_series.loc[start_time:end_time].dropna()
            if window_series.empty:
                price_windows[idx] = {
                    "timestamps": [],
                    "prices": [],
                    "release": release_time.isoformat(),
                }
            else:
                timestamps = [timestamp.isoformat() for timestamp in window_series.index]
                price_windows[idx] = {
                    "timestamps": timestamps,
                    "prices": [float(value) for value in window_series.tolist()],
                    "release": release_time.isoformat(),
                }

        self._table_rows = table_rows
        self._price_windows = price_windows

    def _plotly_loader_script(self) -> str:
        if self._plotly_js_mode == "cdn":
            url = self._plotly_cdn_url or "https://cdn.plot.ly/plotly-latest.min.js"
            return f'<script src="{url}" crossorigin="anonymous"></script>'
        try:
            from plotly import offline as plotly_offline  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Plotly offline support is unavailable. Install 'plotly' or switch to the CDN backend."
            ) from exc

        try:
            plotly_lib_js_content = plotly_offline.get_plotlyjs()
        except Exception:
            try:
                dummy_div = plotly_offline.plot(  # type: ignore[arg-type]
                    {"data": [], "layout": {}},
                    include_plotlyjs=True,
                    output_type="div",
                    auto_open=False,
                    show_link=False,
                )
            except Exception as plot_exc:  # pragma: no cover - defensive guard
                raise RuntimeError(
                    "Unable to retrieve the Plotly.js offline bundle. Ensure the 'plotly' package is fully installed or select the CDN backend."
                ) from plot_exc
            script_match = re.search(r"<script[^>]*>(.*?)</script>", dummy_div, flags=re.DOTALL)
            if not script_match:
                raise RuntimeError(
                    "Failed to extract the Plotly.js script from the offline HTML output."
                )
            plotly_lib_js_content = script_match.group(1)
        return f"<script>{plotly_lib_js_content}</script>"

    def _render_template(self) -> str:
        config = {
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }
        html_figures = json.dumps(self._figures)
        html_table = json.dumps(self._table_rows, allow_nan=False)
        html_prices = json.dumps(self._price_windows, allow_nan=False)
        html_cards = json.dumps(self._summary_cards, allow_nan=False)
        html_messages = json.dumps(self._figure_messages, ensure_ascii=False)
        plotly_lib_js = self._plotly_loader_script()

        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BTC CPI Fake-out Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: 'Inter', 'Segoe UI', sans-serif;
      --background: #f5f7fb;
      --card-bg: #ffffff;
      --card-border: #e1e5ee;
      --text-primary: #111827;
      --text-secondary: #4b5563;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      background-color: var(--background);
      color: var(--text-primary);
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 600;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 4rem;
    }}
    header {{
      margin-bottom: 2rem;
    }}
    header p {{
      color: var(--text-secondary);
      margin-top: 0.5rem;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .summary-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }}
    .summary-card .label {{
      color: var(--text-secondary);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-card .value {{
      font-size: 1.75rem;
      font-weight: 600;
    }}
    .summary-card .description {{
      color: var(--text-secondary);
      font-size: 0.9rem;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
    }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.5rem;
      box-shadow: 0 18px 30px rgba(15, 23, 42, 0.12);
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .chart-card header {{
      margin: 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .chart-card h2 {{
      font-size: 1.1rem;
      font-weight: 600;
    }}
    .chart-card .actions {{
      display: flex;
      gap: 0.5rem;
    }}
    .chart-card button {{
      border: 1px solid var(--card-border);
      background: #fff;
      color: var(--text-secondary);
      border-radius: 999px;
      padding: 0.35rem 0.85rem;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .chart-card button:hover {{
      color: var(--accent);
      border-color: var(--accent);
    }}
    .detail-grid {{
      margin-top: 2.5rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 1.75rem;
    }}
    .table-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.5rem;
      box-shadow: 0 18px 30px rgba(15, 23, 42, 0.12);
      overflow: hidden;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    .filters label {{
      font-size: 0.85rem;
      color: var(--text-secondary);
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}
    .filters input,
    .filters select {{
      border-radius: 10px;
      border: 1px solid var(--card-border);
      padding: 0.45rem 0.7rem;
      font-size: 0.95rem;
      background: #fff;
      min-width: 160px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border-spacing: 0;
    }}
    thead th {{
      text-align: left;
      font-size: 0.75rem;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 0.65rem 0.75rem;
      border-bottom: 1px solid var(--card-border);
      background: rgba(37, 99, 235, 0.05);
    }}
    tbody td {{
      padding: 0.65rem 0.75rem;
      border-bottom: 1px solid var(--card-border);
      font-size: 0.9rem;
    }}
    tbody tr {{
      cursor: pointer;
      transition: background 0.2s ease;
    }}
    tbody tr:hover {{
      background: rgba(37, 99, 235, 0.08);
    }}
    tbody tr.selected {{
      background: rgba(37, 99, 235, 0.15);
    }}
    .event-detail {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.5rem;
      box-shadow: 0 18px 30px rgba(15, 23, 42, 0.12);
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}
    .event-meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
    }}
    .event-meta div {{
      padding: 0.75rem;
      border-radius: 12px;
      background: rgba(37, 99, 235, 0.08);
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }}
    .event-meta span.label {{
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--text-secondary);
    }}
    .event-meta span.value {{
      font-size: 1.05rem;
      font-weight: 600;
    }}
    .event-meta span.value.negative {{
      color: #dc2626;
    }}
    .event-meta span.value.positive {{
      color: #16a34a;
    }}
    footer {{
      margin-top: 3rem;
      color: var(--text-secondary);
      font-size: 0.8rem;
      text-align: center;
    }}
    @media (max-width: 768px) {{
      .chart-card {{
        padding: 1.1rem;
      }}
      .chart-card h2 {{
        font-size: 1rem;
      }}
      thead {{
        display: none;
      }}
      table, tbody, tr, td {{
        display: block;
        width: 100%;
      }}
      tbody tr {{
        margin-bottom: 1rem;
        border-radius: 12px;
        border: 1px solid var(--card-border);
        background: #fff;
      }}
      tbody td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--card-border);
      }}
      tbody td::before {{
        content: attr(data-label);
        font-weight: 600;
        display: block;
        margin-bottom: 0.35rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-size: 0.75rem;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1>BTC vs CPI Fake-out Dashboard</h1>
      <p>Interactive exploration of CPI releases and Bitcoin price reactions. Click rows for event-level insights and hover charts for details.</p>
    </header>
    <section class=\"summary-grid\" id=\"summary-cards\"></section>
    <section class=\"chart-grid\" id=\"overview-section\">
      <div class=\"chart-card\">
        <header>
          <h2>Fake-out rates by evaluation window</h2>
          <div class=\"actions\">
            <button data-chart-id=\"fake_by_window\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"fake_by_window\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"fake_by_window\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Fake-out rates by surprise type</h2>
          <div class=\"actions\">
            <button data-chart-id=\"fake_by_surprise\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"fake_by_surprise\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"fake_by_surprise\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>CPI release timeline</h2>
          <div class=\"actions\">
            <button data-chart-id=\"timeline\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"timeline\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"timeline\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>CPI surprise distribution</h2>
          <div class=\"actions\">
            <button data-chart-id=\"surprise_distribution\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"surprise_distribution\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"surprise_distribution\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Reaction vs surprise</h2>
          <div class=\"actions\">
            <button data-chart-id=\"reaction_vs_surprise\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"reaction_vs_surprise\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"reaction_vs_surprise\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Fake probability vs surprise</h2>
          <div class=\"actions\">
            <button data-chart-id=\"fake_probability\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"fake_probability\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"fake_probability\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Fake move duration</h2>
          <div class=\"actions\">
            <button data-chart-id=\"fake_durations\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"fake_durations\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"fake_durations\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Reversal by surprise type</h2>
          <div class=\"actions\">
            <button data-chart-id=\"duration_by_surprise\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"duration_by_surprise\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"duration_by_surprise\"></div>
      </div>
      <div class=\"chart-card\">
        <header>
          <h2>Price trajectories</h2>
          <div class=\"actions\">
            <button data-chart-id=\"price_trajectories\" data-download-format=\"png\">Export PNG</button>
            <button data-chart-id=\"price_trajectories\" data-download-format=\"svg\">Export SVG</button>
          </div>
        </header>
        <div id=\"price_trajectories\"></div>
      </div>
    </section>
    <section class=\"detail-grid\">
      <div class=\"table-card\">
        <h2 style=\"margin-bottom: 1rem;\">CPI release details</h2>
        <div class=\"filters\">
          <label>Search
            <input type=\"search\" id=\"filter-search\" placeholder=\"Search releases, values\" />
          </label>
          <label>Surprise type
            <select id=\"filter-surprise\">
              <option value=\"all\">All</option>
              <option value=\"Positive\">Positive</option>
              <option value=\"Negative\">Negative</option>
              <option value=\"Inline\">Inline</option>
              <option value=\"Unknown\">Unknown</option>
            </select>
          </label>
          <label>Outcome
            <select id=\"filter-outcome\">
              <option value=\"all\">All</option>
              <option value=\"Fake\">Fake</option>
              <option value=\"Sustained\">Sustained</option>
              <option value=\"Incomplete\">Incomplete</option>
            </select>
          </label>
        </div>
        <div class=\"table-wrapper\" style=\"overflow-x: auto;\">
          <table>
            <thead>
              <tr>
                <th>Release</th>
                <th>Actual</th>
                <th>Expected</th>
                <th>Surprise</th>
                <th>Type</th>
                <th>Reaction</th>
                <th>Evaluation</th>
                <th>Outcome</th>
                <th>Reversal</th>
              </tr>
            </thead>
            <tbody id=\"event-table-body\"></tbody>
          </table>
        </div>
      </div>
      <div class=\"event-detail\">
        <div>
          <h2 id=\"event-detail-title\">Select a CPI release</h2>
          <p style=\"color: var(--text-secondary); margin-top: 0.5rem;\">Detailed BTC price action and CPI stats for the selected release.</p>
        </div>
        <div class=\"event-meta\">
          <div>
            <span class=\"label\">Actual</span>
            <span class=\"value\" id=\"detail-actual\">—</span>
          </div>
          <div>
            <span class=\"label\">Expected</span>
            <span class=\"value\" id=\"detail-expected\">—</span>
          </div>
          <div>
            <span class=\"label\">Surprise</span>
            <span class=\"value\" id=\"detail-surprise\">—</span>
          </div>
          <div>
            <span class=\"label\">Outcome</span>
            <span class=\"value\" id=\"detail-outcome\">—</span>
          </div>
          <div>
            <span class=\"label\">Reversal Time</span>
            <span class=\"value\" id=\"detail-reversal\">—</span>
          </div>
        </div>
        <div>
          <h3 style=\"margin-bottom: 0.5rem;\">BTC price around release</h3>
          <div id=\"event-price-chart\"></div>
        </div>
      </div>
    </section>
    <footer>
      Generated with Plotly · Works fully offline · Use filters and downloads to explore the dataset.
    </footer>
  </div>
  {plotly_lib_js}
  <script>
    const figures = {html_figures};
    const figureMessages = {html_messages};
    const tableRows = {html_table};
    const priceWindows = {html_prices};
    const summaryCards = {html_cards};
    const plotConfig = {json.dumps(config)};
    let activeRowId = null;

    function formatNumber(value, digits=2) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return '—';
      }}
      return Number(value).toFixed(digits);
    }}

    function formatPercentage(value, digits=2) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return '—';
      }}
      return `${{Number(value).toFixed(digits)}}%`;
    }}

    function populateSummaryCards() {{
      const node = document.getElementById('summary-cards');
      node.innerHTML = '';
      summaryCards.forEach(card => {{
        const element = document.createElement('article');
        element.className = 'summary-card';
        element.innerHTML = `
          <span class="label">${{card.label}}</span>
          <span class="value">${{card.value}}</span>
          <span class="description">${{card.description}}</span>
        `;
        node.appendChild(element);
      }});
    }}

    function renderCharts() {{
      const rendered = new Set();
      for (const [chartId, payload] of Object.entries(figures)) {{
        const node = document.getElementById(chartId);
        if (!node) continue;
        Plotly.newPlot(node, payload.data || [], payload.layout || {{}}, plotConfig);
        rendered.add(chartId);
      }}
      document.querySelectorAll('.chart-card').forEach(card => {{
        const chartNode = card.querySelector('div[id]');
        if (!chartNode) return;
        const chartId = chartNode.id;
        if (rendered.has(chartId)) return;
        const message = figureMessages[chartId] || 'No data available for this chart.';
        chartNode.innerHTML = `<p style="color: var(--text-secondary);">${{message}}</p>`;
        card.querySelectorAll(`button[data-chart-id="${{chartId}}"]`).forEach(button => {{
          button.disabled = true;
          button.style.opacity = '0.55';
          button.style.cursor = 'not-allowed';
        }});
      }});
    }}

    function attachDownloadButtons() {{
      document.querySelectorAll('button[data-chart-id]').forEach(button => {{
        button.addEventListener('click', () => {{
          if (button.disabled) return;
          const chartId = button.getAttribute('data-chart-id');
          const format = button.getAttribute('data-download-format') || 'png';
          const node = document.getElementById(chartId);
          if (!node) return;
          Plotly.downloadImage(node, {{
            format,
            filename: `btc-cpi-dashboard_${{chartId}}`,
            height: 600,
            width: 900,
            scale: 2,
          }});
        }});
      }});
    }}

    function createTableRow(row) {{
      const element = document.createElement('tr');
      element.dataset.rowId = row.id;
      const reactionLabels = Object.entries(row.reactionReturns)
        .filter(([, value]) => value !== null && !Number.isNaN(value))
        .map(([label, value]) => `${{label}}: ${{formatPercentage(value)}}`)
        .join('<br />');
      const evaluationLabels = Object.entries(row.evaluationReturns)
        .filter(([, value]) => value !== null && !Number.isNaN(value))
        .map(([label, value]) => `${{label}}: ${{formatPercentage(value)}}`)
        .join('<br />');
      const fakeDuration = row.fakeDurationMinutes ? `${{row.fakeDurationMinutes}} min` : '—';
      const cells = [
        ['Release', row.releaseLabel],
        ['Actual', formatNumber(row.actual, 2)],
        ['Expected', formatNumber(row.expected, 2)],
        ['Surprise', formatNumber(row.surprise, 2)],
        ['Type', row.surpriseType],
        ['Reaction', reactionLabels || '—'],
        ['Evaluation', evaluationLabels || '—'],
        ['Outcome', row.outcome],
        ['Reversal', fakeDuration],
      ];
      cells.forEach(([label, value]) => {{
        const cell = document.createElement('td');
        cell.dataset.label = label;
        cell.innerHTML = value;
        element.appendChild(cell);
      }});
      return element;
    }}

    function populateTable(rows) {{
      const body = document.getElementById('event-table-body');
      body.innerHTML = '';
      rows.forEach(row => {{
        const element = createTableRow(row);
        body.appendChild(element);
      }});
    }}

    function applyFilters() {{
      const searchValue = document.getElementById('filter-search').value.trim().toLowerCase();
      const surpriseFilter = document.getElementById('filter-surprise').value;
      const outcomeFilter = document.getElementById('filter-outcome').value;

      const filtered = tableRows.filter(row => {{
        const matchesSearch = searchValue === '' || Object.values(row).some(value => {{
          if (value === null || value === undefined) return false;
          if (typeof value === 'object') return false;
          return String(value).toLowerCase().includes(searchValue);
        }});
        const matchesSurprise = surpriseFilter === 'all' || row.surpriseType === surpriseFilter;
        const matchesOutcome = outcomeFilter === 'all' || row.outcome === outcomeFilter;
        return matchesSearch && matchesSurprise && matchesOutcome;
      }});

      populateTable(filtered);

      const tableBody = document.getElementById('event-table-body');
      const rowElements = Array.from(tableBody.querySelectorAll('tr'));
      let selectedRow = null;

      if (filtered.length > 0) {{
        if (activeRowId !== null) {{
          selectedRow = filtered.find(row => row.id === activeRowId) || null;
        }}
        if (!selectedRow) {{
          selectedRow = filtered[0];
          activeRowId = selectedRow.id;
        }}
        rowElements.forEach(node => {{
          const rowId = Number(node.dataset.rowId);
          node.classList.toggle('selected', rowId === activeRowId);
        }});
        updateDetailPanel(selectedRow);
        updatePriceChart(selectedRow);
      }} else {{
        activeRowId = null;
        document.getElementById('event-detail-title').textContent = 'No releases match your filters';
        document.getElementById('detail-actual').textContent = '—';
        document.getElementById('detail-expected').textContent = '—';
        const surpriseElement = document.getElementById('detail-surprise');
        surpriseElement.textContent = '—';
        surpriseElement.classList.remove('positive', 'negative');
        document.getElementById('detail-outcome').textContent = '—';
        document.getElementById('detail-reversal').textContent = '—';
        const priceNode = document.getElementById('event-price-chart');
        Plotly.react(priceNode, [], {{ title: 'No data' }}, plotConfig);
      }}

      return filtered;
    }}

    function formatDetailValue(value, suffix='') {{
      if (value === null || value === undefined || Number.isNaN(value)) return '—';
      return `${{Number(value).toFixed(2)}}${{suffix}}`;
    }}

    function updateDetailPanel(row) {{
      document.getElementById('event-detail-title').textContent = row.releaseLabel;
      document.getElementById('detail-actual').textContent = formatDetailValue(row.actual);
      document.getElementById('detail-expected').textContent = formatDetailValue(row.expected);
      document.getElementById('detail-surprise').textContent = formatDetailValue(row.surprise);
      document.getElementById('detail-outcome').textContent = row.outcome;
      document.getElementById('detail-reversal').textContent = row.fakeDurationMinutes ? `${{row.fakeDurationMinutes}} min` : '—';

      const surpriseValue = Number(row.surprise);
      const surpriseElement = document.getElementById('detail-surprise');
      surpriseElement.classList.remove('positive', 'negative');
      if (!Number.isNaN(surpriseValue)) {{
        if (surpriseValue > 0) surpriseElement.classList.add('positive');
        if (surpriseValue < 0) surpriseElement.classList.add('negative');
      }}
    }}

    function updatePriceChart(row) {{
      const payload = priceWindows[row.id];
      const node = document.getElementById('event-price-chart');
      if (!payload || !payload.timestamps || payload.timestamps.length === 0) {{
        Plotly.react(node, [], {{
          title: 'Price data unavailable',
          annotations: [{{
            text: 'No price data available for the selected window.',
            showarrow: false,
          }}]
        }}, plotConfig);
        return;
      }}
      const releaseTime = payload.release;
      const trace = {{
        x: payload.timestamps,
        y: payload.prices,
        mode: 'lines',
        name: 'BTC close',
        line: {{ color: '#2563eb', width: 2 }},
      }};
      const layout = {{
        title: 'BTC price trajectory',
        xaxis: {{ title: 'Timestamp', showgrid: false }},
        yaxis: {{ title: 'Price', zeroline: false }},
        shapes: [{{
          type: 'line',
          x0: releaseTime,
          x1: releaseTime,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: {{ color: '#ef4444', width: 2, dash: 'dot' }},
        }}],
        hovermode: 'x unified',
      }};
      Plotly.react(node, [trace], layout, plotConfig);
    }}

    function attachRowListeners() {{
      const body = document.getElementById('event-table-body');
      body.addEventListener('click', event => {{
        const rowElement = event.target.closest('tr');
        if (!rowElement) return;
        const identifier = Number(rowElement.dataset.rowId);
        activeRowId = identifier;
        document.querySelectorAll('#event-table-body tr').forEach(node => node.classList.remove('selected'));
        rowElement.classList.add('selected');
        const row = tableRows.find(item => item.id === identifier);
        if (!row) return;
        updateDetailPanel(row);
        updatePriceChart(row);
      }});
    }}

    function initialize() {{
      populateSummaryCards();
      renderCharts();
      attachDownloadButtons();
      populateTable(tableRows);
      attachRowListeners();

      const searchInput = document.getElementById('filter-search');
      const surpriseSelect = document.getElementById('filter-surprise');
      const outcomeSelect = document.getElementById('filter-outcome');
      const triggerFilters = () => applyFilters();

      searchInput.addEventListener('input', triggerFilters);
      surpriseSelect.addEventListener('change', triggerFilters);
      outcomeSelect.addEventListener('change', triggerFilters);

      const initialRow = tableRows[0];
      if (initialRow) {{
        activeRowId = initialRow.id;
        updateDetailPanel(initialRow);
        updatePriceChart(initialRow);
        const firstRowNode = document.querySelector('#event-table-body tr');
        if (firstRowNode) firstRowNode.classList.add('selected');
      }} else {{
        document.getElementById('event-price-chart').innerHTML = '<p style="color: var(--text-secondary);">No events available.</p>';
      }}
    }}

    document.addEventListener('DOMContentLoaded', initialize);
  </script>
</body>
</html>
"""
        return html


def _render_mpld3_dashboard(
    *,
    result: FakeoutAnalysisResult,
    config: FakeoutConfig,
    error_message: str | None = None,
) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import mpld3  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "mpld3 backend is not available. Install it with 'pip install mpld3' or select a Plotly backend."
        ) from exc

    events = result.events.copy()
    primary_evaluation = config.evaluation_windows[0][0] if config.evaluation_windows else None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("BTC CPI Fake-out Overview")
    ax.set_xlabel("Release datetime")
    ax.set_ylabel("Return (%)")

    plotted = False
    if not events.empty and primary_evaluation:
        column = f"return_{primary_evaluation}_pct"
        if column in events.columns:
            timeline = events.dropna(subset=["release_datetime", column]).copy()
            if not timeline.empty:
                timeline["release_datetime"] = pd.to_datetime(timeline["release_datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
                timeline.sort_values("release_datetime", inplace=True)
                ax.plot(
                    timeline["release_datetime"],
                    timeline[column],
                    marker="o",
                    linestyle="-",
                    color="#2563eb",
                    label=f"{primary_evaluation} return",
                )
                ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1)
                ax.legend(loc="best")
                plotted = True

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No evaluation window return data available for mpld3 fallback.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6b7280",
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    figure_html = mpld3.fig_to_html(fig)
    plt.close(fig)

    available_columns = [
        "release_datetime",
        "cpi_actual",
        "cpi_expected",
        "cpi_surprise",
    ]
    if primary_evaluation:
        return_column = f"return_{primary_evaluation}_pct"
        available_columns.append(return_column)
    available_columns = [column for column in available_columns if column in events.columns]

    if available_columns:
        preview = events[available_columns].copy()
        if "release_datetime" in preview.columns:
            preview["release_datetime"] = pd.to_datetime(preview["release_datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        numeric_columns = preview.select_dtypes(include=["number"]).columns
        preview[numeric_columns] = preview[numeric_columns].round(3)
        preview = preview.head(25)
        table_html = preview.to_html(index=False, classes="fallback-table")
    else:
        table_html = "<p>No CPI release data is available for summary display.</p>"

    alert_block = ""
    if error_message:
        alert_block = (
            "<div class=\"alert\">"
            f"Plotly dashboard generation failed: {html.escape(error_message)}"
            "</div>"
        )

    fallback_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BTC CPI Fake-out Dashboard (mpld3 fallback)</title>
  <style>
    body {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      margin: 0;
      padding: 2rem 1.25rem 4rem;
      background: #f9fafb;
      color: #111827;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
    }}
    header h1 {{
      margin-bottom: 0.35rem;
      font-size: 1.9rem;
    }}
    header p {{
      color: #4b5563;
      margin: 0;
    }}
    .alert {{
      background: #fee2e2;
      border: 1px solid #fecaca;
      color: #b91c1c;
      padding: 1rem 1.25rem;
      border-radius: 0.75rem;
      box-shadow: 0 12px 24px rgba(239, 68, 68, 0.18);
    }}
    .card {{
      background: #ffffff;
      border-radius: 1rem;
      padding: 1.5rem;
      box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
      border: 1px solid #e5e7eb;
    }}
    table.fallback-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    table.fallback-table th,
    table.fallback-table td {{
      border: 1px solid #e5e7eb;
      padding: 0.6rem 0.75rem;
      text-align: left;
    }}
    table.fallback-table th {{
      background: #f3f4f6;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 0.75rem;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1>BTC CPI Fake-out Dashboard</h1>
      <p>This simplified dashboard uses matplotlib + mpld3 as a fallback rendering backend.</p>
    </header>
    {alert_block}
    <section class=\"card\">
      <h2 style=\"margin-top: 0;\">Evaluation window returns</h2>
      {figure_html}
      <p style=\"color: #4b5563; font-size: 0.85rem;\">Tooltips, zooming, and panning are provided by mpld3.</p>
    </section>
    <section class=\"card\">
      <h2 style=\"margin-top: 0;\">Recent CPI releases</h2>
      {table_html}
    </section>
  </div>
</body>
</html>
"""
    return fallback_html


def render_dashboard_html(
    *,
    result: FakeoutAnalysisResult,
    releases: Sequence[CPIRelease],
    price_series: pd.Series,
    config: FakeoutConfig,
    output_path: Path,
    open_browser: bool = False,
    neutral_surprise_band: float = 0.05,
    price_window_before: timedelta = timedelta(hours=6),
    price_window_after: timedelta = timedelta(hours=12),
    dashboard_backend: Literal["plotly-cdn", "plotly-inline", "mpld3"] = "plotly-cdn",
) -> Path:
    """Render an interactive dashboard describing fake-out analysis results.

    Args:
        result: Fakeout analysis outcome describing returns and fake-out labels.
        releases: Ordered CPI release metadata.
        price_series: BTC price series used for price window charts.
        config: Configuration describing reaction/evaluation windows.
        output_path: Where to write the generated HTML.
        open_browser: Whether to open the generated dashboard in a browser tab.
        neutral_surprise_band: Surprise threshold for categorisation.
        price_window_before: Duration before a release to include in price window charts.
        price_window_after: Duration after a release to include in price window charts.
        dashboard_backend: Rendering backend to use. "plotly-cdn" loads Plotly from the
            public CDN, "plotly-inline" embeds the Plotly bundle directly, and "mpld3"
            generates a simplified matplotlib + mpld3 dashboard.
    """

    backend_choice = dashboard_backend.lower()
    if backend_choice not in {"plotly-cdn", "plotly-inline", "mpld3"}:
        raise ValueError(
            "dashboard_backend must be one of 'plotly-cdn', 'plotly-inline', or 'mpld3'."
        )

    html: str
    if backend_choice == "mpld3":
        html = _render_mpld3_dashboard(result=result, config=config)
    else:
        plotly_js_mode = "cdn" if backend_choice == "plotly-cdn" else "inline"
        builder = DashboardBuilder(
            result=result,
            releases=releases,
            price_series=price_series,
            config=config,
            neutral_surprise_band=neutral_surprise_band,
            price_window_before=price_window_before,
            price_window_after=price_window_after,
            plotly_js_mode=plotly_js_mode,
        )
        try:
            html = builder.build_html()
        except RuntimeError as exc:
            if backend_choice == "plotly-inline":
                raise RuntimeError(
                    "Plotly inline mode failed; ensure the offline Plotly bundle is available "
                    "or switch to the 'plotly-cdn' backend."
                ) from exc
            html = _render_mpld3_dashboard(
                result=result,
                config=config,
                error_message=str(exc),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if open_browser:
        try:
            import webbrowser

            webbrowser.open_new_tab(output_path.resolve().as_uri())
        except Exception:  # pragma: no cover - best effort
            pass
    return output_path
