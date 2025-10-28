from __future__ import annotations

import html
import json
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

from .analysis import FakeoutAnalysisResult, FakeoutConfig
from .cpi_loader import CPIRelease

DEFAULT_DASHBOARD_PATH = Path("data") / "analysis_dashboard.html"

__all__ = ["DEFAULT_DASHBOARD_PATH", "render_dashboard_html"]


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
        if not events.empty:
            events = events.copy()
        # Ensure release datetime is timezone-aware and sorted
        if "release_datetime" in events.columns:
            events["release_datetime"] = pd.to_datetime(events["release_datetime"], utc=True)
            events.sort_values("release_datetime", inplace=True)
            events.reset_index(drop=True, inplace=True)

        # compute percent columns for returns
        for label in self._reaction_labels + self._evaluation_labels:
            col = f"return_{label}"
            if col in events.columns:
                events[f"return_{label}_pct"] = events[col] * 100.0

        # map of fake columns for each combination
        fake_column_map: dict[tuple[str, str], str] = {}
        for reaction_label in self._reaction_labels:
            for evaluation_label in self._evaluation_labels:
                column_name = f"fake_{reaction_label}_{evaluation_label}"
                if column_name in events.columns:
                    fake_column_map[(reaction_label, evaluation_label)] = column_name

        # aggregated flags by evaluation window
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

        # aggregated flags by reaction window
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

        # compute fake move duration (minutes until reversal)
        evaluation_minutes = {
            label: _timedelta_minutes(delta) for label, delta in self._evaluation_windows
        }
        durations: list[int | None] = []
        for idx, row in events.iterrows():
            minutes_candidates: list[int] = []
            for evaluation_label in self._evaluation_labels:
                minute_value = evaluation_minutes.get(evaluation_label)
                if minute_value is None:
                    continue
                for reaction_label in self._reaction_labels:
                    column_name = fake_column_map.get((reaction_label, evaluation_label))
                    if column_name is None:
                        continue
                    value = row.get(column_name)
                    if value is True:
                        minutes_candidates.append(minute_value)
            durations.append(min(minutes_candidates) if minutes_candidates else None)
        if durations:
            events["fake_duration_minutes"] = pd.Series(durations, index=events.index)

        # categorize surprises
        events["surprise_type"] = events["cpi_surprise"].apply(
            lambda value: _categorize_surprise(value, neutral_band=self._neutral_surprise_band)
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

        if events.empty:
            self._figures = figures
            return

        # Fake-out rates by evaluation window
        fake_by_eval_records = []
        for evaluation_label in self._evaluation_labels:
            column = f"fake_by_evaluation_{evaluation_label}"
            if column not in events.columns:
                continue
            valid = events[column].dropna()
            rate = float(valid.mean()) if not valid.empty else math.nan
            fake_by_eval_records.append({
                "evaluation_label": evaluation_label,
                "fake_rate": rate,
                "count": int(valid.size),
            })
        if fake_by_eval_records:
            fake_by_eval_df = pd.DataFrame(fake_by_eval_records)
            fake_by_eval_df["fake_rate_pct"] = fake_by_eval_df["fake_rate"] * 100.0
            fake_by_eval_df["label"] = fake_by_eval_df["fake_rate_pct"].apply(
                lambda value: "n/a" if pd.isna(value) else f"{value:.1f}%"
            )
            fake_rate_fig = px.bar(
                fake_by_eval_df,
                x="evaluation_label",
                y="fake_rate_pct",
                text="label",
            )
            fake_rate_fig.update_traces(textposition="outside")
            fake_rate_fig.update_yaxes(title="Fake-out rate (%)")
            fake_rate_fig.update_xaxes(title="Evaluation window")
            fake_rate_fig.update_layout(title="Fake-out rates by evaluation window")
            figures["fake_by_window"] = self._serialize_figure(fake_rate_fig)

        # Fake-out rates by surprise type (grouped)
        grouped_records = []
        for evaluation_label in self._evaluation_labels:
            column = f"fake_by_evaluation_{evaluation_label}"
            if column not in events.columns:
                continue
            for surprise_type, group in events.groupby("surprise_type", dropna=False):
                valid = group[column].dropna()
                if valid.empty:
                    continue
                grouped_records.append({
                    "evaluation_label": evaluation_label,
                    "surprise_type": surprise_type,
                    "fake_rate": float(valid.mean()),
                    "count": int(valid.size),
                })
        if grouped_records:
            grouped_df = pd.DataFrame(grouped_records)
            grouped_df["fake_rate_pct"] = grouped_df["fake_rate"] * 100.0
            grouped_fig = px.bar(
                grouped_df,
                x="surprise_type",
                y="fake_rate_pct",
                color="evaluation_label",
                barmode="group",
            )
            grouped_fig.update_yaxes(title="Fake-out rate (%)")
            grouped_fig.update_xaxes(title="Surprise type")
            grouped_fig.update_layout(title="Fake-out rates by surprise type")
            figures["fake_by_surprise"] = self._serialize_figure(grouped_fig)

        # Timeline scatter of CPI releases
        if self._primary_evaluation:
            timeline_df = events.copy()
            outcome_labels = []
            column = f"fake_by_evaluation_{self._primary_evaluation}"
            for value in timeline_df.get(column, pd.Series(dtype="boolean")):
                if pd.isna(value):
                    outcome_labels.append("Incomplete")
                elif value:
                    outcome_labels.append("Fake-out")
                else:
                    outcome_labels.append("Sustained")
            timeline_df["outcome"] = outcome_labels
            return_column = f"return_{self._primary_evaluation}_pct"
            if return_column in timeline_df.columns:
                timeline_fig = px.scatter(
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
                timeline_fig.update_traces(marker=dict(size=11), selector=dict(mode="markers"))
                timeline_fig.update_layout(
                    title=f"Timeline of CPI releases ({self._primary_evaluation} outcome)",
                    yaxis_title=f"Return {self._primary_evaluation} (%)",
                    xaxis_title="Release datetime",
                )
                figures["timeline"] = self._serialize_figure(timeline_fig)

        # Distribution of CPI surprises
        if not events["cpi_surprise"].dropna().empty:
            nbins = min(30, max(10, int(math.sqrt(len(events)))))
            surprise_hist = px.histogram(
                events,
                x="cpi_surprise",
                nbins=nbins,
            )
            surprise_hist.update_layout(
                title="Distribution of CPI surprises",
                xaxis_title="Surprise (actual - expected, %)",
                yaxis_title="Count",
            )
            figures["surprise_distribution"] = self._serialize_figure(surprise_hist)

        # Price reaction vs surprise magnitude
        if self._primary_reaction:
            reaction_column = f"return_{self._primary_reaction}_pct"
            scatter_df = events.dropna(subset=["cpi_surprise", reaction_column])
            if not scatter_df.empty:
                scatter_fig = px.scatter(
                    scatter_df,
                    x="cpi_surprise",
                    y=reaction_column,
                    hover_name="release_datetime",
                )
                if len(scatter_df) >= 2:
                    x_vals = scatter_df["cpi_surprise"].to_numpy(float)
                    y_vals = scatter_df[reaction_column].to_numpy(float)
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
                    trendline = go.Scatter(
                        x=x_range,
                        y=slope * x_range + intercept,
                        mode="lines",
                        name="Trendline",
                        line=dict(color="#EF553B", width=2),
                    )
                    scatter_fig.add_trace(trendline)
                scatter_fig.update_layout(
                    title=f"Price reaction vs CPI surprise ({self._primary_reaction})",
                    xaxis_title="CPI surprise (%)",
                    yaxis_title=f"Return {self._primary_reaction} (%)",
                )
                figures["reaction_vs_surprise"] = self._serialize_figure(scatter_fig)

        # Fake-out probability vs surprise magnitude
        if self._primary_evaluation:
            fake_column = f"fake_by_evaluation_{self._primary_evaluation}"
            if fake_column in events.columns:
                fake_df = events.dropna(subset=["cpi_surprise", fake_column])
                if not fake_df.empty:
                    surprise_values = fake_df["cpi_surprise"].to_numpy(float)
                    unique_surprise = float(surprise_values.max() - surprise_values.min())
                    if unique_surprise == 0:
                        bins = np.array([surprise_values.min() - 0.5, surprise_values.max() + 0.5])
                    else:
                        bin_count = min(12, max(4, int(math.sqrt(len(fake_df)))))
                        bins = np.linspace(surprise_values.min(), surprise_values.max(), bin_count + 1)
                    fake_df = fake_df.assign(
                        surprise_bin=pd.cut(fake_df["cpi_surprise"], bins=bins, include_lowest=True)
                    )
                    grouped = (
                        fake_df.groupby("surprise_bin", observed=True)
                        .agg(
                            fake_rate=(fake_column, "mean"),
                            sample_size=(fake_column, "size"),
                            avg_surprise=("cpi_surprise", "mean"),
                        )
                        .dropna()
                    )
                    if not grouped.empty:
                        grouped["fake_rate_pct"] = grouped["fake_rate"] * 100.0
                        fake_prob_fig = px.scatter(
                            grouped,
                            x="avg_surprise",
                            y="fake_rate_pct",
                            size="sample_size",
                            hover_data={"sample_size": True},
                        )
                        fake_prob_fig.update_layout(
                            title=f"Fake-out probability vs surprise ({self._primary_evaluation})",
                            xaxis_title="Average surprise in bin (%)",
                            yaxis_title="Fake-out rate (%)",
                        )
                        figures["fake_probability"] = self._serialize_figure(fake_prob_fig)

        # Fake move duration distribution (histogram + box)
        if "fake_duration_minutes" in events.columns:
            duration_df = events.dropna(subset=["fake_duration_minutes"])
            if not duration_df.empty:
                subplot_fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box plot"))
                histogram = go.Histogram(
                    x=duration_df["fake_duration_minutes"],
                    nbinsx=min(20, max(5, int(len(duration_df) ** 0.5))),
                    name="Duration",
                    marker_color="#636EFA",
                    opacity=0.75,
                )
                box = go.Box(
                    x=duration_df["fake_duration_minutes"],
                    name="",
                    marker_color="#EF553B",
                    boxmean=True,
                    orientation="h",
                )
                subplot_fig.add_trace(histogram, row=1, col=1)
                subplot_fig.add_trace(box, row=1, col=2)
                subplot_fig.update_xaxes(title_text="Minutes", row=1, col=1)
                subplot_fig.update_yaxes(title_text="Count", row=1, col=1)
                subplot_fig.update_xaxes(title_text="Minutes", row=1, col=2)
                subplot_fig.update_layout(title="Distribution of fake move durations", showlegend=False)
                figures["fake_durations"] = self._serialize_figure(subplot_fig)

            # Average reversal time by surprise type
            grouped_duration = (
                duration_df.groupby("surprise_type", dropna=False)["fake_duration_minutes"].mean()
            )
            if not grouped_duration.empty:
                duration_records = grouped_duration.reset_index(name="avg_minutes")
                duration_records["avg_minutes"] = duration_records["avg_minutes"].astype(float)
                duration_fig = px.bar(
                    duration_records,
                    x="surprise_type",
                    y="avg_minutes",
                )
                duration_fig.update_layout(
                    title="Average reversal time by surprise type",
                    xaxis_title="Surprise type",
                    yaxis_title="Average minutes",
                )
                figures["duration_by_surprise"] = self._serialize_figure(duration_fig)

        # Price movement trajectories for fake vs sustained moves
        time_records = []
        combined_windows = []
        for label, delta in self._reaction_windows + self._evaluation_windows:
            combined_windows.append((label, delta))
        seen_minutes: set[int] = set()
        ordered_windows: list[tuple[str, timedelta]] = []
        for label, delta in sorted(combined_windows, key=lambda item: item[1]):
            minutes = _timedelta_minutes(delta)
            if minutes in seen_minutes:
                continue
            seen_minutes.add(minutes)
            ordered_windows.append((label, delta))
        for idx, row in events.iterrows():
            category = "Fake move" if row.get("fake_any") is True else "Sustained move"
            if pd.isna(row.get("fake_any")):
                category = "Incomplete"
            for label, delta in ordered_windows:
                col = f"return_{label}_pct"
                if col not in events.columns:
                    continue
                value = row.get(col)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                time_records.append({
                    "category": category,
                    "minutes": _timedelta_minutes(delta),
                    "label": label,
                    "return_pct": float(value),
                })
        if time_records:
            trajectory_df = pd.DataFrame(time_records)
            trajectory_df = trajectory_df[trajectory_df["category"] != "Incomplete"]
            if not trajectory_df.empty:
                trajectory_summary = (
                    trajectory_df.groupby(["category", "minutes", "label"], as_index=False)["return_pct"].mean()
                )
                trajectory_fig = px.line(
                    trajectory_summary,
                    x="minutes",
                    y="return_pct",
                    color="category",
                    markers=True,
                    hover_data={"label": True},
                )
                trajectory_fig.update_layout(
                    title="Average price trajectory",
                    xaxis_title="Minutes from release",
                    yaxis_title="Return (%)",
                )
                figures["price_trajectories"] = self._serialize_figure(trajectory_fig)

        self._figures = figures

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
        chartNode.innerHTML = '<p style="color: var(--text-secondary);">No data available for this chart.</p>';
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
