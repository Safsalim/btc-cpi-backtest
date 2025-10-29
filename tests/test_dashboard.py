from __future__ import annotations

import json
from pathlib import Path

from plotly import io as pio

from btc_cpi_backtest.analysis import FakeoutConfig, analyze_fakeouts
from btc_cpi_backtest.cpi_loader import load_sample_cpi_data
from btc_cpi_backtest.dashboard import (
    DEFAULT_DASHBOARD_PATH,
    DashboardBuilder,
    render_dashboard_html,
)
from btc_cpi_backtest.price_loader import load_sample_price_series


def test_render_dashboard_html(tmp_path: Path) -> None:
    releases = load_sample_cpi_data()
    price_series = load_sample_price_series()

    result = analyze_fakeouts(releases, price_series, config=FakeoutConfig())

    output_path = tmp_path / DEFAULT_DASHBOARD_PATH.name
    generated = render_dashboard_html(
        result=result,
        releases=releases,
        price_series=price_series,
        config=FakeoutConfig(),
        output_path=output_path,
        open_browser=False,
    )

    assert generated.exists()
    html_content = output_path.read_text(encoding="utf-8")

    assert "BTC vs CPI Fake-out Dashboard" in html_content
    assert "fake_by_window" in html_content
    assert 'id="timeline"' in html_content
    assert "tableRows" in html_content
    assert "figureMessages" in html_content
    assert "Plotly" in html_content
    assert "cdn.plot.ly" not in html_content

    for removed_id in ("surprise_distribution", "reaction_vs_surprise", "fake_durations", "price_trajectories"):
        assert removed_id not in html_content


def test_individual_chart_builders(tmp_path: Path) -> None:
    releases = load_sample_cpi_data()
    price_series = load_sample_price_series()
    result = analyze_fakeouts(releases, price_series, config=FakeoutConfig())

    builder = DashboardBuilder(
        result=result,
        releases=releases,
        price_series=price_series,
        config=FakeoutConfig(),
        plotly_js_mode="inline",
    )

    html_output = builder.build_html()
    assert "timeline" in html_output

    expected_charts = {
        "timeline",
        "fake_by_window",
        "fake_probability",
        "fake_by_surprise",
        "duration_by_surprise",
    }
    assert expected_charts.issubset(builder._figures.keys())

    for chart_id in expected_charts:
        payload = builder._figures[chart_id]
        figure = pio.from_json(json.dumps(payload))
        assert figure.data, f"{chart_id} produced no traces"
        output_file = tmp_path / f"{chart_id}.html"
        figure.write_html(str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    removed_charts = {
        "surprise_distribution",
        "reaction_vs_surprise",
        "fake_durations",
        "price_trajectories",
    }
    assert removed_charts.isdisjoint(builder._figures.keys())
    assert removed_charts.isdisjoint(builder._figure_messages.keys())
