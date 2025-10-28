from __future__ import annotations

from pathlib import Path

from btc_cpi_backtest.analysis import FakeoutConfig, analyze_fakeouts
from btc_cpi_backtest.cpi_loader import load_sample_cpi_data
from btc_cpi_backtest.dashboard import DEFAULT_DASHBOARD_PATH, render_dashboard_html
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
    assert "tableRows" in html_content
    assert "figureMessages" in html_content
    assert "Plotly" in html_content
    assert "cdn.plot.ly" not in html_content
