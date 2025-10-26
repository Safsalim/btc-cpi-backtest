from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from btc_cpi_backtest.analysis import FakeoutConfig, analyze_fakeouts
from btc_cpi_backtest.cpi_loader import CPIRelease


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _build_price_series(data: list[tuple[str, float]]) -> pd.Series:
    timestamps = [datetime.fromisoformat(ts).astimezone(timezone.utc) for ts, _ in data]
    index = pd.DatetimeIndex(timestamps)
    prices = [price for _, price in data]
    return pd.Series(prices, index=index)


def test_fakeout_analysis_classifies_reversals() -> None:
    releases = [
        CPIRelease(
            release_datetime=_ts("2024-01-10T13:30:00+00:00"),
            actual=3.6,
            expected=3.4,
        ),
        CPIRelease(
            release_datetime=_ts("2024-02-13T13:30:00+00:00"),
            actual=3.4,
            expected=3.5,
        ),
        CPIRelease(
            release_datetime=_ts("2024-03-12T12:30:00+00:00"),
            actual=3.5,
            expected=3.5,
        ),
    ]

    price_data = [
        ("2024-01-10T13:30:00+00:00", 100.0),
        ("2024-01-10T13:35:00+00:00", 102.0),
        ("2024-01-10T13:45:00+00:00", 101.0),
        ("2024-01-10T14:00:00+00:00", 100.5),
        ("2024-01-10T14:30:00+00:00", 98.0),
        ("2024-01-10T15:30:00+00:00", 97.5),
        ("2024-01-10T17:30:00+00:00", 99.0),
        ("2024-02-13T13:30:00+00:00", 200.0),
        ("2024-02-13T13:35:00+00:00", 195.0),
        ("2024-02-13T13:45:00+00:00", 194.5),
        ("2024-02-13T14:00:00+00:00", 193.0),
        ("2024-02-13T14:30:00+00:00", 205.0),
        ("2024-02-13T15:30:00+00:00", 210.0),
        ("2024-02-13T17:30:00+00:00", 220.0),
        ("2024-03-12T12:30:00+00:00", 300.0),
        ("2024-03-12T12:35:00+00:00", 303.0),
        ("2024-03-12T12:45:00+00:00", 304.0),
        ("2024-03-12T13:00:00+00:00", 305.0),
        ("2024-03-12T13:30:00+00:00", 308.0),
        ("2024-03-12T14:30:00+00:00", 310.0),
    ]
    price_series = _build_price_series(price_data)

    result = analyze_fakeouts(releases, price_series)

    events = result.events
    assert len(events) == 3

    # First release: positive 5m move, negative 1h move -> fake-out
    assert events.loc[0, "fake_5m_1h"] is True
    # Second release: negative 5m move, positive 1h move -> fake-out
    assert events.loc[1, "fake_5m_1h"] is True
    # Third release: sustained positive move -> not a fake-out
    assert events.loc[2, "fake_5m_1h"] is False

    # Missing 4h data for the third release should mark fake classification as None
    assert events.loc[2, "fake_5m_4h"] is None

    summary = result.summary
    stats_5m_1h = summary.fake_out_stats["5m"]["1h"]
    assert stats_5m_1h.total == 3
    assert stats_5m_1h.fake_count == 2

    # Average returns are expressed in decimal form
    avg_5m = summary.average_returns["reaction"]["5m"]
    assert pytest.approx(avg_5m, rel=1e-6) == ((0.02) + (-0.025) + (0.01)) / 3

    correlation = summary.surprise_correlations["1h"]
    assert not math.isnan(correlation)


def test_analyze_fakeouts_handles_missing_base_price() -> None:
    release = CPIRelease(
        release_datetime=_ts("2024-05-15T12:30:00+00:00"),
        actual=3.3,
        expected=3.2,
    )
    price_series = _build_price_series([
        ("2024-05-15T12:35:00+00:00", 400.0),
        ("2024-05-15T13:30:00+00:00", 402.0),
    ])

    result = analyze_fakeouts([release], price_series)
    events = result.events

    assert events.loc[0, "base_price"] is None
    assert events.loc[0, "return_5m"] is None
    assert events.loc[0, "fake_5m_1h"] is None

    stats = result.summary.fake_out_stats["5m"]["1h"]
    assert stats.total == 0
    assert stats.fake_count == 0


def test_custom_configuration_applies() -> None:
    release = CPIRelease(
        release_datetime=_ts("2024-06-12T13:30:00+00:00"),
        actual=3.0,
        expected=3.0,
    )
    price_series = _build_price_series([
        ("2024-06-12T13:30:00+00:00", 500.0),
        ("2024-06-12T13:40:00+00:00", 510.0),
        ("2024-06-12T14:30:00+00:00", 480.0),
    ])

    config = FakeoutConfig(
        reaction_windows=(("10m", timedelta(minutes=10)),),
        evaluation_windows=(("1h", timedelta(hours=1)),),
    )

    result = analyze_fakeouts([release], price_series, config=config)
    events = result.events
    assert "return_10m" in events.columns
    assert events.loc[0, "fake_10m_1h"] is True
