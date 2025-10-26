from __future__ import annotations

from datetime import datetime, timedelta, timezone

from btc_cpi_backtest.cpi_loader import CPIRelease
from btc_cpi_backtest.market_data import (
    CandleFetcher,
    build_release_windows,
    download_cpi_market_data,
)


class FakeExchange:
    def __init__(self, responses: list[list[list[float]]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, int | str]] = []
        self.rateLimit = 200
        self.closed = False

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        self.calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "since": since,
            "limit": limit,
        })
        if self.responses:
            return self.responses.pop(0)
        return []

    def close(self) -> None:
        self.closed = True


def _mk_timestamp(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_candle_fetcher_handles_pagination(tmp_path) -> None:
    start = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=3)

    responses = [
        [
            [_mk_timestamp(start), 100, 110, 90, 105, 1.0],
            [_mk_timestamp(start + timedelta(minutes=1)), 105, 115, 95, 110, 1.2],
        ],
        [
            [_mk_timestamp(start + timedelta(minutes=2)), 110, 120, 100, 115, 0.8],
            [_mk_timestamp(start + timedelta(minutes=3)), 115, 125, 105, 118, 0.6],
        ],
    ]

    exchange = FakeExchange(responses)
    fetcher = CandleFetcher(
        exchange,
        symbol="BTC/USDT",
        timeframe="1m",
        cache_dir=tmp_path,
        batch_limit=2,
    )

    df, downloaded, cache_path = fetcher.fetch_window(start, end)

    assert downloaded is True
    assert len(df) == 4
    assert df["timestamp"].iloc[0] == start
    assert df["timestamp"].iloc[-1] == start + timedelta(minutes=3)
    assert cache_path.exists()
    assert len(exchange.calls) == 2
    assert exchange.calls[0]["since"] == _mk_timestamp(start)


def test_candle_fetcher_uses_cache(tmp_path) -> None:
    start = datetime(2022, 1, 2, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=1)
    first_responses = [[[_mk_timestamp(start), 120, 125, 115, 123, 0.5]]]

    first_exchange = FakeExchange(first_responses)
    fetcher = CandleFetcher(
        first_exchange,
        symbol="BTC/USDT",
        timeframe="1m",
        cache_dir=tmp_path,
    )
    df_first, downloaded_first, cache_path = fetcher.fetch_window(start, end)

    assert downloaded_first is True
    assert len(df_first) == 1
    assert cache_path.exists()

    second_exchange = FakeExchange([])
    fetcher_cached = CandleFetcher(
        second_exchange,
        symbol="BTC/USDT",
        timeframe="1m",
        cache_dir=tmp_path,
    )
    df_second, downloaded_second, _ = fetcher_cached.fetch_window(start, end)

    assert downloaded_second is False
    assert df_second.equals(df_first)
    assert second_exchange.calls == []


def test_build_release_windows_applies_offsets() -> None:
    release_dt = datetime(2023, 5, 10, 12, 30, tzinfo=timezone.utc)
    release = CPIRelease(release_datetime=release_dt, actual=3.2, expected=3.1)

    windows = build_release_windows([release])

    assert len(windows) == 1
    assert windows[0].start == release_dt - timedelta(minutes=5)
    assert windows[0].end == release_dt + timedelta(hours=4)
    assert windows[0].release is release


def test_download_cpi_market_data_summary(tmp_path) -> None:
    release_a = CPIRelease(
        release_datetime=datetime(2023, 1, 1, 13, 30, tzinfo=timezone.utc),
        actual=6.5,
        expected=6.4,
    )
    release_b = CPIRelease(
        release_datetime=datetime(2023, 2, 1, 13, 30, tzinfo=timezone.utc),
        actual=6.4,
        expected=6.2,
    )

    responses = [
        [[_mk_timestamp(release_a.release_datetime), 100, 101, 99, 100.5, 1.0]],
        [],
        [[_mk_timestamp(release_b.release_datetime), 101, 102, 100, 101.5, 1.2]],
        [],
    ]
    exchange = FakeExchange(responses)

    summary = download_cpi_market_data(
        exchange_name="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        releases=[release_a, release_b],
        cache_dir=tmp_path,
        exchange_client=exchange,
    )

    assert summary.total_releases == 2
    assert summary.downloaded == 2
    assert summary.reused == 0
    assert summary.total_candles == 2
    assert exchange.closed is False

    reuse_exchange = FakeExchange([])
    reuse_summary = download_cpi_market_data(
        exchange_name="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        releases=[release_a, release_b],
        cache_dir=tmp_path,
        exchange_client=reuse_exchange,
    )

    assert reuse_summary.total_releases == 2
    assert reuse_summary.downloaded == 0
    assert reuse_summary.reused == 2
    assert reuse_summary.total_candles == 2
    assert reuse_exchange.calls == []
