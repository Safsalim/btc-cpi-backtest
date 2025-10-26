from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import ccxt
import pandas as pd

from .cpi_loader import CPIRelease
from .settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CandleFetcher",
    "FetchSummary",
    "ReleaseWindow",
    "build_release_windows",
    "create_exchange_client",
    "download_cpi_market_data",
    "timeframe_to_seconds",
]

_TIMEFRAME_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}

DEFAULT_BATCH_LIMIT = 1000
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_PRE_WINDOW = timedelta(minutes=5)
DEFAULT_POST_WINDOW = timedelta(hours=4)


def timeframe_to_seconds(timeframe: str) -> int:
    """Translate a CCXT timeframe string to a number of seconds."""

    timeframe = timeframe.strip().lower()
    if not timeframe[:-1].isdigit() or timeframe[-1] not in _TIMEFRAME_UNIT_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    value = int(timeframe[:-1])
    if value <= 0:
        raise ValueError("Timeframe must be positive")

    unit_seconds = _TIMEFRAME_UNIT_SECONDS[timeframe[-1]]
    return value * unit_seconds


@dataclass(frozen=True)
class ReleaseWindow:
    """Calculated candle window around a CPI release."""

    release: CPIRelease
    start: datetime
    end: datetime


@dataclass(frozen=True)
class FetchSummary:
    """Summary information about a candle download session."""

    total_releases: int
    downloaded: int
    reused: int
    total_candles: int


class CandleFetcher:
    """Download OHLCV candles via CCXT with transparent caching."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        *,
        symbol: str,
        timeframe: str,
        cache_dir: Path,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_limit = batch_limit
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._timeframe_ms = timeframe_to_seconds(timeframe) * 1000

    def _cache_path(self, start: datetime, end: datetime) -> Path:
        symbol_part = self.symbol.replace("/", "-")
        start_part = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end_part = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{symbol_part}_{self.timeframe}_{start_part}_{end_part}.csv"
        return self.cache_dir / filename

    def fetch_window(self, start: datetime, end: datetime) -> tuple[pd.DataFrame, bool, Path]:
        """Fetch candles for the specified window, using cache when available."""

        if start >= end:
            raise ValueError("Start time must be before end time")

        cache_path = self._cache_path(start, end)
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            LOGGER.debug("Loaded %s candles from cache %s", len(df), cache_path)
            return df, False, cache_path

        candles = self._download(start, end)
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if df.empty:
            df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
            for column in ["open", "high", "low", "close", "volume"]:
                df[column] = pd.Series(dtype="float64")
            to_save = df.copy()
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            to_save = df.copy()
            to_save["timestamp"] = to_save["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        to_save.to_csv(cache_path, index=False)
        LOGGER.debug("Downloaded %s candles to %s", len(df), cache_path)
        return df, True, cache_path

    def _download(self, start: datetime, end: datetime) -> list[list[float]]:
        start_ms = math.floor(start.astimezone(timezone.utc).timestamp() * 1000)
        end_ms = math.floor(end.astimezone(timezone.utc).timestamp() * 1000)

        since = start_ms
        candles: list[list[float]] = []
        retries = 0

        while since <= end_ms:
            try:
                batch = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.batch_limit,
                )
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                retries += 1
                if retries > self.max_retries:
                    raise
                wait_seconds = self.exchange.rateLimit / 1000 if getattr(self.exchange, "rateLimit", None) else 1
                wait_seconds *= self.backoff_factor ** (retries - 1)
                LOGGER.warning(
                    "Temporary CCXT error fetching candles for %s (%s); retry %s/%s in %.2fs: %s",
                    self.symbol,
                    self.timeframe,
                    retries,
                    self.max_retries,
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)
                continue

            retries = 0
            if not batch:
                break

            for entry in batch:
                ts = int(entry[0])
                if ts < start_ms:
                    continue
                if ts > end_ms:
                    break
                if candles and ts <= int(candles[-1][0]):
                    continue
                candles.append(entry)

            last_ts = int(batch[-1][0])
            since = last_ts + self._timeframe_ms

            if last_ts >= end_ms:
                break

        return candles


def build_release_windows(
    releases: Sequence[CPIRelease],
    *,
    pre_window: timedelta = DEFAULT_PRE_WINDOW,
    post_window: timedelta = DEFAULT_POST_WINDOW,
) -> list[ReleaseWindow]:
    """Create windows spanning CPI releases for candle downloads."""

    windows: list[ReleaseWindow] = []
    for release in releases:
        start = release.release_datetime - pre_window
        end = release.release_datetime + post_window
        windows.append(ReleaseWindow(release=release, start=start, end=end))
    return windows


def create_exchange_client(
    exchange_name: str,
    *,
    settings: Optional[Settings] = None,
    sandbox: bool = False,
) -> ccxt.Exchange:
    """Instantiate a CCXT exchange client."""

    settings = settings or get_settings()

    try:
        exchange_class = getattr(ccxt, exchange_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown exchange: {exchange_name}") from exc

    if not isinstance(exchange_class, type):
        raise ValueError(f"Invalid exchange class for {exchange_name}")

    params: dict[str, object] = {"enableRateLimit": True}

    if settings.ccxt_api_key:
        params["apiKey"] = settings.ccxt_api_key
    if settings.ccxt_api_secret:
        params["secret"] = settings.ccxt_api_secret
    if settings.ccxt_password:
        params["password"] = settings.ccxt_password

    client: ccxt.Exchange = exchange_class(params)

    if sandbox and hasattr(client, "set_sandbox_mode"):
        client.set_sandbox_mode(True)

    return client


def download_cpi_market_data(
    *,
    exchange_name: str,
    symbol: str,
    timeframe: str,
    releases: Sequence[CPIRelease],
    cache_dir: Path,
    settings: Optional[Settings] = None,
    sandbox: bool = False,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    pre_window: timedelta = DEFAULT_PRE_WINDOW,
    post_window: timedelta = DEFAULT_POST_WINDOW,
    exchange_client: Optional[ccxt.Exchange] = None,
) -> FetchSummary:
    """Download BTC candles around CPI releases, returning a summary."""

    settings = settings or get_settings()

    start_filter = start.astimezone(timezone.utc) if start else None
    end_filter = end.astimezone(timezone.utc) if end else None

    filtered_releases = [
        release
        for release in releases
        if (start_filter is None or release.release_datetime >= start_filter)
        and (end_filter is None or release.release_datetime <= end_filter)
    ]
    filtered_releases = sorted(filtered_releases, key=lambda release: release.release_datetime)

    total_releases = len(filtered_releases)
    if total_releases == 0:
        LOGGER.info("No CPI releases matched the provided filters; nothing to download.")
        return FetchSummary(total_releases=0, downloaded=0, reused=0, total_candles=0)

    client = exchange_client or create_exchange_client(
        exchange_name, settings=settings, sandbox=sandbox
    )
    manage_client = exchange_client is None

    try:
        fetcher = CandleFetcher(
            client,
            symbol=symbol,
            timeframe=timeframe,
            cache_dir=cache_dir,
        )

        windows = build_release_windows(
            filtered_releases, pre_window=pre_window, post_window=post_window
        )

        downloaded_count = 0
        reused_count = 0
        total_candles = 0

        for window in windows:
            LOGGER.info(
                "Processing CPI release %s (symbol=%s, timeframe=%s)",
                window.release.release_datetime.isoformat(),
                symbol,
                timeframe,
            )
            df, downloaded, cache_path = fetcher.fetch_window(window.start, window.end)
            total_candles += len(df)
            if downloaded:
                downloaded_count += 1
                LOGGER.info(
                    "Downloaded %s candles to %s", len(df), cache_path
                )
            else:
                reused_count += 1
                LOGGER.info(
                    "Reused cached data at %s (%s candles)", cache_path, len(df)
                )

    finally:
        if manage_client and hasattr(client, "close"):
            client.close()

    return FetchSummary(
        total_releases=total_releases,
        downloaded=downloaded_count,
        reused=reused_count,
        total_candles=total_candles,
    )
