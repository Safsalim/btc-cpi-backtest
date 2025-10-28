# BTC CPI Backtest

Utilities for exploring how Bitcoin price action reacts to Consumer Price Index (CPI) releases. The
project bundles a Typer-based command-line interface (CLI), sample CPI and BTC datasets, and reusable
loading/analysis helpers so you can experiment quickly or plug in your own data feeds.

## At a glance

- **Language & tooling:** Python 3.10+, [Typer](https://typer.tiangolo.com/), [Pandas](https://pandas.pydata.org/),
  [CCXT](https://github.com/ccxt/ccxt).
- **Core workflows:** fetch BTC OHLCV candles, ingest CPI releases, run a fake-move analysis comparing
  short-term vs. follow-up returns.
- **Sample assets:** bundled CPI and BTC CSV files under `data/` to confirm the tooling end to end.
- **Extensibility:** utilities in `src/btc_cpi_backtest/` can be reused from notebooks or other Python code.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data directory & sample formats](#data-directory--sample-formats)
4. [Environment configuration & API keys](#environment-configuration--api-keys)
5. [Command-line usage](#command-line-usage)
   - [`fetch-data`](#fetching-btc-candles-fetch-data)
   - [`analyze`](#running-the-fake-move-analysis-analyze)
   - [Other utilities](#other-cli-utilities)
   - [Caching behaviour](#caching-behaviour)
6. [Expected outputs](#expected-outputs)
7. [Troubleshooting & tips](#troubleshooting--tips)
8. [Running the analysis from Python](#running-the-analysis-from-python)

---

## Prerequisites

- Python **3.10 or newer**.
- A shell environment with `pip` and `venv` (or your preferred virtual environment manager).
- Optional (for plots): a display backend that supports Matplotlib if you plan to use `--plot`.
- Optional (for authenticated data fetches): API credentials for the exchange you query via CCXT.

## Installation

1. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the project in editable mode**

   ```bash
   pip install -U pip
   pip install -e .[dev]
   ```

   > **Important dependency pinning**
   >
   > - Settings management relies on [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
   >   Installing the project pulls in compatible versions of both `pydantic` and `pydantic-settings`.
   > - Some versions of Typer/Click do not yet support `typing.Sequence` options. If you encounter the
   >   error `RuntimeError: Type not yet supported: typing.Sequence[str]`, see the
   >   [Troubleshooting](#troubleshooting--tips) section for workarounds.

3. (Optional) **Install visualization extras**

   ```bash
   pip install -e .[visualization]
   ```

After installing, you should be able to inspect the CLI entry point:

```bash
$ btc-cpi-backtest --help
```

If Typer raises a compatibility error, skip ahead to the troubleshooting guidance or use the Python helper
script described later in this document.

## Data directory & sample formats

The repository tracks curated datasets inside the `data/` directory so you can explore the workflow without
needing external data immediately.

```
.
├── data/
│   ├── btc_price_sample.csv      # Bundled BTC candle sample
│   └── cpi_sample.csv            # Bundled CPI releases (UTC timestamps)
```

### CPI data format specification

The CPI loader expects at minimum timestamp, actual, and expected columns. Optional columns (previous and
surprise) are used when available.

| Column         | Required | Example value              | Notes                                                                 |
| -------------- | -------- | -------------------------- | --------------------------------------------------------------------- |
| `release_time` | ✅       | `2023-11-14T13:30:00+00:00` | ISO 8601 string. Provide timezone-aware values or use `--cpi-timezone` |
| `actual`       | ✅       | `3.2`                      | The realized CPI reading (percentage, not decimal)                    |
| `expected`     | ✅       | `3.3`                      | Consensus forecast value                                              |
| `previous`     | optional | `3.7`                      | Prior release value (if present)                                      |
| `surprise`     | optional | `-0.1`                     | Pre-computed surprise. If omitted the CLI computes `actual - expected`|

Sample file excerpt (`data/cpi_sample.csv`):

```csv
release_time,actual,expected,previous,surprise
2023-10-12T12:30:00+00:00,3.7,3.6,3.7,0.1
2023-11-14T13:30:00+00:00,3.2,3.3,3.7,-0.1
2023-12-12T13:30:00+00:00,3.1,3.1,3.2,0.0
```

These timestamps are already in UTC, so no additional timezone flag is required when loading them.

### BTC price data format specification

Price series are treated as a Pandas Series indexed by timezone-aware timestamps.

| Column      | Required | Example value              | Notes                                                                    |
| ----------- | -------- | -------------------------- | ------------------------------------------------------------------------ |
| `timestamp` | ✅       | `2023-11-14T13:30:00+00:00` | ISO 8601 string. Provide timezone-aware values or use `--price-timezone`. |
| `close`     | ✅       | `34000`                    | Closing price for the candle. Parsed as floating point.                  |

Bundled sample (`data/btc_price_sample.csv`) contains short intraday windows around each CPI release in the
sample CPI file. This dataset is sufficient for running the demo analysis.

## Environment configuration & API keys

Authenticated CCXT requests (for private endpoints or elevated rate limits) require API credentials. Copy the
sample environment file and populate the values:

```bash
cp .env.example .env
```

Supported keys:

- `CCXT_API_KEY`
- `CCXT_API_SECRET`
- `CCXT_PASSWORD` (for exchanges that require a password/phrase)

The CLI automatically loads environment variables using `python-dotenv`, so any values in `.env` are picked up
when commands run.

## Command-line usage

Run `btc-cpi-backtest --help` for the top-level command list. You can append `--help` to any sub-command to see
all available options. All commands support `--log-level` (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`).

> **Note on Typer compatibility**
> The CLI defines several options that accept multiple values (e.g., `--reaction-window`). Certain Typer/Click
> combinations emit `RuntimeError: Type not yet supported: typing.Sequence[str]`. Until Typer officially supports
> `typing.Sequence`, pinning to Typer ≥0.12 with Click 8.1 still exposes this limitation. See
> [Troubleshooting](#troubleshooting--tips) for workarounds if you need to run the command immediately.

### Fetching BTC candles (`fetch-data`)

```bash
btc-cpi-backtest fetch-data \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1h \
  --limit 500 \
  --destination data/binance_btcusdt_1h.csv \
  --sandbox
```

Arguments & flags:

| Option             | Default                     | Description                                                                                  |
| ------------------ | --------------------------- | -------------------------------------------------------------------------------------------- |
| `--exchange/-e`    | `binance`                   | Any CCXT-supported exchange id.                                                              |
| `--symbol/-s`      | `BTC/USDT`                  | Trading pair to download.                                                                    |
| `--timeframe/-t`   | `1d`                        | CCXT timeframe (e.g., `1m`, `1h`, `1d`).                                                     |
| `--limit/-n`       | `365`                       | Candle count requested from the exchange.                                                    |
| `--destination/-d` | `data/btc_market_data.csv`  | Output CSV path. Intermediate directories are created automatically.                         |
| `--sandbox`        | `False`                     | Toggle exchange sandbox/testnet mode when available.                                         |

> The current implementation logs the request details but does **not** perform the download yet. It exists as a
> scaffold for future work. Use it today to confirm configuration and logging.

**Rate limits & API keys:** Exchanges commonly throttle unauthenticated bursts. If you observe CCXT
`NetworkError` or rate limit messages, lower `--limit`, increase the timeframe, or supply API keys via `.env`.

### Running the fake-move analysis (`analyze`)

The `analyze` command combines CPI releases and BTC close prices to highlight potential fake moves (sharp initial
reaction followed by an opposing move within configurable windows).

```bash
btc-cpi-backtest analyze \
  --cpi-data data/cpi_releases.csv \
  --btc-data btcusd_1-min_data.csv \
  --output data/fakeout_analysis.csv \
  --output-format csv \
  --reaction-window 10m \
  --reaction-window 30m \
  --evaluation-window 2h \
  --plot


Key options:

| Option                    | Default                     | Description                                                                                                   |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--strategy/-s`           | `fake-moving-average`       | Free-form label for the run; currently informational.                                                         |
| `--lookback-days/-n`      | `365`                       | How many days of price history to retain before the earliest CPI release.                                     |
| `--cpi-data/-c`           | `data/cpi_releases.csv`     | Path to a CPI CSV file (use `sample` for the bundled dataset).                                                |
| `--btc-data/-b`           | `btcusd_1-min_data.csv`     | Path to a BTC OHLCV CSV file (use `sample` for the bundled dataset).                                          |
| `--cpi-timestamp-column`  | `release_time`              | CPI timestamp column name.                                                                                    |
| `--cpi-actual-column`     | `actual`                    | CPI actual value column.                                                                                      |
| `--cpi-expected-column`   | `expected`                  | CPI expected/consensus column.                                                                                |
| `--cpi-surprise-column`   | `surprise`                  | CPI surprise column (set to `none` when absent).                                                              |
| `--cpi-timezone`          | `UTC`                       | Timezone applied to naive CPI timestamps (use `none` to leave naive).                                         |
| `--btc-timestamp-column`  | `Timestamp`                 | BTC candle timestamp column.                                                                                  |
| `--btc-close-column`      | `Close`                     | BTC close column used for analysis.                                                                           |
| `--btc-timezone`          | `UTC`                       | Timezone for naive BTC timestamps.                                                                            |
| `--reaction-window`       | defaults: `5m`, `15m`, `30m`| Override reaction windows; values must end with `m` or `h` (e.g., `--reaction-window 45m`).                    |
| `--evaluation-window`     | defaults: `1h`, `2h`, `4h`  | Override evaluation windows; same format as reaction windows.                                                 |
| `--output/-o`             | `data/fakeout_analysis.csv` | Destination for the per-release report.                                                                       |
| `--output-format/-f`      | `csv`                       | `csv` or `json`.                                                                                              |
| `--html-output`           | `data/analysis_dashboard.html` | Path for the interactive HTML dashboard (`none` to skip generation).                                           |
| `--open-html/--no-open-html` | `--no-open-html`          | Automatically open the generated dashboard in the default browser.                                            |
| `--dashboard-backend`     | `plotly-cdn`                | How Plotly.js is provided: `plotly-cdn` (default), `plotly-inline` (embed bundle), or `mpld3` (matplotlib fallback). |
| `--plot/--no-plot`        | `False`                     | Generate a Matplotlib line plot of evaluation window returns (requires visualization extras).                 |

By default the command reads from `btcusd_1-min_data.csv` (minute-level BTC candles stored under `data/btcusd_1-min_data.csv`) and `data/cpi_releases.csv`.
Use `--cpi-data sample --btc-data sample` to run against the lightweight bundled datasets.
The `analyze` command also writes an interactive dashboard to `data/analysis_dashboard.html` unless you pass
`--html-output none`; add `--open-html` to launch it in your browser automatically once generation completes.
By default, Plotly.js is loaded from the CDN to avoid bundle download errors. Use `--dashboard-backend plotly-inline`
to embed the library for offline viewing or `--dashboard-backend mpld3` to generate the simplified matplotlib-based
fallback when Plotly is unavailable. Ensure the price series spans the requested lookback and evaluation windows; if
coverage is incomplete the CLI logs warnings and affected returns are reported as missing rather than aborting the run.

### Other CLI utilities

- `btc-cpi-backtest cpi-summary --help`

  Quick validation tool that prints how many CPI releases were parsed, the date range, and highlights missing
  columns or timezone issues. Use it whenever you ingest a new CPI feed.

- Global log level: `--log-level DEBUG` is available on every command for more verbose diagnostics.

### Caching behaviour

Price series loading is cached in-memory within a process using an `lru_cache` keyed by the tuple
`(source, timestamp_column, close_column, timezone)`. Repeated `analyze` invocations that point to the same
inputs reuse the Series without hitting disk again.

- To force a reload within the same Python process, change any of the cache key parameters (e.g., pass the
  absolute path to the price CSV or tweak the timezone argument).
- The cache is per-process; restarting the CLI or your Python interpreter always loads fresh data.

## Expected outputs

Running the sample dataset (with a compatible Typer version or via the Python helper script) yields a summary
similar to the following:

```
Loaded 3 CPI releases from bundled sample dataset.
Using 21 BTC price points from bundled sample price dataset.
Analysis complete for 3 CPI releases.
Fake-out rates:
  5m -> 1h: 2/3 (66.7%)
  5m -> 2h: 3/3 (100.0%)
  5m -> 4h: 2/3 (66.7%)
  15m -> 1h: 3/3 (100.0%)
  15m -> 2h: 2/3 (66.7%)
  15m -> 4h: 3/3 (100.0%)
  30m -> 1h: 2/3 (66.7%)
  30m -> 2h: 3/3 (100.0%)
  30m -> 4h: 2/3 (66.7%)
Average returns (%):
  Reaction windows:
    5m: 0.21%
    15m: -0.33%
    30m: 0.02%
  Evaluation windows:
    1h: 0.54%
    2h: 0.06%
    4h: 1.71%
Correlation with CPI surprise:
  1h: -0.863
  2h: -0.952
  4h: -0.938
Saved detailed report to data/fakeout_analysis.csv
```

The CLI also writes a Plotly-powered dashboard to `data/analysis_dashboard.html` (unless disabled) with interactive charts,
filters, and per-release drilldowns that mirror the metrics above. Use `--dashboard-backend` if you prefer an inline bundle or the mpld3 fallback.

The CSV (or JSON) report contains per-release metrics such as base price, returns for each window, and boolean
flags that indicate whether the move qualifies as a fake-out (`fake_15m_4h`, etc.).

## Troubleshooting & tips

| Issue / Symptom                                                                                           | Recommendation                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError: Type not yet supported: typing.Sequence[str]` or `Secondary flag is not valid…`            | Typer/Click version mismatch. Until Typer adds native `Sequence` support, run the helper script described below or monkey-patch the CLI by replacing `Sequence` with `List` locally. |
| `pydantic.errors.PydanticImportError: BaseSettings has been moved…`                                       | Install `pydantic<2`. Example: `pip install 'pydantic<2'`.                                                                                                               |
| CCXT `NetworkError`, `RequestTimeout`, or HTTP 429 rate limits                                            | Reduce `--limit`, expand `--timeframe`, toggle `--sandbox`, or provide API credentials in `.env`. Consider adding sleeps between repeated runs to respect exchange limits. |
| CPI timestamps parsed as naive / loader raises `Missing required column(s)`                               | Check column names against the specification. Use `--cpi-timezone US/Eastern` when your CSV omits offsets, or preprocess to ISO 8601 with offsets.                         |
| Price dataset does not cover requested window / analysis exits early                                     | Ensure your price CSV includes data from `lookback_days` before the first release through the largest evaluation horizon after the final release.                          |
| Generated report overwrites previous runs                                                                 | Point `--output` to a unique path per experiment or include markers in the filename (e.g., `data/reports/fakeout_%Y%m%d.csv`).                                             |
| Want to change candle granularity or window definitions                                                   | Use `--timeframe`/`--limit` on `fetch-data`, and `--reaction-window` / `--evaluation-window` on `analyze`. Window values must end with `m` (minutes) or `h` (hours).        |

## Running the analysis from Python

If Typer compatibility blocks the CLI from launching, you can run the same analysis through a short Python script
(jupyter notebook friendly):

```python
from btc_cpi_backtest.cpi_loader import load_sample_cpi_data
from btc_cpi_backtest.price_loader import load_sample_price_series
from btc_cpi_backtest.analysis import analyze_fakeouts

releases = load_sample_cpi_data()  # or load_cpi_from_csv("/path/to/your_cpi.csv", input_timezone="UTC")
prices = load_sample_price_series()  # or load_price_series_from_csv("/path/to/your_prices.csv")

result = analyze_fakeouts(releases, prices)
print(result.summary)
result.events.to_csv("data/fakeout_analysis.csv", index=False)
```

This produces the same summary and detailed report that the CLI would emit once Typer resolves the `Sequence`
limitation. You can tweak `FakeoutConfig` parameters programmatically if you need custom reaction/evaluation
windows.

## Next steps

- Replace the placeholder `fetch-data` implementation with real CCXT integration and retry logic.
- Expand CPI ingestion helpers with direct API clients (e.g., BLS, FRED) and normalization utilities.
- Enhance reporting (interactive dashboards, richer visualizations) once the analysis pipeline stabilizes.

Happy experimenting!
