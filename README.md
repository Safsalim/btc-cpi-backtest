# BTC CPI Backtest

Modern Python scaffolding for exploring Bitcoin performance relative to consumer price index (CPI) data. The
repository is prepared for backtesting experiments, data exploration, and future strategy research.

## Project Structure

```
.
├── data/                     # Source datasets and exported artifacts
├── src/
│   └── btc_cpi_backtest/     # Python package with CLI entry points and utilities
├── tests/                    # Automated tests
├── .env.example              # Template for environment variables
└── pyproject.toml            # Package metadata and dependencies
```

Existing CSV datasets have been moved into the `data/` directory for convenience. Future scripts can read
and write files inside this folder, which is not ignored by default so that curated datasets can be versioned.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Optional: a virtual environment manager such as `venv`, `virtualenv`, or `conda`

### Installation

1. Create and activate a virtual environment.
2. Install the package and its dependencies in editable mode:

   ```bash
   pip install -e .[dev]
   ```

3. (Optional) Install visualization extras:

   ```bash
   pip install -e .[visualization]
   ```

### Environment Configuration

Some integrations (e.g., authenticated CCXT endpoints) require API credentials. Start by copying the example
file and filling in the necessary values:

```bash
cp .env.example .env
```

The project uses [python-dotenv](https://pypi.org/project/python-dotenv/) and Pydantic settings to load
environment variables automatically.

## Command Line Interface

The Typer-based CLI exposes placeholder commands that will be expanded in upcoming iterations.

```bash
python -m btc_cpi_backtest --help
btc-cpi-backtest fetch-data --help
btc-cpi-backtest analyze --help
```

- `fetch-data` — lays the groundwork for downloading OHLCV data from exchanges via CCXT
- `analyze` — placeholder for backtesting logic comparing BTC performance to CPI benchmarks

## Running Tests

The project uses `pytest` for testing. After installing the development dependencies, run:

```bash
pytest
```

## Next Steps

- Implement data fetching using CCXT and caching raw datasets in `data/`
- Add CPI data ingestion pipelines and normalization utilities
- Flesh out strategy backtests, metrics, and reporting capabilities

Contributions and ideas are welcome—feel free to open issues or submit pull requests as the project evolves.
