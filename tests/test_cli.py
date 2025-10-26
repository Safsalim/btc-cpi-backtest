from __future__ import annotations

from typer.testing import CliRunner

from btc_cpi_backtest.cli import app
from btc_cpi_backtest.market_data import FetchSummary


runner = CliRunner()


def test_cli_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fetch-data" in result.stdout
    assert "cpi-summary" in result.stdout
    assert "analyze" in result.stdout


def test_fetch_data_command_reports_summary(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return FetchSummary(total_releases=2, downloaded=1, reused=1, total_candles=120)

    monkeypatch.setattr("btc_cpi_backtest.cli.download_cpi_market_data", fake_download)

    result = runner.invoke(
        app,
        [
            "fetch-data",
            "--cache-dir",
            str(tmp_path),
            "--start",
            "2023-01-01T00:00:00+00:00",
            "--end",
            "2023-02-01T00:00:00+00:00",
        ],
    )

    assert result.exit_code == 0
    assert "Processed 2 CPI releases" in result.stdout
    assert "Downloaded: 1" in result.stdout
    assert "Reused cache: 1" in result.stdout
    assert "Cache directory" in result.stdout

    assert captured["cache_dir"] == tmp_path
    assert captured["exchange_name"] == "binance"
    assert captured["timeframe"] == "1m"
    assert captured["start"].isoformat() == "2023-01-01T00:00:00+00:00"
    assert captured["end"].isoformat() == "2023-02-01T00:00:00+00:00"


def test_cpi_summary_sample_dataset() -> None:
    result = runner.invoke(app, ["cpi-summary"])

    assert result.exit_code == 0
    assert "Loaded 3 CPI releases from bundled sample dataset." in result.stdout
    assert "2023-10-12T12:30:00+00:00" in result.stdout
    assert "2023-12-12T13:30:00+00:00" in result.stdout


def test_analyze_placeholder_runs() -> None:
    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 0
    assert "Analysis not yet implemented" in result.stdout
