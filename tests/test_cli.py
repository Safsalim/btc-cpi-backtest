from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from btc_cpi_backtest.cli import app


runner = CliRunner()


def test_cli_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fetch-data" in result.stdout
    assert "cpi-summary" in result.stdout
    assert "analyze" in result.stdout


def test_fetch_data_placeholder_runs() -> None:
    result = runner.invoke(app, ["fetch-data"])

    assert result.exit_code == 0
    assert "Download not yet implemented" in result.stdout


def test_cpi_summary_sample_dataset() -> None:
    result = runner.invoke(app, ["cpi-summary"])

    assert result.exit_code == 0
    assert "Loaded 3 CPI releases from bundled sample dataset." in result.stdout
    assert "2023-10-12T12:30:00+00:00" in result.stdout
    assert "2023-12-12T13:30:00+00:00" in result.stdout


def test_analyze_generates_report(tmp_path) -> None:
    output_path = tmp_path / "fakeout_report.csv"
    result = runner.invoke(
        app,
        [
            "analyze",
            "--output",
            str(output_path),
            "--cpi-data",
            "sample",
            "--btc-data",
            "sample",
        ],
    )

    assert result.exit_code == 0
    assert "Analysis complete for" in result.stdout
    assert "Fake-out rates" in result.stdout
    assert "Saved detailed report to" in result.stdout
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert not df.empty
    assert "fake_5m_1h" in df.columns
