from __future__ import annotations

from typer.testing import CliRunner

from btc_cpi_backtest.cli import app


runner = CliRunner()


def test_cli_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "fetch-data" in result.stdout
    assert "analyze" in result.stdout


def test_fetch_data_placeholder_runs() -> None:
    result = runner.invoke(app, ["fetch-data"])

    assert result.exit_code == 0
    assert "Download not yet implemented" in result.stdout


def test_analyze_placeholder_runs() -> None:
    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 0
    assert "Analysis not yet implemented" in result.stdout
