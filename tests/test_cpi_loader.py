from __future__ import annotations

from datetime import datetime, timezone

import pytest

from btc_cpi_backtest.cpi_loader import CPIColumnConfig, load_cpi_from_csv


def test_load_cpi_from_csv_parses_timezone_and_sorts(tmp_path) -> None:
    csv_path = tmp_path / "cpi.csv"
    csv_path.write_text(
        "release_time,actual,expected,previous\n"
        "2024-01-11 08:30:00,3.1,3.2,3.5\n"
        "2023-12-12 08:30:00,3.0,3.1,3.2\n"
    )

    releases = load_cpi_from_csv(csv_path, input_timezone="US/Eastern")

    assert len(releases) == 2
    assert releases[0].release_datetime == datetime(2023, 12, 12, 13, 30, tzinfo=timezone.utc)
    assert releases[1].release_datetime == datetime(2024, 1, 11, 13, 30, tzinfo=timezone.utc)
    assert releases[0].actual == pytest.approx(3.0)
    assert releases[0].expected == pytest.approx(3.1)
    assert releases[0].previous == pytest.approx(3.2)
    assert releases[0].surprise is None


def test_load_cpi_from_csv_combines_date_and_time(tmp_path) -> None:
    csv_path = tmp_path / "cpi_schedule.csv"
    csv_path.write_text(
        "Release Date,Time,Actual,Forecast,Previous\n"
        '"Oct 12, 2023 (Sep)",08:30,3.7%,3.6%,3.7%\n'
        '"Nov 14, 2023 (Oct)",09:30,3.2%,3.3%,3.7%\n'
    )

    config = CPIColumnConfig(
        timestamp=None,
        date="Release Date",
        time="Time",
        actual="Actual",
        expected="Forecast",
        previous="Previous",
    )

    releases = load_cpi_from_csv(csv_path, columns=config, input_timezone="US/Eastern")

    assert len(releases) == 2
    assert releases[0].release_datetime == datetime(2023, 10, 12, 12, 30, tzinfo=timezone.utc)
    assert releases[1].release_datetime == datetime(2023, 11, 14, 14, 30, tzinfo=timezone.utc)
    assert releases[0].actual == pytest.approx(3.7)
    assert releases[1].expected == pytest.approx(3.3)
    assert releases[1].previous == pytest.approx(3.7)


def test_load_cpi_from_csv_missing_required_column(tmp_path) -> None:
    csv_path = tmp_path / "cpi_missing.csv"
    csv_path.write_text(
        "release_time,actual\n"
        "2024-01-11 08:30:00,3.1\n"
    )

    with pytest.raises(ValueError, match="Missing required column"):
        load_cpi_from_csv(csv_path)


def test_load_cpi_from_csv_rejects_invalid_timestamps(tmp_path) -> None:
    csv_path = tmp_path / "cpi_invalid.csv"
    csv_path.write_text(
        "release_time,actual,expected\n"
        "not-a-datetime,3.1,3.2\n"
    )

    with pytest.raises(ValueError, match="Invalid or missing release timestamps"):
        load_cpi_from_csv(csv_path)
