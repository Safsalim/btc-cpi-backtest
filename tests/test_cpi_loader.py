from __future__ import annotations

from datetime import datetime, timezone

import pytest

from btc_cpi_backtest.cpi_loader import load_cpi_from_csv


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
