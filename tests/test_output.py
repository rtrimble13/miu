"""CSV/JSON writers and summary table."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from miu.index import IndexResult
from miu.output import print_summary, write_csv, write_json


def _fake_result() -> IndexResult:
    series = pd.DataFrame(
        [
            {"date": date(2020, 1, 2), "index_level": 1000.0, "daily_return": 0.0, "n_constituents": 2},
            {"date": date(2020, 1, 3), "index_level": 1010.0, "daily_return": 0.01, "n_constituents": 2},
        ]
    )
    constituents = pd.DataFrame(
        [
            {"date": date(2020, 1, 2), "entity_id": "A", "ticker": "A", "weight": 0.5, "is_rebalance_date": True},
            {"date": date(2020, 1, 2), "entity_id": "B", "ticker": "B", "weight": 0.5, "is_rebalance_date": True},
        ]
    )
    summary = {
        "total_return": 0.01,
        "annualized_return": 0.05,
        "annualized_vol": 0.12,
        "max_drawdown": 0.0,
        "avg_constituents": 2.0,
        "rebalances": 1,
        "delisted_in_sample": 0,
    }
    return IndexResult(series=series, constituents=constituents, summary=summary)


def test_write_csv_creates_pair(tmp_path: Path) -> None:
    result = _fake_result()
    series_path, const_path = write_csv(result, tmp_path / "idx.csv")
    assert series_path == tmp_path / "idx.csv"
    assert const_path == tmp_path / "idx_constituents.csv"
    assert series_path.exists() and const_path.exists()
    df = pd.read_csv(series_path)
    assert list(df.columns) == ["date", "index_level", "daily_return", "n_constituents"]


def test_write_json_matches_spec_shape(tmp_path: Path) -> None:
    result = _fake_result()
    meta = {
        "sector": "Health Care",
        "industry": None,
        "weighting": "market-cap",
        "start": "2020-01-02",
        "end": "2020-01-03",
        "rebalance": "quarterly",
        "base_value": 1000.0,
    }
    path = write_json(result, tmp_path / "idx.json", meta)
    body = json.loads(path.read_text())
    assert set(body.keys()) == {"meta", "series", "constituents"}
    assert body["meta"]["weighting"] == "market-cap"
    assert body["meta"]["miu_version"]
    assert body["meta"]["created_at"]
    assert body["series"][0]["level"] == 1000.0
    assert body["constituents"][0]["entity_id"] == "A"


def test_print_summary_renders(capsys) -> None:
    result = _fake_result()
    print_summary(result)
    out = capsys.readouterr().out
    assert "total return" in out
    assert "annualized" in out
