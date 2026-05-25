"""Tests for the ETF data layer (etf.py)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from miu.config import MiuConfigError, Settings
from miu.etf import (
    IndexReturns,
    align_returns,
    discover_etf_candidates,
    load_etf_returns_panel,
    load_index_from_file,
)
from miu.fmp.client import FmpClient
from miu.index import IndexResult
from miu.output import write_csv, write_json


def _make_index_result() -> IndexResult:
    days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
    series = pd.DataFrame(
        {
            "date": days,
            "index_level": [1000.0, 1010.0, 1005.0, 1020.0],
            "daily_return": [0.0, 0.01, -0.00495, 0.01492],
            "n_constituents": [3, 3, 3, 3],
        }
    )
    constituents = pd.DataFrame(
        columns=["date", "entity_id", "ticker", "weight", "is_rebalance_date"]
    )
    summary: dict[str, float | int] = {
        "total_return": 0.02,
        "annualized_return": 0.0,
        "annualized_vol": 0.0,
        "max_drawdown": 0.0,
        "avg_constituents": 3.0,
        "rebalances": 1,
        "delisted_in_sample": 0,
    }
    return IndexResult(series=series, constituents=constituents, summary=summary)


def test_load_index_from_csv_round_trip(tmp_path: Path) -> None:
    result = _make_index_result()
    series_path, _ = write_csv(result, tmp_path / "idx.csv")
    ir = load_index_from_file(series_path)
    assert isinstance(ir, IndexReturns)
    # First row's daily_return is 0.0 — kept as a real datum, not dropped.
    assert len(ir.returns) == 4
    assert pytest.approx(ir.returns.iloc[1]) == 0.01


def test_load_index_from_json_round_trip(tmp_path: Path) -> None:
    result = _make_index_result()
    json_path = write_json(result, tmp_path / "idx.json", {"sector": "Test"})
    ir = load_index_from_file(json_path)
    assert ir.meta.get("sector") == "Test"
    assert len(ir.returns) == 4


def test_load_index_recomputes_from_level_only(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "index_level": [1000.0, 1010.0, 1020.1],
        }
    )
    p = tmp_path / "level_only.csv"
    df.to_csv(p, index=False)
    ir = load_index_from_file(p)
    # First row drops to NaN under pct_change → dropped; expect 2 rows.
    assert len(ir.returns) == 2
    assert pytest.approx(ir.returns.iloc[0], rel=1e-6) == 0.01


def test_load_index_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MiuConfigError):
        load_index_from_file(tmp_path / "nope.csv")


def test_load_index_rejects_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("date,index_level\n2020-01-02,1000\n")
    with pytest.raises(MiuConfigError):
        load_index_from_file(p)


def test_index_returns_slice() -> None:
    s = pd.Series(
        [0.01, -0.005, 0.02, 0.0, -0.01],
        index=pd.Index(
            [
                date(2020, 1, 2), date(2020, 1, 3),
                date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8),
            ],
            name="date",
        ),
    )
    ir = IndexReturns(returns=s)
    sliced = ir.slice(date(2020, 1, 3), date(2020, 1, 7))
    assert list(sliced.returns.index) == [date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]


def test_align_returns_inner_join() -> None:
    days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    idx = pd.Series([0.01, 0.02, -0.01], index=pd.Index(days, name="date"))
    panel = pd.DataFrame(
        {
            "A": [0.005, 0.018, -0.012],
            "B": [float("nan"), 0.022, -0.008],
        },
        index=pd.Index(days, name="date"),
    )
    r_idx, r_panel = align_returns(idx, panel)
    assert len(r_idx) == 2  # row 0 dropped (B is NaN)
    assert list(r_idx.index) == [date(2020, 1, 3), date(2020, 1, 6)]


@pytest.mark.asyncio
async def test_discover_etf_candidates_orders_by_market_cap(tmp_path: Path) -> None:
    settings = Settings(
        api_key="test", cache_dir=tmp_path / "cache", base_url="https://example.test"
    )
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        mock.get("/company-screener").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"symbol": "BIG", "marketCap": 5e10, "exchangeShortName": "NYSE"},
                    {"symbol": "MID", "marketCap": 2e10, "exchangeShortName": "NYSE"},
                    {"symbol": "TINY", "marketCap": 1e7, "exchangeShortName": "NYSE"},
                ],
            )
        )
        async with FmpClient(settings) as client:
            picks = await discover_etf_candidates(
                client, sector="Healthcare", industry=None, min_aum=1e9, max_candidates=10
            )
    # TINY drops out (below min_aum); BIG before MID.
    assert picks == ["BIG", "MID"]


@pytest.mark.asyncio
async def test_load_etf_returns_panel_builds_wide_frame(tmp_path: Path) -> None:
    settings = Settings(
        api_key="test", cache_dir=tmp_path / "cache", base_url="https://example.test"
    )
    days = pd.date_range("2020-01-02", periods=5, freq="B")

    def series_for(base: float, drift: float) -> list[dict]:
        return [
            {"date": d.date().isoformat(), "adjClose": base * (1 + drift) ** i}
            for i, d in enumerate(days)
        ]

    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "XLE"}
        ).mock(return_value=httpx.Response(200, json=series_for(50.0, 0.001)))
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "VDE"}
        ).mock(return_value=httpx.Response(200, json=series_for(80.0, 0.0008)))

        async with FmpClient(settings) as client:
            panel = await load_etf_returns_panel(
                client, ["XLE", "VDE"], start=date(2020, 1, 1), end=date(2020, 1, 31)
            )
    assert list(panel.columns) == ["VDE", "XLE"] or list(panel.columns) == ["XLE", "VDE"]
    # 5 prices → 4 returns
    assert len(panel) == 4
    assert pytest.approx(panel["XLE"].iloc[0], rel=1e-6) == 0.001
