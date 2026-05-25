"""Index engine: cash-buyout, equal-weight smoke, drift between rebalances."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from miu.index import EngineConfig, IndexEngine, MnaResolution
from miu.universe import Constituent, TickerSpan


def _const(eid: str, ipo: date = date(2010, 1, 1), delisting: date | None = None) -> Constituent:
    return Constituent(
        entity_id=eid,
        ticker_history=[TickerSpan(ticker=eid, start=None, end=delisting)],
        ipo_date=ipo,
        delisting_date=delisting,
        sector="Health Care",
    )


def _calendar(n: int = 60) -> list[date]:
    return [d.date() for d in pd.date_range("2020-01-02", periods=n, freq="B")]


def test_equal_weight_smoke_run() -> None:
    cal = _calendar(40)
    constituents = [_const("A"), _const("B")]
    prices = {
        "A": {d: 100.0 * (1.001**i) for i, d in enumerate(cal)},
        "B": {d: 50.0 * (1.0005**i) for i, d in enumerate(cal)},
    }
    market_caps = {eid: {d: 1e9 for d in cal} for eid in ("A", "B")}
    config = EngineConfig(
        weighting="equal",
        start=cal[0],
        end=cal[-1],
        rebalance="quarterly",
        base_value=1000.0,
        sector="Health Care",
    )
    result = IndexEngine(constituents, prices, market_caps, config).run()
    assert len(result.series) == len(cal)
    assert result.series["index_level"].iloc[0] == 1000.0
    assert result.series["index_level"].iloc[-1] > 1000.0
    # Returns should compound consistently.
    final = 1000.0 * (1.0 + result.series["daily_return"].iloc[1:]).prod()
    assert math.isclose(final, result.series["index_level"].iloc[-1], rel_tol=1e-9)


def test_market_cap_weight_dominates_large_names() -> None:
    cal = _calendar(20)
    constituents = [_const("BIG"), _const("SMALL")]
    # BIG has a -1% daily move, SMALL has +1%. Cap-weighted should track BIG.
    prices = {
        "BIG": {d: 100.0 * (0.99**i) for i, d in enumerate(cal)},
        "SMALL": {d: 100.0 * (1.01**i) for i, d in enumerate(cal)},
    }
    market_caps = {
        "BIG": {d: 1_000_000_000.0 for d in cal},
        "SMALL": {d: 1_000_000.0 for d in cal},
    }
    config = EngineConfig(
        weighting="market-cap", start=cal[0], end=cal[-1], rebalance="annual"
    )
    result = IndexEngine(constituents, prices, market_caps, config).run()
    # With 99.9% weight on BIG, level should fall.
    assert result.series["index_level"].iloc[-1] < 1000.0


def test_cash_buyout_terminal_value() -> None:
    cal = _calendar(30)
    delisting = cal[15]
    constituents = [_const("STAY"), _const("BUYOUT", delisting=delisting)]
    prices = {
        "STAY": {d: 100.0 for d in cal},
        # BUYOUT trades at 50 until delisting, then no quotes.
        "BUYOUT": {d: 50.0 for d in cal[:15]},
    }
    market_caps = {eid: {d: 1e9 for d in cal[:15]} for eid in ("STAY", "BUYOUT")}
    market_caps["STAY"] = {d: 1e9 for d in cal}
    mna = {"BUYOUT": MnaResolution(cash_value=75.0)}  # 50 → 75 = +50% terminal
    config = EngineConfig(
        weighting="equal", start=cal[0], end=cal[-1], rebalance="none", base_value=1000.0
    )
    result = IndexEngine(
        constituents, prices, market_caps, config, mna_resolutions=mna
    ).run()
    # On the delisting day, BUYOUT half contributes +50%, STAY half ~0% → +25% day.
    day_return = float(result.series.loc[result.series["date"] == delisting, "daily_return"].iloc[0])
    assert day_return > 0.20  # robust to small drift
    # After delisting, only STAY contributes, so subsequent returns should be ~0.
    post = result.series[result.series["date"] > delisting]["daily_return"]
    assert all(abs(r) < 1e-9 for r in post)


def test_quarterly_rebalance_count() -> None:
    cal = _calendar(220)  # ~1 trading year
    constituents = [_const("A"), _const("B")]
    prices = {"A": {d: 100.0 for d in cal}, "B": {d: 100.0 for d in cal}}
    mcaps = {"A": {d: 1e9 for d in cal}, "B": {d: 1e9 for d in cal}}
    config = EngineConfig(
        weighting="equal", start=cal[0], end=cal[-1], rebalance="quarterly"
    )
    result = IndexEngine(constituents, prices, mcaps, config).run()
    assert result.summary["rebalances"] == 4


def test_drift_between_rebalances_keeps_membership() -> None:
    cal = _calendar(3)
    constituents = [_const("A"), _const("B")]
    prices = {
        "A": {cal[0]: 100.0, cal[1]: 200.0, cal[2]: 300.0},
        "B": {cal[0]: 100.0, cal[1]: 100.0, cal[2]: 50.0},
    }
    mcaps = {"A": {d: 1e9 for d in cal}, "B": {d: 1e9 for d in cal}}
    config = EngineConfig(weighting="equal", start=cal[0], end=cal[-1], rebalance="none")
    result = IndexEngine(constituents, prices, mcaps, config).run()
    rebal_rows = result.constituents[result.constituents["is_rebalance_date"]]
    non_rebal_rows = result.constituents[~result.constituents["is_rebalance_date"]]
    assert rebal_rows["date"].nunique() == 1
    assert rebal_rows["date"].iloc[0] == cal[0]
    assert non_rebal_rows["date"].nunique() == 2
    assert set(result.constituents["date"]) == set(cal)
    # Day 3 return must use day-2 drifted weights (A outperformed on day 2).
    day3_return = float(result.series.loc[result.series["date"] == cal[2], "daily_return"].iloc[0])
    assert day3_return > 0.15


def test_min_market_cap_excludes_small() -> None:
    cal = _calendar(20)
    constituents = [_const("BIG"), _const("SMALL")]
    prices = {eid: {d: 100.0 for d in cal} for eid in ("BIG", "SMALL")}
    mcaps = {
        "BIG": {d: 5e9 for d in cal},
        "SMALL": {d: 50e6 for d in cal},
    }
    config = EngineConfig(
        weighting="equal",
        start=cal[0],
        end=cal[-1],
        rebalance="annual",
        min_market_cap=100e6,
    )
    result = IndexEngine(constituents, prices, mcaps, config).run()
    # Only BIG should be included; SMALL filtered out by min-mcap.
    rebal = result.constituents[result.constituents["is_rebalance_date"]]
    assert set(rebal["entity_id"]) == {"BIG"}
