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


def test_rebalance_day_return_uses_prior_basket() -> None:
    """On a rebalance day the day's return must reflect the OLD basket held
    entering the day, not the freshly-selected basket. Construct a scenario
    where one name flips into eligibility on a rebalance day and jumps in
    price the same day — only the buggy "use new basket" path would capture
    that jump in the rebalance day's return."""
    cal = _calendar(70)
    constituents = [_const("A"), _const("B")]
    # A: flat all the way through.
    # B: flat at 100 through Jan, jumps to 200 on the Feb rebalance day.
    feb_idx = next(i for i, d in enumerate(cal) if d.month == 2)
    prices = {
        "A": {d: 100.0 for d in cal},
        "B": {
            **{d: 100.0 for d in cal[:feb_idx]},
            **{d: 200.0 for d in cal[feb_idx:]},
        },
    }
    # B fails min-mcap on the bootstrap rebalance (Jan) but passes by the
    # Feb rebalance — mcap(t-1) checked at cal[feb_idx-1].
    mcaps = {
        "A": {d: 5e9 for d in cal},
        "B": {
            **{d: 50e6 for d in cal[: feb_idx - 1]},
            **{d: 5e9 for d in cal[feb_idx - 1 :]},
        },
    }
    config = EngineConfig(
        weighting="equal",
        start=cal[0],
        end=cal[-1],
        rebalance="monthly",
        min_market_cap=100e6,
    )
    result = IndexEngine(constituents, prices, mcaps, config).run()
    feb_day = cal[feb_idx]
    feb_return = float(
        result.series.loc[result.series["date"] == feb_day, "daily_return"].iloc[0]
    )
    # OLD basket entering Feb rebal day = {A only}. A is flat → ~0% day return.
    # If the engine wrongly used the NEW basket {A, B} equal-weight, B's
    # 100% jump on this day would push the return toward +50%.
    assert abs(feb_return) < 0.01, f"feb return {feb_return} suggests new basket leaked into the return"
    # Sanity: the Feb rebalance row should expose the NEW basket (A + B).
    feb_rows = result.constituents[result.constituents["date"] == feb_day]
    assert set(feb_rows["entity_id"]) == {"A", "B"}
    assert all(feb_rows["is_rebalance_date"])


def test_stock_for_stock_mna_redistributes_via_acquirer_price() -> None:
    """Stock-for-stock M&A: terminal value = ratio × acquirer price on the
    delisting day, not the target's last-print fallback."""
    cal = _calendar(30)
    delisting = cal[15]
    constituents = [_const("ACQ"), _const("TGT", delisting=delisting)]
    prices = {
        # Acquirer trades flat at 200 until delisting day, then jumps to 250
        # — the engine must mark TGT to ratio × 250, not at TGT's last-print.
        "ACQ": {**{d: 200.0 for d in cal[:15]}, **{d: 250.0 for d in cal[15:]}},
        # Target trades flat at 100 until delisting (no quotes after).
        "TGT": {d: 100.0 for d in cal[:15]},
    }
    mcaps = {eid: {d: 1e9 for d in cal} for eid in ("ACQ", "TGT")}
    mcaps["TGT"] = {d: 1e9 for d in cal[:15]}
    # 1 TGT share converts to 0.5 ACQ shares.
    mna = {"TGT": MnaResolution(acquirer_id="ACQ", ratio=0.5)}
    config = EngineConfig(
        weighting="equal", start=cal[0], end=cal[-1], rebalance="none", base_value=1000.0
    )
    result = IndexEngine(
        constituents, prices, mcaps, config, mna_resolutions=mna
    ).run()
    # Day return on the delisting day:
    #   TGT: terminal = 0.5 × 250 = 125 vs prev 100 = +25%
    #   ACQ: 250/200 - 1 = +25%
    # Equal-weighted: +25% combined.
    day_return = float(
        result.series.loc[result.series["date"] == delisting, "daily_return"].iloc[0]
    )
    assert math.isclose(day_return, 0.25, rel_tol=1e-6, abs_tol=1e-6)


def test_stock_for_stock_mna_tracks_acquirer_post_delisting() -> None:
    """After delisting, the converted target position must continue to track
    the acquirer's return until the next rebalance — not return 0."""
    cal = _calendar(30)
    delisting = cal[10]
    constituents = [_const("ACQ"), _const("TGT", delisting=delisting)]
    prices = {
        # Acquirer: flat at 200 before delisting, then rises 1% per day after.
        "ACQ": {
            **{d: 200.0 for d in cal[:10]},
            **{d: 200.0 * (1.01 ** (i + 1)) for i, d in enumerate(cal[10:])},
        },
        # Target: flat at 100 until delisting (no quotes after).
        "TGT": {d: 100.0 for d in cal[:10]},
    }
    mcaps = {eid: {d: 1e9 for d in cal} for eid in ("ACQ", "TGT")}
    mcaps["TGT"] = {d: 1e9 for d in cal[:10]}
    # 1 TGT share converts to 0.5 ACQ shares.
    mna = {"TGT": MnaResolution(acquirer_id="ACQ", ratio=0.5)}
    config = EngineConfig(
        weighting="equal", start=cal[0], end=cal[-1], rebalance="none", base_value=1000.0
    )
    result = IndexEngine(
        constituents, prices, mcaps, config, mna_resolutions=mna
    ).run()
    post = result.series[result.series["date"] > delisting]
    # Both the direct ACQ position and the converted TGT position track ACQ,
    # so the portfolio return each day must be ~+1% (the acquirer's daily gain).
    for r in post["daily_return"]:
        assert math.isclose(float(r), 0.01, rel_tol=1e-4), (
            f"post-delisting return {r} should track acquirer's 1% gain"
        )



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
