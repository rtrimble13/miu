"""Eligibility rules under point-in-time mocking."""

from __future__ import annotations

from datetime import date, timedelta

from miu.eligibility import EligibilityInputs, is_eligible, select_eligible
from miu.universe import Constituent, TickerSpan


def make_c(eid: str, **kwargs) -> Constituent:
    base = {
        "entity_id": eid,
        "ticker_history": [TickerSpan(ticker=eid, start=None, end=None)],
        "ipo_date": date(2010, 1, 1),
        "sector": "Health Care",
    }
    base.update(kwargs)
    return Constituent(**base)


def inputs(price_date: date | None, mcap: float | None = 1e9) -> EligibilityInputs:
    return EligibilityInputs(last_price_date=price_date, market_cap_t_minus_1=mcap)


def test_seasoning_rule_blocks_recent_ipos() -> None:
    c = make_c("A", ipo_date=date(2020, 1, 1))
    t = date(2020, 3, 1)  # ~60 days post-IPO
    assert not is_eligible(c, t, inputs(date(2020, 2, 28)))


def test_seasoning_rule_allows_after_180_days() -> None:
    c = make_c("A", ipo_date=date(2020, 1, 1))
    t = date(2020, 7, 1)  # ~180 days
    assert is_eligible(c, t, inputs(date(2020, 6, 30)))


def test_delisting_excludes_after_cutoff() -> None:
    c = make_c("A", delisting_date=date(2021, 6, 15))
    assert not is_eligible(c, date(2021, 6, 16), inputs(date(2021, 6, 14)))
    assert is_eligible(c, date(2021, 6, 14), inputs(date(2021, 6, 13)))


def test_sector_filter() -> None:
    c = make_c("A", sector="Energy")
    t = date(2021, 6, 1)
    assert not is_eligible(c, t, inputs(t - timedelta(days=1)), sector="Health Care")
    assert is_eligible(c, t, inputs(t - timedelta(days=1)), sector="energy")  # case-insensitive


def test_min_market_cap() -> None:
    c = make_c("A")
    t = date(2021, 6, 1)
    assert not is_eligible(c, t, inputs(t, mcap=10e6), min_market_cap=100e6)
    assert is_eligible(c, t, inputs(t, mcap=200e6), min_market_cap=100e6)


def test_missing_price_blocks() -> None:
    c = make_c("A")
    t = date(2021, 6, 1)
    assert not is_eligible(c, t, inputs(None))


def test_stale_price_blocks() -> None:
    c = make_c("A")
    t = date(2021, 6, 15)
    # 8 calendar days old → outside the 5-trading-day window
    assert not is_eligible(c, t, inputs(date(2021, 6, 7)))


def test_max_constituents_keeps_top_n_by_mcap() -> None:
    cs = [make_c(f"E{i}") for i in range(5)]
    t = date(2021, 6, 1)
    inp = {
        "E0": inputs(t - timedelta(days=1), mcap=100),
        "E1": inputs(t - timedelta(days=1), mcap=500),
        "E2": inputs(t - timedelta(days=1), mcap=200),
        "E3": inputs(t - timedelta(days=1), mcap=400),
        "E4": inputs(t - timedelta(days=1), mcap=300),
    }
    kept = select_eligible(cs, t, inp, max_constituents=3)
    assert {c.entity_id for c in kept} == {"E1", "E3", "E4"}


def test_point_in_time_invariance(monkeypatch) -> None:
    """Mocking 'now' to a wildly different date must not change the result."""
    import datetime as dt

    c = make_c("A", ipo_date=date(2018, 1, 1))
    t = date(2020, 6, 1)
    truth = is_eligible(c, t, inputs(t - timedelta(days=1)))

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401, ARG003
            return dt.datetime(2099, 12, 31)

    monkeypatch.setattr(dt, "datetime", FakeDT)
    assert is_eligible(c, t, inputs(t - timedelta(days=1))) == truth
