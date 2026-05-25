"""Three weighting schemes: hand-computed expectations."""

from __future__ import annotations

import math
from datetime import date

from miu.universe import Constituent, TickerSpan
from miu.weights import compute_weights, equal_weights, market_cap_weights, price_weights


def make_c(eid: str) -> Constituent:
    return Constituent(entity_id=eid, ticker_history=[TickerSpan(ticker=eid, start=None, end=None)])


def test_equal_weights_sum_to_one() -> None:
    cs = [make_c("A"), make_c("B"), make_c("C")]
    w = equal_weights(cs, date(2020, 1, 1))
    assert set(w) == {"A", "B", "C"}
    assert math.isclose(sum(w.values()), 1.0)
    assert all(math.isclose(v, 1 / 3) for v in w.values())


def test_market_cap_weights_proportional() -> None:
    cs = [make_c("A"), make_c("B"), make_c("C")]
    mcap = {"A": 100.0, "B": 300.0, "C": 600.0}
    w = market_cap_weights(cs, mcap, date(2020, 1, 1))
    assert math.isclose(w["A"], 0.1)
    assert math.isclose(w["B"], 0.3)
    assert math.isclose(w["C"], 0.6)
    assert math.isclose(sum(w.values()), 1.0)


def test_market_cap_weights_ignore_missing() -> None:
    cs = [make_c("A"), make_c("B")]
    mcap = {"A": 100.0}  # B missing
    w = market_cap_weights(cs, mcap, date(2020, 1, 1))
    assert w == {"A": 1.0}


def test_price_weights_dow_style() -> None:
    cs = [make_c("A"), make_c("B")]
    prices = {"A": 50.0, "B": 150.0}
    w = price_weights(cs, prices, date(2020, 1, 1))
    assert math.isclose(w["A"], 0.25)
    assert math.isclose(w["B"], 0.75)


def test_compute_weights_dispatcher() -> None:
    cs = [make_c("A"), make_c("B")]
    eq = compute_weights("equal", cs, prices={}, market_caps={}, as_of=date(2020, 1, 1))
    mc = compute_weights(
        "market-cap", cs, prices={}, market_caps={"A": 1.0, "B": 1.0}, as_of=date(2020, 1, 1)
    )
    pw = compute_weights(
        "price", cs, prices={"A": 1.0, "B": 3.0}, market_caps={}, as_of=date(2020, 1, 1)
    )
    assert math.isclose(sum(eq.values()), 1.0)
    assert math.isclose(sum(mc.values()), 1.0)
    assert math.isclose(sum(pw.values()), 1.0)


def test_compute_weights_unknown_scheme_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_weights("rms", [], prices={}, market_caps={}, as_of=date(2020, 1, 1))


def test_empty_constituents_returns_empty() -> None:
    assert equal_weights([], date(2020, 1, 1)) == {}
    assert market_cap_weights([], {}, date(2020, 1, 1)) == {}
    assert price_weights([], {}, date(2020, 1, 1)) == {}
