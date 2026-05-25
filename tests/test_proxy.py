"""Tests for the ETF proxy ranking module (proxy.py)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from miu.fmp.models import EtfProfile
from miu.proxy import compute_metric, rank_proxies, recommend


def _make_series(values: list[float], start: str = "2020-01-02") -> pd.Series:
    days = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=pd.Index([d.date() for d in days], name="date"))


def test_compute_metric_perfect_tracker() -> None:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, size=200).tolist()
    r_idx = _make_series(rets)
    r_etf = r_idx.copy()
    m = compute_metric(r_idx, r_etf, ticker="PERF", profile=None)
    assert m.te == pytest.approx(0.0, abs=1e-12)
    assert m.r2 == pytest.approx(1.0, abs=1e-12)
    assert m.beta == pytest.approx(1.0, abs=1e-9)


def test_compute_metric_anti_correlated() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.01, size=200).tolist()
    r_idx = _make_series(rets)
    r_etf = _make_series([-x for x in rets])
    m = compute_metric(r_idx, r_etf, ticker="ANTI", profile=None)
    assert m.corr == pytest.approx(-1.0, abs=1e-9)
    assert m.beta == pytest.approx(-1.0, abs=1e-9)
    assert m.te > 0.1  # huge tracking error


def test_compute_metric_flat_index_handles_zero_variance() -> None:
    r_idx = _make_series([0.0] * 100)
    r_etf = _make_series([0.0005] * 100)
    m = compute_metric(r_idx, r_etf, ticker="ZERO", profile=None)
    # Variance of index is zero → beta and r2 are NaN, but TE is still computable.
    assert math.isnan(m.beta)
    assert math.isnan(m.r2)
    assert not math.isnan(m.te)


def test_compute_metric_attaches_profile_fields() -> None:
    r = _make_series([0.001] * 100)
    profile = EtfProfile(symbol="XLE", expense_ratio=0.001, aum=2.5e10, name="SPDR Energy")
    m = compute_metric(r, r, ticker="XLE", profile=profile)
    assert m.expense_ratio == 0.001
    assert m.aum == 2.5e10
    assert m.name == "SPDR Energy"


def test_rank_proxies_orders_by_te() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(0.0005, 0.012, size=120).tolist()
    r_idx = _make_series(base)
    # `tight` follows the index closely; `loose` adds large noise.
    tight = _make_series([x + rng.normal(0, 1e-5) for x in base])
    loose = _make_series([x + rng.normal(0, 0.01) for x in base])
    panel = pd.DataFrame({"TIGHT": tight, "LOOSE": loose})
    metrics = rank_proxies(r_idx, panel, profiles={})
    assert [m.ticker for m in metrics] == ["TIGHT", "LOOSE"]
    assert metrics[0].te < metrics[1].te


def test_rank_proxies_demotes_insufficient_overlap() -> None:
    r_idx = _make_series([0.001] * 200)
    # Only 10 observations of overlap.
    short_days = pd.bdate_range("2020-01-02", periods=10)
    short = pd.Series(
        [0.001] * 10, index=pd.Index([d.date() for d in short_days], name="date")
    )
    panel = pd.DataFrame({"SHORT": short})
    metrics = rank_proxies(r_idx, panel, profiles={}, min_overlap=60)
    assert metrics[0].ticker == "SHORT"
    assert math.isnan(metrics[0].te)


def test_recommend_picks_lowest_te() -> None:
    rng = np.random.default_rng(99)
    base = rng.normal(0.0, 0.01, size=100).tolist()
    r_idx = _make_series(base)
    panel = pd.DataFrame(
        {
            "A": _make_series([x * 1.5 for x in base]),  # higher beta → higher TE
            "B": _make_series(base),  # perfect
        }
    )
    winner, table = recommend(r_idx, panel, profiles={})
    assert winner is not None
    assert winner.ticker == "B"
    assert len(table) == 2


def test_recommend_returns_none_when_all_demoted() -> None:
    r_idx = _make_series([0.0] * 200)
    short_idx = pd.Index([date(2020, 1, 2 + i) for i in range(5)], name="date")
    short = pd.Series([0.0] * 5, index=short_idx)
    panel = pd.DataFrame({"X": short})
    winner, table = recommend(r_idx, panel, profiles={}, min_overlap=60)
    assert winner is None
    assert len(table) == 1
