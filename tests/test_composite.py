"""Tests for the constrained-OLS composite (composite.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from miu.composite import fit_composite, top_k_by_te
from miu.config import MiuOptimizerError
from miu.proxy import ProxyMetric


def _make_series(values: list[float], start: str = "2020-01-02") -> pd.Series:
    days = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=pd.Index([d.date() for d in days], name="date"))


def test_top_k_by_te_picks_smallest() -> None:
    def _m(t: str, te: float, r2: float = 0.8) -> ProxyMetric:
        return ProxyMetric(
            t, te=te, corr=0.9, beta=1.0, r2=r2,
            expense_ratio=None, aum=None, n_obs=200,
        )
    metrics = [_m("A", 0.01), _m("B", 0.005, r2=0.9), _m("C", float("nan")), _m("D", 0.02)]
    assert top_k_by_te(metrics, 2) == ["B", "A"]


def test_fit_composite_recovers_known_two_asset_mix() -> None:
    """Synthesize an index as a known 60/40 mix of two ETFs and recover it."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0005, 0.01, size=300).tolist()
    b = rng.normal(0.0003, 0.012, size=300).tolist()
    idx_vals = [0.6 * x + 0.4 * y for x, y in zip(a, b, strict=True)]
    r_idx = _make_series(idx_vals)
    panel = pd.DataFrame({"A": _make_series(a), "B": _make_series(b)})
    result = fit_composite(r_idx, panel)
    assert result.converged is True
    assert result.weights["A"] == pytest.approx(0.6, abs=1e-4)
    assert result.weights["B"] == pytest.approx(0.4, abs=1e-4)
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert result.te < 1e-6


def test_fit_composite_respects_nonneg_constraint() -> None:
    """If the true mix needs a negative weight, the constrained solution clamps to 0."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.001, 0.01, size=200).tolist()
    b = rng.normal(0.001, 0.01, size=200).tolist()
    # Target with negative weight on B
    idx_vals = [1.4 * x - 0.4 * y for x, y in zip(a, b, strict=True)]
    r_idx = _make_series(idx_vals)
    panel = pd.DataFrame({"A": _make_series(a), "B": _make_series(b)})
    result = fit_composite(r_idx, panel)
    assert min(result.weights.values()) >= 0.0
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_fit_composite_beats_single_asset_te() -> None:
    """The composite's TE must be no worse than the best single asset, since
    each unit vector is in the simplex's feasible set."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.0005, 0.011, size=250).tolist()
    b = rng.normal(0.0004, 0.013, size=250).tolist()
    c = rng.normal(0.0003, 0.015, size=250).tolist()
    idx_vals = [0.5 * x + 0.3 * y + 0.2 * z for x, y, z in zip(a, b, c, strict=True)]
    r_idx = _make_series(idx_vals)
    panel = pd.DataFrame({"A": _make_series(a), "B": _make_series(b), "C": _make_series(c)})
    result = fit_composite(r_idx, panel)
    # Compare against the best single-asset TE (annualized).
    single_te = []
    import math
    for col in panel.columns:
        diff = panel[col] - r_idx
        single_te.append(float(diff.std(ddof=0) * math.sqrt(252)))
    assert result.te <= min(single_te) + 1e-9


def test_fit_composite_rejects_infeasible_bounds() -> None:
    r_idx = _make_series([0.001] * 100)
    panel = pd.DataFrame({"A": _make_series([0.001] * 100), "B": _make_series([0.0009] * 100)})
    # k=2, max_weight=0.3 → max possible sum = 0.6 < 1.0
    with pytest.raises(MiuOptimizerError):
        fit_composite(r_idx, panel, max_weight=0.3)


def test_fit_composite_rejects_constant_only_columns() -> None:
    r_idx = _make_series([0.001] * 100)
    panel = pd.DataFrame({"FLAT": _make_series([0.0] * 100)})
    with pytest.raises(MiuOptimizerError):
        fit_composite(r_idx, panel)


def test_fit_composite_rejects_empty_panel() -> None:
    r_idx = _make_series([0.001] * 100)
    with pytest.raises(MiuOptimizerError):
        fit_composite(r_idx, pd.DataFrame())
