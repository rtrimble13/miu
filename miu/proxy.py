"""Single-best-ETF proxy ranking.

Given an index daily-return series and a panel of candidate ETF daily returns,
compute per-ETF tracking error / correlation / beta / R² and return them
sorted ascending by annualized tracking error.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from miu.etf import align_returns
from miu.fmp.models import EtfProfile

log = logging.getLogger("miu.proxy")

TRADING_DAYS_PER_YEAR = 252


@dataclass
class ProxyMetric:
    ticker: str
    te: float  # annualized tracking error, decimal
    corr: float
    beta: float
    r2: float
    expense_ratio: float | None
    aum: float | None
    n_obs: int
    name: str | None = None


def _annualized_te(diff: pd.Series) -> float:
    if len(diff) < 2:
        return float("nan")
    return float(diff.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _beta(r_etf: pd.Series, r_idx: pd.Series) -> float:
    var_idx = float(r_idx.var(ddof=0))
    if var_idx <= 0 or math.isnan(var_idx):
        return float("nan")
    cov = float(np.cov(r_etf.values, r_idx.values, ddof=0)[0, 1])
    return cov / var_idx


def compute_metric(
    r_idx: pd.Series,
    r_etf: pd.Series,
    *,
    ticker: str,
    profile: EtfProfile | None,
) -> ProxyMetric:
    """All metrics for a single ETF on already-aligned return series."""
    n = len(r_idx)
    if n < 2:
        return ProxyMetric(
            ticker=ticker,
            te=float("nan"),
            corr=float("nan"),
            beta=float("nan"),
            r2=float("nan"),
            expense_ratio=profile.expense_ratio if profile else None,
            aum=profile.aum if profile else None,
            n_obs=n,
            name=profile.name if profile else None,
        )
    diff = r_etf - r_idx
    te = _annualized_te(diff)
    # corr/beta on a zero-variance input is mathematically undefined; numpy
    # emits RuntimeWarnings via divide-by-zero. We translate those into the
    # NaN we already handle downstream.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = float(r_idx.corr(r_etf))
        beta = _beta(r_etf, r_idx)
    r2 = corr * corr if not math.isnan(corr) else float("nan")
    return ProxyMetric(
        ticker=ticker,
        te=te,
        corr=corr,
        beta=beta,
        r2=r2,
        expense_ratio=profile.expense_ratio if profile else None,
        aum=profile.aum if profile else None,
        n_obs=n,
        name=profile.name if profile else None,
    )


def rank_proxies(
    index_returns: pd.Series,
    etf_panel: pd.DataFrame,
    profiles: dict[str, EtfProfile],
    *,
    min_overlap: int = 60,
) -> list[ProxyMetric]:
    """Compute and rank metrics for every candidate in the panel.

    Each ETF is aligned individually against the index so that one ETF's
    short history does not truncate the others. Candidates with fewer than
    `min_overlap` aligned observations are still reported but with `te=NaN`.
    Results are sorted ascending by TE; NaN TE sinks to the bottom, ties
    broken by higher R².
    """
    metrics: list[ProxyMetric] = []
    for ticker in etf_panel.columns:
        single = etf_panel[[ticker]].dropna()
        r_idx, panel = align_returns(index_returns, single)
        prof = profiles.get(ticker)
        if panel.empty:
            metrics.append(
                ProxyMetric(
                    ticker=ticker,
                    te=float("nan"),
                    corr=float("nan"),
                    beta=float("nan"),
                    r2=float("nan"),
                    expense_ratio=prof.expense_ratio if prof else None,
                    aum=prof.aum if prof else None,
                    n_obs=0,
                    name=prof.name if prof else None,
                )
            )
            continue
        r_etf = panel[ticker]
        n_obs = len(r_idx)
        if n_obs < min_overlap:
            log.warning(
                "%s: only %d aligned observations (< %d); reporting but TE flagged",
                ticker,
                n_obs,
                min_overlap,
            )
            m = compute_metric(r_idx, r_etf, ticker=ticker, profile=profiles.get(ticker))
            # Demote: keep raw stats but blank the TE so the sort sinks it.
            m = ProxyMetric(
                ticker=m.ticker,
                te=float("nan"),
                corr=m.corr,
                beta=m.beta,
                r2=m.r2,
                expense_ratio=m.expense_ratio,
                aum=m.aum,
                n_obs=m.n_obs,
                name=m.name,
            )
            metrics.append(m)
            continue
        metrics.append(
            compute_metric(r_idx, r_etf, ticker=ticker, profile=profiles.get(ticker))
        )

    def _sort_key(m: ProxyMetric) -> tuple[int, float, float]:
        # (NaN-TE sinks to bottom, then ascending TE, then descending R²)
        te_nan = 1 if math.isnan(m.te) else 0
        te = m.te if not math.isnan(m.te) else float("inf")
        # Negate r2 so higher comes first on ascending sort.
        r2_key = -m.r2 if not math.isnan(m.r2) else 0.0
        return (te_nan, te, r2_key)

    metrics.sort(key=_sort_key)
    return metrics


Strategy = Literal["best-te"]


def recommend(
    index_returns: pd.Series,
    etf_panel: pd.DataFrame,
    profiles: dict[str, EtfProfile],
    *,
    min_overlap: int = 60,
) -> tuple[ProxyMetric | None, list[ProxyMetric]]:
    """Pick the single best proxy by annualized tracking error.

    Returns (winner, full_table). Winner is None when no candidate has enough
    overlap with the index.
    """
    table = rank_proxies(index_returns, etf_panel, profiles, min_overlap=min_overlap)
    winner = next((m for m in table if not math.isnan(m.te)), None)
    return winner, table
