"""ETF discovery, metadata fetch, and daily-returns panel loading.

This module is the data layer that `proxy.py` (ranking) and `composite.py`
(constrained OLS) consume. It also reads back index files written by
`miu/output.py` so the recommend/composite commands can target a previously
built index without re-running the engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from miu.config import MiuConfigError
from miu.fmp import endpoints as ep
from miu.fmp.client import FmpClient
from miu.fmp.models import EtfProfile

log = logging.getLogger("miu.etf")


@dataclass
class IndexReturns:
    """Daily returns of a target custom index, plus a description.

    `returns` is a pandas Series indexed by `datetime.date` of decimal daily
    returns (0.01 = +1%). The first observation is dropped — by construction
    a return series needs a preceding price.
    """

    returns: pd.Series
    meta: dict[str, Any] = field(default_factory=dict)

    def slice(self, start: date | None, end: date | None) -> IndexReturns:
        s = self.returns
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        return IndexReturns(returns=s, meta=dict(self.meta))


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def load_index_from_file(path: Path) -> IndexReturns:
    """Read an index series previously produced by `miu build`.

    Supports both shapes that `miu/output.py` writes:
      * CSV: columns `date, index_level, daily_return, n_constituents`.
        If `daily_return` is missing, it's recomputed from `index_level`.
      * JSON: `{"meta": {...}, "series": [{"date","level","return","n"}, ...]}`.
    """
    path = Path(path)
    if not path.exists():
        raise MiuConfigError(f"index file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise MiuConfigError(f"{path}: CSV missing `date` column")
        df["date"] = df["date"].map(_coerce_date)
        if "daily_return" in df.columns:
            df["__r"] = df["daily_return"].astype(float)
        elif "index_level" in df.columns:
            df["__r"] = df["index_level"].astype(float).pct_change()
        elif "level" in df.columns:
            df["__r"] = df["level"].astype(float).pct_change()
        else:
            raise MiuConfigError(
                f"{path}: need one of `daily_return`, `index_level`, or `level`"
            )
        s = pd.Series(df["__r"].values, index=df["date"], name="index_return").dropna()
        meta: dict[str, Any] = {"source": str(path)}
    elif suffix == ".json":
        body = json.loads(path.read_text())
        if not isinstance(body, dict) or "series" not in body:
            raise MiuConfigError(f"{path}: JSON missing `series` array")
        series_rows = body["series"]
        df = pd.DataFrame(series_rows)
        if df.empty or "date" not in df.columns:
            raise MiuConfigError(f"{path}: empty or missing `date` in series")
        df["date"] = df["date"].map(_coerce_date)
        if "return" in df.columns:
            df["__r"] = df["return"].astype(float)
        elif "level" in df.columns:
            df["__r"] = df["level"].astype(float).pct_change()
        elif "index_level" in df.columns:
            df["__r"] = df["index_level"].astype(float).pct_change()
        else:
            raise MiuConfigError(f"{path}: need one of `return`, `level`, or `index_level`")
        s = pd.Series(df["__r"].values, index=df["date"], name="index_return").dropna()
        meta = {"source": str(path), **(body.get("meta") or {})}
    else:
        raise MiuConfigError(f"unsupported index file extension {suffix!r} (use .csv or .json)")

    # The first row of a synthesized return series is NaN; drop it (already
    # done by .dropna() above). Guard against empty result.
    if s.empty:
        raise MiuConfigError(f"{path}: no usable return rows after parsing")
    s.index = pd.Index(s.index, name="date")
    return IndexReturns(returns=s, meta=meta)


async def discover_etf_candidates(
    client: FmpClient,
    *,
    sector: str | None,
    industry: str | None,
    exchanges: list[str] | None = None,
    min_aum: float = 0.0,
    max_candidates: int = 25,
) -> list[str]:
    """Find ETFs whose sector/industry matches the target index.

    Returns tickers sorted by market cap descending. `min_aum` is honored at
    the screener level via the `marketCap` proxy when available, and again
    at metadata-fetch time if AUM is reported.
    """
    rows = await ep.etf_search(
        client,
        sector=sector,
        industry=industry,
        exchanges=exchanges,
    )
    rows.sort(key=lambda r: (-(r.market_cap or 0.0), r.symbol))
    out: list[str] = []
    for r in rows:
        if min_aum > 0 and (r.market_cap or 0.0) < min_aum:
            continue
        out.append(r.symbol.upper())
        if len(out) >= max_candidates:
            break
    return out


async def load_etf_profiles(
    client: FmpClient, symbols: list[str]
) -> dict[str, EtfProfile]:
    """Fetch ETF info concurrently. Missing rows simply omitted."""
    unique = sorted({s.upper() for s in symbols if s})

    async def fetch(sym: str) -> tuple[str, EtfProfile | None]:
        try:
            p = await ep.etf_info(client, sym)
        except Exception as exc:  # noqa: BLE001 — keep the sweep alive
            log.warning("etf_info failed for %s: %s", sym, exc)
            return sym, None
        return sym, p

    results: dict[str, EtfProfile] = {}
    tasks = [asyncio.create_task(fetch(s)) for s in unique]
    for fut in asyncio.as_completed(tasks):
        sym, p = await fut
        if p is not None:
            results[sym] = p
    return results


async def load_etf_returns_panel(
    client: FmpClient,
    symbols: list[str],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Wide DataFrame: rows = trading dates, columns = tickers, values = daily returns.

    Uses dividend-adjusted prices (same endpoint as constituent data) so the
    comparison against the index is apples-to-apples on a total-return basis.
    Tickers that return no data are dropped silently with a warning.
    """
    unique = sorted({s.upper() for s in symbols if s})

    async def fetch(sym: str) -> tuple[str, pd.Series | None]:
        try:
            rows = await ep.historical_prices(client, sym, start=start, end=end)
        except Exception as exc:  # noqa: BLE001 — partial panels are tolerable
            log.warning("historical_prices failed for %s: %s", sym, exc)
            return sym, None
        if not rows:
            return sym, None
        # Prefer adj_close; HistoricalPrice.price returns adj_close or close.
        by_date: dict[date, float] = {}
        for r in rows:
            if r.price is not None:
                by_date.setdefault(r.date, r.price)
        if not by_date:
            return sym, None
        idx = sorted(by_date)
        prices = pd.Series([by_date[d] for d in idx], index=pd.Index(idx, name="date"))
        return sym, prices.pct_change().dropna().rename(sym)

    tasks = [asyncio.create_task(fetch(s)) for s in unique]
    series: dict[str, pd.Series] = {}
    for fut in asyncio.as_completed(tasks):
        sym, s = await fut
        if s is not None and not s.empty:
            series[sym] = s

    if not series:
        return pd.DataFrame()
    panel = pd.concat(series, axis=1)
    panel.index = pd.Index(panel.index, name="date")
    return panel


def align_returns(
    index_returns: pd.Series, etf_panel: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join the index return series with the ETF returns panel by date.

    Drops any row with NaN in any column. Returns (aligned_index, aligned_panel)
    sharing the same date index.
    """
    if etf_panel.empty:
        return index_returns.iloc[0:0], etf_panel
    joined = etf_panel.join(index_returns.rename("__INDEX__"), how="inner").dropna(how="any")
    if joined.empty:
        return index_returns.iloc[0:0], etf_panel.iloc[0:0]
    idx = joined["__INDEX__"]
    panel = joined.drop(columns="__INDEX__")
    return idx, panel
