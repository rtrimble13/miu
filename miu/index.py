"""Daily index calculation engine (spec §7).

Inputs:
  * a Constituent universe
  * a price panel: entity_id -> {date: adjusted_close}
  * a market-cap panel: entity_id -> {date: market_cap}
  * optional M&A events keyed by entity_id

Outputs:
  * a time-series DataFrame [date, index_level, daily_return, n_constituents]
  * a constituents DataFrame [date, entity_id, ticker, weight, is_rebalance_date]
  * a summary dict
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from miu.eligibility import EligibilityInputs, select_eligible
from miu.universe import Constituent
from miu.weights import Weights, compute_weights

log = logging.getLogger("miu.index")

Rebalance = Literal["monthly", "quarterly", "annual", "none"]


@dataclass(frozen=True)
class MnaResolution:
    """Resolved corporate-action effect at delisting.

    For cash deals: `cash_value` is the terminal per-share value.
    For stock-for-stock with acquirer in the index: `acquirer_id` and `ratio`
    redistribute the position. Otherwise we fall back to last traded price
    (handled in the engine, not here).
    """

    cash_value: float | None = None
    acquirer_id: str | None = None
    ratio: float | None = None


@dataclass
class IndexResult:
    series: pd.DataFrame  # columns: date, index_level, daily_return, n_constituents
    constituents: pd.DataFrame  # columns: date, entity_id, ticker, weight, is_rebalance_date
    summary: dict[str, float | int]


@dataclass
class EngineConfig:
    weighting: str
    start: date
    end: date
    rebalance: Rebalance = "quarterly"
    base_value: float = 1000.0
    min_market_cap: float = 0.0
    max_constituents: int | None = None
    sector: str | None = None
    industry: str | None = None


@dataclass
class _Snapshot:
    rebalance_date: date
    constituents: list[Constituent]
    weights: Weights
    # Reference prices used as the "purchase price" basis for drifting weights
    ref_prices: dict[str, float] = field(default_factory=dict)


class IndexEngine:
    def __init__(
        self,
        constituents: list[Constituent],
        prices: dict[str, dict[date, float]],
        market_caps: dict[str, dict[date, float]],
        config: EngineConfig,
        *,
        mna_resolutions: dict[str, MnaResolution] | None = None,
    ):
        self.constituents = constituents
        self.prices = prices
        self.market_caps = market_caps
        self.config = config
        self.mna_resolutions = mna_resolutions or {}
        self._by_id = {c.entity_id: c for c in constituents}

    def run(self) -> IndexResult:
        cal = self._trading_calendar()
        if not cal:
            raise ValueError("no trading days observed in the requested window")

        rebal_days = _rebalance_dates(cal, self.config.rebalance)
        log.info("trading days: %d  rebalances: %d", len(cal), len(rebal_days))

        series_rows: list[dict] = []
        const_rows: list[dict] = []

        level = self.config.base_value
        snapshot: _Snapshot | None = None
        prev_day: date | None = None

        for day in cal:
            # `snapshot` here is the basket held entering today. On the
            # bootstrap day we have no prior basket; otherwise it's whatever
            # was set at the last rebalance.
            bootstrap = snapshot is None
            is_rebal_today = bootstrap or (day in rebal_days)

            # Compute today's return against the OLD basket — the one held
            # entering the day — BEFORE any EOD rebalance. On bootstrap the
            # return is zero (nothing to mark against).
            if not bootstrap and snapshot is not None and snapshot.weights and prev_day is not None:
                day_return = self._period_return(snapshot, prev_day, day)
            else:
                day_return = 0.0

            # Apply EOD rebalance (if any) AFTER the return is locked in.
            if is_rebal_today:
                snapshot = self._rebalance(day)

            # Emit constituent rows. On a rebalance day, show the freshly
            # selected basket; on a non-rebalance day, show the drifted basket.
            if snapshot is not None and snapshot.weights:
                day_weights = (
                    snapshot.weights
                    if is_rebal_today
                    else self._drifted_weights(snapshot, day)
                )
                for eid, w in day_weights.items():
                    const_rows.append(
                        {
                            "date": day,
                            "entity_id": eid,
                            "ticker": self._by_id[eid].ticker_at(day),
                            "weight": w,
                            "is_rebalance_date": is_rebal_today,
                        }
                    )
                n_const = sum(1 for w in day_weights.values() if w > 0)
            else:
                n_const = 0

            if prev_day is not None:
                level = level * (1.0 + day_return)
            series_rows.append(
                {
                    "date": day,
                    "index_level": level,
                    "daily_return": day_return,
                    "n_constituents": n_const,
                }
            )
            prev_day = day

        series_df = pd.DataFrame(series_rows)
        const_df = pd.DataFrame(const_rows)
        summary = _summarize(series_df, const_df, self.constituents, self.config)
        return IndexResult(series=series_df, constituents=const_df, summary=summary)

    # ----- internals -----

    def _trading_calendar(self) -> list[date]:
        days: set[date] = set()
        for series in self.prices.values():
            for d in series:
                if self.config.start <= d <= self.config.end:
                    days.add(d)
        return sorted(days)

    def _rebalance(self, day: date) -> _Snapshot:
        inputs = self._eligibility_inputs(day)
        eligible = select_eligible(
            self.constituents,
            day,
            inputs,
            sector=self.config.sector,
            industry=self.config.industry,
            min_market_cap=self.config.min_market_cap,
            max_constituents=self.config.max_constituents,
        )
        prices_t1 = self._prices_on(day - timedelta(days=1), eligible) or self._prices_on(day, eligible)
        mcaps_t1 = self._mcaps_on(day - timedelta(days=1), eligible) or self._mcaps_on(day, eligible)
        weights = compute_weights(
            self.config.weighting,
            eligible,
            prices=prices_t1,
            market_caps=mcaps_t1,
            as_of=day,
        )
        # Drift's "purchase basis" must share an as-of with the weights, else
        # the basket implies inconsistent share counts on subsequent days.
        snap = _Snapshot(
            rebalance_date=day,
            constituents=eligible,
            weights=weights,
            ref_prices=dict(prices_t1),
        )
        return snap

    def _eligibility_inputs(self, t: date) -> dict[str, EligibilityInputs]:
        out: dict[str, EligibilityInputs] = {}
        for c in self.constituents:
            last_price_date = self._last_price_date_before(c.entity_id, t)
            series = self.market_caps.get(c.entity_id, {})
            mcap = self._last_value_before(series, t - timedelta(days=1))
            if mcap is None:
                # At inception there's no t-1 observation; fall back to t.
                mcap = self._last_value_before(series, t)
            out[c.entity_id] = EligibilityInputs(
                last_price_date=last_price_date,
                market_cap_t_minus_1=mcap,
            )
        return out

    def _last_price_date_before(self, eid: str, t: date) -> date | None:
        series = self.prices.get(eid, {})
        latest: date | None = None
        for d in series:
            if d <= t and (latest is None or d > latest):
                latest = d
        return latest

    def _last_value_before(self, series: dict[date, float], t: date) -> float | None:
        latest: date | None = None
        for d in series:
            if d <= t and (latest is None or d > latest):
                latest = d
        return None if latest is None else series[latest]

    def _prices_on(self, day: date, names: list[Constituent]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in names:
            series = self.prices.get(c.entity_id)
            if not series:
                continue
            best: tuple[date, float] | None = None
            for d, v in series.items():
                if d <= day and (best is None or d > best[0]):
                    best = (d, v)
            if best is not None:
                out[c.entity_id] = best[1]
        return out

    def _mcaps_on(self, day: date, names: list[Constituent]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in names:
            series = self.market_caps.get(c.entity_id)
            if not series:
                continue
            best: tuple[date, float] | None = None
            for d, v in series.items():
                if d <= day and (best is None or d > best[0]):
                    best = (d, v)
            if best is not None:
                out[c.entity_id] = best[1]
        return out

    def _period_return(self, snap: _Snapshot, prev_day: date, day: date) -> float:
        """Return of the held basket between prev_day and day.

        Drifts weights with price moves (no rebalance between rebalance dates).
        Applies M&A resolution at the delisting date.
        """
        ret = 0.0
        drifted = self._drifted_weights(snap, prev_day)
        for eid, weight in drifted.items():
            if weight <= 0:
                continue
            r = self._security_return(eid, prev_day, day)
            ret += weight * r
        return ret

    def _drifted_weights(self, snap: _Snapshot, day: date) -> Weights:
        values: dict[str, float] = {}
        for eid, weight in snap.weights.items():
            if weight <= 0:
                continue
            p_ref = snap.ref_prices.get(eid)
            if p_ref is None or p_ref <= 0:
                continue
            p_day = self._price_at(eid, day)
            if p_day is None or p_day <= 0:
                continue
            # Normalize to an arbitrary base portfolio value of 1.0.
            shares = weight / p_ref
            values[eid] = shares * p_day
        total = sum(values.values())
        if total <= 0:
            return {eid: w for eid, w in snap.weights.items() if w > 0}
        return {eid: val / total for eid, val in values.items()}

    def _security_return(self, eid: str, prev_day: date, day: date) -> float:
        c = self._by_id.get(eid)
        if c is None:
            return 0.0
        delisting = c.delisting_date

        p_prev = self._price_at(eid, prev_day)
        if p_prev is None or p_prev <= 0:
            return 0.0

        if delisting is not None and prev_day < delisting <= day:
            mna = self.mna_resolutions.get(eid)
            if mna and mna.cash_value is not None:
                return (mna.cash_value / p_prev) - 1.0
            if mna and mna.acquirer_id and mna.ratio is not None:
                # Stock-for-stock: terminal value = ratio shares of the
                # acquirer marked at acquirer's price on the delisting day.
                p_acq = self._price_at(mna.acquirer_id, day)
                if p_acq is not None and p_acq > 0:
                    return ((mna.ratio * p_acq) / p_prev) - 1.0
            # Fallback: last observed price within window
            last = self._price_at(eid, delisting - timedelta(days=1)) or p_prev
            return (last / p_prev) - 1.0
        if delisting is not None and day >= delisting:
            mna = self.mna_resolutions.get(eid)
            if mna and mna.acquirer_id and mna.ratio is not None:
                # Post-delisting: the converted position tracks the acquirer.
                p_acq_prev = self._price_at(mna.acquirer_id, prev_day)
                p_acq_now = self._price_at(mna.acquirer_id, day)
                if (
                    p_acq_prev is not None
                    and p_acq_prev > 0
                    and p_acq_now is not None
                    and p_acq_now > 0
                ):
                    return (p_acq_now / p_acq_prev) - 1.0
            return 0.0  # held flat (already resolved on the delisting day)

        p_now = self._price_at(eid, day)
        if p_now is None or p_now <= 0:
            return 0.0
        return (p_now / p_prev) - 1.0

    def _price_at(self, eid: str, day: date) -> float | None:
        series = self.prices.get(eid)
        if not series:
            return None
        if day in series:
            return series[day]
        best: tuple[date, float] | None = None
        for d, v in series.items():
            if d <= day and (best is None or d > best[0]):
                best = (d, v)
        return None if best is None else best[1]


def _rebalance_dates(calendar: list[date], cadence: Rebalance) -> set[date]:
    if cadence == "none" or not calendar:
        return {calendar[0]} if calendar else set()
    out: set[date] = {calendar[0]}
    by_month: dict[tuple[int, int], list[date]] = {}
    for d in calendar:
        by_month.setdefault((d.year, d.month), []).append(d)
    months = sorted(by_month)
    if cadence == "monthly":
        keep_months = months
    elif cadence == "quarterly":
        keep_months = [m for m in months if m[1] in (1, 4, 7, 10)]
    elif cadence == "annual":
        keep_months = [m for m in months if m[1] == 1]
    else:
        keep_months = []
    for m in keep_months:
        out.add(by_month[m][0])
    return out


def _summarize(
    series: pd.DataFrame,
    constituents: pd.DataFrame,
    universe: Iterable[Constituent],
    config: EngineConfig,
) -> dict[str, float | int]:
    if series.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "max_drawdown": 0.0,
            "avg_constituents": 0,
            "rebalances": 0,
            "delisted_in_sample": 0,
        }
    s = series.copy()
    s = s.reset_index(drop=True)
    first = float(s["index_level"].iloc[0])
    last = float(s["index_level"].iloc[-1])
    total_return = last / first - 1.0 if first else 0.0

    days = (s["date"].iloc[-1] - s["date"].iloc[0]).days
    years = max(days / 365.25, 1e-9)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if (1.0 + total_return) > 0 else float("nan")

    rets = s["daily_return"].astype(float)
    ann_vol = float(rets.std(ddof=0) * math.sqrt(252)) if len(rets) > 1 else 0.0

    running_max = s["index_level"].cummax()
    drawdown = (s["index_level"] / running_max) - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    delisted = sum(
        1
        for c in universe
        if c.delisting_date is not None and config.start <= c.delisting_date <= config.end
    )
    if not constituents.empty:
        rebal_dates = constituents.loc[constituents["is_rebalance_date"], "date"].unique()
        rebalances = len(rebal_dates)
    else:
        rebalances = 0
    avg_n = float(s["n_constituents"].mean()) if not s.empty else 0.0
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return) if not math.isnan(ann_return) else 0.0,
        "annualized_vol": float(ann_vol),
        "max_drawdown": float(max_dd),
        "avg_constituents": float(avg_n),
        "rebalances": int(rebalances),
        "delisted_in_sample": int(delisted),
    }
