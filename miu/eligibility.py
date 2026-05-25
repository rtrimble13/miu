"""Point-in-time eligibility checks (spec §5).

Pure functions. No reads of "now", no implicit currency assumptions — the
caller passes in the as-of date and the prices/market-caps it has loaded.
A constituent is eligible iff, as of `t`:

  * ipo_date + 180d <= t                     (seasoning)
  * t < delisting_date (or delisting_date is None)
  * sector/industry filter still matches
  * market cap at t-1 >= min_market_cap
  * a price observation exists within the 5 trading days before t
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from miu.universe import Constituent

SEASONING_DAYS = 180
PRICE_WINDOW_DAYS = 7  # 5 trading days <= 7 calendar days


@dataclass(frozen=True)
class EligibilityInputs:
    """All the point-in-time data the gate needs for one constituent on date `t`."""

    last_price_date: date | None
    market_cap_t_minus_1: float | None


def is_eligible(
    constituent: Constituent,
    t: date,
    inputs: EligibilityInputs,
    *,
    sector: str | None = None,
    industry: str | None = None,
    min_market_cap: float = 0.0,
) -> bool:
    if constituent.ipo_date is not None and t < constituent.ipo_date + timedelta(days=SEASONING_DAYS):
        return False
    if constituent.delisting_date is not None and t >= constituent.delisting_date:
        return False
    if sector and (constituent.sector or "").strip().lower() != sector.lower():
        return False
    if industry and (constituent.industry or "").strip().lower() != industry.lower():
        return False
    if min_market_cap > 0:
        if inputs.market_cap_t_minus_1 is None:
            return False
        if inputs.market_cap_t_minus_1 < min_market_cap:
            return False
    if inputs.last_price_date is None:
        return False
    if (t - inputs.last_price_date).days > PRICE_WINDOW_DAYS:
        return False
    if inputs.last_price_date > t:
        return False
    return True


def select_eligible(
    constituents: list[Constituent],
    t: date,
    inputs_by_entity: dict[str, EligibilityInputs],
    *,
    sector: str | None = None,
    industry: str | None = None,
    min_market_cap: float = 0.0,
    max_constituents: int | None = None,
) -> list[Constituent]:
    """Return the eligible subset of `constituents` at date `t`.

    If `max_constituents` is set, rank by point-in-time market cap desc and
    keep the top N.
    """
    passed: list[tuple[Constituent, float]] = []
    for c in constituents:
        info = inputs_by_entity.get(c.entity_id)
        if info is None:
            continue
        if not is_eligible(
            c,
            t,
            info,
            sector=sector,
            industry=industry,
            min_market_cap=min_market_cap,
        ):
            continue
        passed.append((c, info.market_cap_t_minus_1 or 0.0))

    if max_constituents is not None and len(passed) > max_constituents:
        passed.sort(key=lambda pair: pair[1], reverse=True)
        passed = passed[:max_constituents]

    # Preserve original ordering when no cap was applied
    if max_constituents is None:
        return [c for c, _ in passed]
    return [c for c, _ in passed]
