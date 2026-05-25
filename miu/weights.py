"""The three weighting schemes (spec §6).

Each function is pure, takes the eligible constituent list plus the
point-in-time data dicts, and returns `{entity_id: weight}` summing to 1.0.
"""

from __future__ import annotations

from datetime import date

from miu.universe import Constituent

Weights = dict[str, float]


def equal_weights(constituents: list[Constituent], _date: date) -> Weights:
    n = len(constituents)
    if n == 0:
        return {}
    w = 1.0 / n
    return {c.entity_id: w for c in constituents}


def market_cap_weights(
    constituents: list[Constituent],
    market_caps: dict[str, float],
    _date: date,
) -> Weights:
    valid = [(c, market_caps[c.entity_id]) for c in constituents if c.entity_id in market_caps and market_caps[c.entity_id] > 0]
    total = sum(mc for _, mc in valid)
    if total <= 0 or not valid:
        return {}
    return {c.entity_id: mc / total for c, mc in valid}


def price_weights(
    constituents: list[Constituent],
    prices: dict[str, float],
    _date: date,
) -> Weights:
    valid = [(c, prices[c.entity_id]) for c in constituents if c.entity_id in prices and prices[c.entity_id] > 0]
    total = sum(p for _, p in valid)
    if total <= 0 or not valid:
        return {}
    return {c.entity_id: p / total for c, p in valid}


def compute_weights(
    scheme: str,
    constituents: list[Constituent],
    *,
    prices: dict[str, float],
    market_caps: dict[str, float],
    as_of: date,
) -> Weights:
    """Dispatch helper used by the engine."""
    if scheme == "equal":
        return equal_weights(constituents, as_of)
    if scheme == "market-cap":
        return market_cap_weights(constituents, market_caps, as_of)
    if scheme == "price":
        return price_weights(constituents, prices, as_of)
    raise ValueError(f"unknown weighting scheme: {scheme!r}")
