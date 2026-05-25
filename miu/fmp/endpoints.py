"""Typed endpoint wrappers around FmpClient.

Each function returns parsed pydantic models. The client returns raw JSON;
endpoints adapt FMP shapes (which are sometimes a bare list, sometimes a
dict with a single key) into the model classes downstream code uses.
"""

from __future__ import annotations

from datetime import date
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from miu.cache import TTL
from miu.config import MiuApiError
from miu.fmp.client import FmpClient
from miu.fmp.models import (
    DelistedRow,
    HistoricalMarketCap,
    HistoricalPrice,
    MnaEvent,
    Profile,
    ScreenerRow,
    SP500Membership,
    SymbolChange,
)


def _as_rows(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # FMP sometimes wraps results under a single top-level key.
        for v in payload.values():
            if isinstance(v, list):
                return v
        return [payload]
    raise ValueError(f"unexpected payload shape: {type(payload)}")


T = TypeVar("T", bound=BaseModel)


def _parse(rows: list[dict[str, Any]], model_cls: type[T], endpoint: str) -> list[T]:
    out: list[T] = []
    for r in rows:
        try:
            out.append(model_cls.model_validate(r))
        except ValidationError as exc:
            raise MiuApiError(endpoint, {}, None, f"parse failure: {exc}") from exc
    return out


async def screener(
    client: FmpClient,
    *,
    sector: str | None = None,
    industry: str | None = None,
    exchanges: list[str] | None = None,
    is_actively_trading: bool = True,
    limit: int = 10000,
) -> list[ScreenerRow]:
    params: dict[str, Any] = {"limit": limit, "country": "US"}
    if sector:
        params["sector"] = sector
    if industry:
        params["industry"] = industry
    if exchanges:
        params["exchange"] = ",".join(exchanges)
    if is_actively_trading is not None:
        params["isActivelyTrading"] = "true" if is_actively_trading else "false"
    rows = _as_rows(await client.get("/company-screener", params, ttl=TTL.SNAPSHOT))
    return _parse(rows, ScreenerRow, "/company-screener")


async def delisted_companies(client: FmpClient, *, max_pages: int = 50) -> list[DelistedRow]:
    all_rows: list[dict[str, Any]] = []
    for page in range(max_pages):
        payload = await client.get("/delisted-companies", {"page": page}, ttl=TTL.SNAPSHOT)
        rows = _as_rows(payload)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < 100:
            break
    return _parse(all_rows, DelistedRow, "/delisted-companies")


async def symbol_changes(client: FmpClient) -> list[SymbolChange]:
    rows = _as_rows(await client.get("/symbol-change", {}, ttl=TTL.REFERENCE))
    return _parse(rows, SymbolChange, "/symbol-change")


async def profile(client: FmpClient, symbol: str) -> Profile | None:
    rows = _as_rows(await client.get("/profile", {"symbol": symbol}, ttl=TTL.REFERENCE))
    if not rows:
        return None
    parsed = _parse(rows[:1], Profile, "/profile")
    return parsed[0] if parsed else None


async def historical_prices(
    client: FmpClient,
    symbol: str,
    *,
    start: date,
    end: date,
) -> list[HistoricalPrice]:
    payload = await client.get(
        "/historical-price-eod/dividend-adjusted",
        {"symbol": symbol, "from": str(start), "to": str(end)},
        ttl=TTL.HISTORICAL,
    )
    rows = _as_rows(payload)
    parsed = _parse(rows, HistoricalPrice, "/historical-price-eod/dividend-adjusted")
    parsed.sort(key=lambda p: p.date)
    return parsed


async def historical_market_cap(
    client: FmpClient,
    symbol: str,
    *,
    start: date,
    end: date,
) -> list[HistoricalMarketCap]:
    payload = await client.get(
        "/historical-market-capitalization",
        {"symbol": symbol, "from": str(start), "to": str(end)},
        ttl=TTL.HISTORICAL,
    )
    rows = _as_rows(payload)
    parsed = _parse(rows, HistoricalMarketCap, "/historical-market-capitalization")
    parsed.sort(key=lambda m: m.date)
    return parsed


async def mergers_acquisitions(
    client: FmpClient,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[MnaEvent]:
    params: dict[str, Any] = {}
    if start:
        params["from"] = str(start)
    if end:
        params["to"] = str(end)
    rows = _as_rows(await client.get("/mergers-acquisitions", params, ttl=TTL.HISTORICAL))
    return _parse(rows, MnaEvent, "/mergers-acquisitions")


async def historical_sp500(client: FmpClient) -> list[SP500Membership]:
    rows = _as_rows(await client.get("/historical-sp-500", {}, ttl=TTL.REFERENCE))
    return _parse(rows, SP500Membership, "/historical-sp-500")
