"""Survivorship-bias-free universe construction (spec §4).

For a given sector/industry filter and date range:
  1. Current candidates from /company-screener.
  2. Delisted candidates from /delisted-companies (US, ≥ start-1y).
  3. Historical S&P 500 sweep — names that may have already cycled out.
  4. Resolve ticker history via /symbol-change to a stable entity_id.
  5. Cache the resolved universe.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from miu.cache import DiskCache
from miu.config import MiuUniverseError
from miu.fmp import endpoints as ep
from miu.fmp.client import FmpClient
from miu.fmp.models import DelistedRow, Profile, ScreenerRow, SP500Membership, SymbolChange

log = logging.getLogger("miu.universe")

US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSEARCA"}


class TickerSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    start: date | None
    end: date | None  # None = current


class Constituent(BaseModel):
    model_config = ConfigDict(frozen=False)
    entity_id: str
    ticker_history: list[TickerSpan]
    ipo_date: date | None = None
    delisting_date: date | None = None
    delisting_reason: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None

    def ticker_at(self, when: date) -> str:
        """Return the symbol active at `when`. Falls back to the most recent."""
        for span in self.ticker_history:
            start = span.start or date.min
            end = span.end or date.max
            if start <= when <= end:
                return span.ticker
        # Fallback: most recent span by end date
        latest = max(
            self.ticker_history,
            key=lambda s: (s.end or date.max, s.start or date.min),
        )
        return latest.ticker

    @property
    def current_ticker(self) -> str:
        return self.ticker_at(date.today())


@dataclass(frozen=True)
class UniverseRequest:
    sector: str | None
    industry: str | None
    start: date
    end: date
    exchanges: tuple[str, ...]
    include_delisted: bool = True


async def build_universe(
    request: UniverseRequest,
    client: FmpClient,
    *,
    universe_cache_dir: Path | None = None,
) -> list[Constituent]:
    """Assemble the universe per spec §4. Cached by request hash."""
    cache = (
        DiskCache(universe_cache_dir)
        if universe_cache_dir is not None
        else DiskCache(client.settings.cache_dir / "universe")
    )
    cache_key = {
        "sector": request.sector,
        "industry": request.industry,
        "start": str(request.start),
        "end": str(request.end),
        "exchanges": list(request.exchanges),
        "include_delisted": request.include_delisted,
    }
    hit = cache.get("universe", cache_key)
    if hit is not None:
        log.info("universe cache hit (%d constituents)", len(hit.payload))
        return [Constituent.model_validate(c) for c in hit.payload]

    if not request.sector and not request.industry:
        raise MiuUniverseError("must specify sector or industry")

    exchanges_filter = set(request.exchanges) or US_EXCHANGES

    current = await _current_candidates(client, request, exchanges_filter)
    log.info("screener returned %d active candidates", len(current))

    delisted: list[DelistedRow] = []
    delisted_profiles: dict[str, Profile] = {}
    if request.include_delisted:
        delisted = await _delisted_candidates(client, request, exchanges_filter)
        log.info("delisted scan: %d candidate names to classify", len(delisted))
        delisted_profiles = await _gather_profiles(client, [d.symbol for d in delisted])

    sp_extras, sp_profiles = await _sp500_sweep(client, request, exchanges_filter)
    log.info("S&P sweep: %d extra candidates", len(sp_extras))

    changes = await ep.symbol_changes(client)
    chain_map = _build_chain_map(changes)

    matched_symbols = _collect_matches(
        current,
        delisted,
        delisted_profiles,
        sp_extras,
        sp_profiles,
        request,
    )

    constituents = _build_constituents(
        matched_symbols,
        current,
        delisted,
        {**delisted_profiles, **sp_profiles},
        chain_map,
    )

    cache.set("universe", cache_key, [c.model_dump(mode="json") for c in constituents], ttl=7 * 86400)
    log.info("universe assembled: %d constituents", len(constituents))
    return constituents


async def _current_candidates(
    client: FmpClient, request: UniverseRequest, exchanges: set[str]
) -> list[ScreenerRow]:
    rows = await ep.screener(
        client,
        sector=request.sector,
        industry=request.industry,
        exchanges=list(exchanges),
        is_actively_trading=True,
    )
    return [r for r in rows if _is_us_exchange(r.exchange_short_name or r.exchange, exchanges)]


async def _delisted_candidates(
    client: FmpClient, request: UniverseRequest, exchanges: set[str]
) -> list[DelistedRow]:
    rows = await ep.delisted_companies(client)
    cutoff = request.start - timedelta(days=365)
    out: list[DelistedRow] = []
    for r in rows:
        if not _is_us_exchange(r.exchange, exchanges):
            continue
        if r.delisted_date is not None and r.delisted_date < cutoff:
            continue
        out.append(r)
    return out


async def _sp500_sweep(
    client: FmpClient, request: UniverseRequest, exchanges: set[str]
) -> tuple[list[SP500Membership], dict[str, Profile]]:
    events = await ep.historical_sp500(client)
    symbols: set[str] = set()
    for e in events:
        # added symbol
        added = (e.symbol or e.added_security or "").strip()
        if added:
            symbols.add(added.split()[0].upper())
        if e.removed_ticker:
            symbols.add(e.removed_ticker.strip().upper())
    profiles = await _gather_profiles(client, sorted(symbols))
    matching: list[SP500Membership] = []
    for e in events:
        ticker = (e.symbol or e.added_security or "").split()[0].upper() if (e.symbol or e.added_security) else None
        if ticker and ticker in profiles:
            p = profiles[ticker]
            if _profile_matches(p, request) and _is_us_exchange(
                p.exchange_short_name or p.exchange, exchanges
            ):
                matching.append(e)
    return matching, profiles


async def _gather_profiles(client: FmpClient, symbols: Iterable[str]) -> dict[str, Profile]:
    unique = sorted({s for s in symbols if s})
    out: dict[str, Profile] = {}

    async def fetch(sym: str) -> tuple[str, Profile | None]:
        try:
            p = await ep.profile(client, sym)
        except Exception as exc:  # noqa: BLE001 — broad to keep the sweep going
            log.warning("profile fetch failed for %s: %s", sym, exc)
            return sym, None
        return sym, p

    tasks = [asyncio.create_task(fetch(s)) for s in unique]
    for fut in asyncio.as_completed(tasks):
        sym, p = await fut
        if p is not None:
            out[sym] = p
    return out


def _profile_matches(profile: Profile, request: UniverseRequest) -> bool:
    if profile.is_etf or profile.is_fund:
        return False
    if request.sector and (profile.sector or "").strip().lower() != request.sector.lower():
        return False
    if request.industry and (profile.industry or "").strip().lower() != request.industry.lower():
        return False
    return True


def _is_us_exchange(name: str | None, allowed: set[str]) -> bool:
    if not name:
        return False
    norm = name.strip().upper()
    if norm in allowed:
        return True
    # FMP returns things like "New York Stock Exchange" sometimes
    if "NYSE" in norm and "NYSE" in allowed:
        return True
    if "NASDAQ" in norm and "NASDAQ" in allowed:
        return True
    if "AMEX" in norm and "AMEX" in allowed:
        return True
    return False


def _build_chain_map(changes: list[SymbolChange]) -> dict[str, str]:
    """old_symbol -> new_symbol map. Walks chains forward to a terminal symbol."""
    direct = {c.old_symbol.upper(): c.new_symbol.upper() for c in changes}
    resolved: dict[str, str] = {}
    for old in direct:
        cur = old
        seen = {cur}
        while cur in direct and direct[cur] not in seen:
            cur = direct[cur]
            seen.add(cur)
        resolved[old] = cur
    return resolved


def _collect_matches(
    current: list[ScreenerRow],
    delisted: list[DelistedRow],
    delisted_profiles: dict[str, Profile],
    sp_extras: list[SP500Membership],
    sp_profiles: dict[str, Profile],
    request: UniverseRequest,
) -> set[str]:
    matched: set[str] = set()
    for r in current:
        matched.add(r.symbol.upper())
    for d in delisted:
        p = delisted_profiles.get(d.symbol)
        if p and _profile_matches(p, request):
            matched.add(d.symbol.upper())
    for e in sp_extras:
        ticker = (e.symbol or e.added_security or "").split()[0].upper() if (e.symbol or e.added_security) else None
        if ticker:
            matched.add(ticker)
    return matched


def _build_constituents(
    matched: set[str],
    current: list[ScreenerRow],
    delisted: list[DelistedRow],
    profiles: dict[str, Profile],
    chain_map: dict[str, str],
) -> list[Constituent]:
    current_by_sym = {r.symbol.upper(): r for r in current}
    delisted_by_sym = {d.symbol.upper(): d for d in delisted}

    # Group symbols by terminal entity (resolve forward through chain_map).
    grouped: dict[str, list[str]] = {}
    for sym in matched:
        terminal = chain_map.get(sym, sym)
        grouped.setdefault(terminal, []).append(sym)

    constituents: list[Constituent] = []
    for terminal, group in sorted(grouped.items()):
        ticker_history = _make_history(group, chain_map, delisted_by_sym)
        # Pick the most-informative profile (prefer current, fall back to any).
        profile = profiles.get(terminal)
        if profile is None:
            for s in group:
                if s in profiles:
                    profile = profiles[s]
                    break
        d_row = delisted_by_sym.get(terminal) or next(
            (delisted_by_sym[s] for s in group if s in delisted_by_sym), None
        )
        s_row = current_by_sym.get(terminal) or next(
            (current_by_sym[s] for s in group if s in current_by_sym), None
        )

        ipo = (profile.ipo_date if profile else None) or (d_row.ipo_date if d_row else None)
        delisting = d_row.delisted_date if d_row else None
        sector = (profile.sector if profile else None) or (s_row.sector if s_row else None)
        industry = (profile.industry if profile else None) or (s_row.industry if s_row else None)
        exchange = (
            (profile.exchange_short_name or profile.exchange if profile else None)
            or (s_row.exchange_short_name or s_row.exchange if s_row else None)
            or (d_row.exchange if d_row else None)
        )
        constituents.append(
            Constituent(
                entity_id=terminal,
                ticker_history=ticker_history,
                ipo_date=ipo,
                delisting_date=delisting,
                delisting_reason="delisted" if delisting else None,
                sector=sector,
                industry=industry,
                exchange=exchange,
            )
        )
    return constituents


def _make_history(
    group: list[str], chain_map: dict[str, str], delisted_by_sym: dict[str, DelistedRow]
) -> list[TickerSpan]:
    """Order the group by symbol-change chain and assign date spans."""
    inverse: dict[str, str] = {old: new for old, new in chain_map.items() if new in group or old in group}
    chain: list[str] = []
    starts = [s for s in group if s not in inverse]
    if not starts:
        starts = sorted(group)
    seen: set[str] = set()
    for s in starts:
        cur = s
        while cur and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            nxt = chain_map.get(cur)
            cur = nxt if nxt and nxt in group else None
    for s in group:
        if s not in seen:
            chain.append(s)
    spans: list[TickerSpan] = []
    for i, sym in enumerate(chain):
        end_date: date | None = None
        if i + 1 < len(chain):
            # We don't have a precise change date here; leave end open and let the
            # next span's start (also unknown) act as the implicit cutover.
            end_date = None
        d = delisted_by_sym.get(sym)
        if d and d.delisted_date:
            end_date = d.delisted_date
        spans.append(TickerSpan(ticker=sym, start=None, end=end_date))
    return spans


def known_sectors() -> list[str]:
    """Static fallback for offline `did you mean...` suggestions.

    The `miu list-sectors` command queries the live API; this list is
    used only when the API is unavailable.
    """
    return [
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ]
