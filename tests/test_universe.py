"""Universe construction: ticker chains, delisted inclusion, SP sweep, cache."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from miu.config import Settings
from miu.fmp.client import FmpClient
from miu.fmp.models import SymbolChange
from miu.universe import (
    Constituent,
    TickerSpan,
    UniverseRequest,
    _build_chain_dates,
    _build_chain_map,
    _build_constituents,
    build_universe,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key", cache_dir=tmp_path / "cache", base_url="https://example.test"
    )


def test_build_chain_map_resolves_chain() -> None:
    changes = [
        SymbolChange(date=date(2018, 1, 1), name="FB", oldSymbol="FB", newSymbol="META"),
        SymbolChange(date=date(2019, 1, 1), name="META", oldSymbol="META", newSymbol="META2"),
    ]
    chain = _build_chain_map(changes)
    assert chain["FB"] == "META2"
    assert chain["META"] == "META2"


def test_build_chain_map_breaks_cycles() -> None:
    changes = [
        SymbolChange(date=date(2018, 1, 1), name="A", oldSymbol="A", newSymbol="B"),
        SymbolChange(date=date(2019, 1, 1), name="B", oldSymbol="B", newSymbol="A"),
    ]
    chain = _build_chain_map(changes)
    # No infinite loop; both map to a terminal in the cycle.
    assert chain["A"] in {"A", "B"}
    assert chain["B"] in {"A", "B"}


def test_ticker_at_prefers_most_recent_overlapping_span() -> None:
    c = Constituent(
        entity_id="META",
        ticker_history=[
            TickerSpan(ticker="FB", start=None, end=None),
            TickerSpan(ticker="META", start=None, end=None),
        ],
    )
    assert c.ticker_at(date.today()) == "META"


def test_ticker_at_is_date_aware_when_spans_have_bounds() -> None:
    """After threading symbol-change dates, ticker_at(when) must return the
    symbol that was active on `when`, not the most recent one."""
    cutover = date(2022, 6, 9)
    c = Constituent(
        entity_id="META",
        ticker_history=[
            TickerSpan(ticker="FB", start=None, end=cutover),
            TickerSpan(ticker="META", start=cutover, end=None),
        ],
    )
    assert c.ticker_at(date(2020, 1, 1)) == "FB"
    assert c.ticker_at(date(2024, 1, 1)) == "META"


def test_build_constituents_assigns_real_span_bounds_from_chain_dates() -> None:
    """_make_history must consume chain_dates so spans get real (start, end)."""
    from miu.fmp.models import ScreenerRow

    changes = [
        SymbolChange(date=date(2022, 6, 9), name="FB", oldSymbol="FB", newSymbol="META"),
    ]
    chain_map = _build_chain_map(changes)
    chain_dates = _build_chain_dates(changes)
    current = [
        ScreenerRow(
            symbol="META",
            companyName="Meta Platforms",
            marketCap=1e12,
            sector="Technology",
            industry="Internet",
            exchangeShortName="NASDAQ",
            isActivelyTrading=True,
        )
    ]
    constituents = _build_constituents(
        matched={"FB", "META"},
        current=current,
        delisted=[],
        profiles={},
        chain_map=chain_map,
        chain_dates=chain_dates,
    )
    assert len(constituents) == 1
    c = constituents[0]
    by_ticker = {span.ticker: span for span in c.ticker_history}
    assert by_ticker["FB"].end == date(2022, 6, 9)
    assert by_ticker["META"].start == date(2022, 6, 9)
    assert c.ticker_at(date(2018, 1, 1)) == "FB"
    assert c.ticker_at(date(2023, 1, 1)) == "META"


@pytest.mark.asyncio
async def test_build_universe_dedupes_ticker_chain(tmp_path: Path) -> None:
    """A ticker that changed (FB → META) must collapse to a single Constituent."""
    base = "https://example.test"
    async with respx.mock(base_url=base, assert_all_called=False) as mock:
        # Screener returns current name only.
        mock.get("/company-screener").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "META",
                        "companyName": "Meta Platforms",
                        "marketCap": 1e12,
                        "sector": "Technology",
                        "industry": "Internet",
                        "exchangeShortName": "NASDAQ",
                        "isActivelyTrading": True,
                    }
                ],
            )
        )
        mock.get("/delisted-companies").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/symbol-change").mock(
            return_value=httpx.Response(
                200,
                json=[{"date": "2022-06-09", "name": "FB", "oldSymbol": "FB", "newSymbol": "META"}],
            )
        )
        mock.get("/historical-sp-500").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "dateAdded": "2013-12-23",
                        "addedSecurity": "Facebook Inc.",
                        "symbol": "FB",
                        "sector": "Information Technology",
                    }
                ],
            )
        )
        mock.get("/profile").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "FB",
                        "companyName": "Facebook (legacy)",
                        "sector": "Technology",
                        "industry": "Internet",
                        "exchangeShortName": "NASDAQ",
                        "country": "US",
                        "ipoDate": "2012-05-18",
                    }
                ],
            )
        )

        async with FmpClient(_settings(tmp_path)) as client:
            request = UniverseRequest(
                sector="Technology",
                industry=None,
                start=date(2018, 1, 1),
                end=date(2023, 12, 31),
                exchanges=("NYSE", "NASDAQ", "AMEX"),
                include_delisted=True,
            )
            universe = await build_universe(request, client)

    assert len(universe) == 1
    c = universe[0]
    assert c.entity_id == "META"
    tickers = {span.ticker for span in c.ticker_history}
    assert {"FB", "META"}.issubset(tickers)


@pytest.mark.asyncio
async def test_build_universe_includes_delisted_match(tmp_path: Path) -> None:
    """A delisted Health Care name (post-cutoff) must be included."""
    base = "https://example.test"
    async with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get("/company-screener").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/delisted-companies").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "DEAD",
                        "companyName": "Dead Co",
                        "exchange": "NASDAQ",
                        "ipoDate": "2005-01-01",
                        "delistedDate": "2021-09-01",
                    }
                ],
            )
        )
        mock.get("/symbol-change").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/historical-sp-500").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/profile").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "DEAD",
                        "sector": "Healthcare",
                        "industry": "Biotechnology",
                        "exchangeShortName": "NASDAQ",
                        "ipoDate": "2005-01-01",
                    }
                ],
            )
        )

        async with FmpClient(_settings(tmp_path)) as client:
            req = UniverseRequest(
                sector="Healthcare",
                industry=None,
                start=date(2020, 1, 1),
                end=date(2023, 12, 31),
                exchanges=("NYSE", "NASDAQ", "AMEX"),
                include_delisted=True,
            )
            universe = await build_universe(req, client)

    ids = {c.entity_id for c in universe}
    assert "DEAD" in ids
    dead = next(c for c in universe if c.entity_id == "DEAD")
    assert dead.delisting_date == date(2021, 9, 1)


@pytest.mark.asyncio
async def test_build_universe_cache_skips_network(tmp_path: Path) -> None:
    """Second build_universe call with same args must not hit the network."""
    base = "https://example.test"
    async with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get("/company-screener").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/delisted-companies").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/symbol-change").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/historical-sp-500").mock(return_value=httpx.Response(200, json=[]))

        async with FmpClient(_settings(tmp_path)) as client:
            req = UniverseRequest(
                sector="Healthcare",
                industry=None,
                start=date(2020, 1, 1),
                end=date(2023, 12, 31),
                exchanges=("NYSE", "NASDAQ", "AMEX"),
                include_delisted=True,
            )
            await build_universe(req, client)
            calls_after_first = sum(r.call_count for r in mock.routes)
            await build_universe(req, client)
            calls_after_second = sum(r.call_count for r in mock.routes)

    assert calls_after_first == calls_after_second, "cache should prevent re-hitting the API"
