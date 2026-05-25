"""End-to-end Health Care index build using recorded respx fixtures.

Builds a small synthetic universe with one current name, one delisted name,
and one ticker change. Confirms the full pipeline (universe → engine → output)
produces a well-formed result that obeys spec invariants.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from miu.config import Settings
from miu.fmp import endpoints as ep
from miu.fmp.client import FmpClient
from miu.index import EngineConfig, IndexEngine
from miu.output import write_csv, write_json
from miu.universe import UniverseRequest, build_universe


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        cache_dir=tmp_path / "cache",
        base_url="https://example.test",
    )


def _price_series(start: date, end: date, base: float, drift: float) -> list[dict]:
    days = pd.date_range(start, end, freq="B")
    return [
        {"date": d.date().isoformat(), "adjClose": base * (1 + drift) ** i}
        for i, d in enumerate(days)
    ]


def _mcap_series(start: date, end: date, base: float) -> list[dict]:
    days = pd.date_range(start, end, freq="B")
    return [{"date": d.date().isoformat(), "marketCap": base} for d in days]


@pytest.mark.asyncio
async def test_end_to_end_health_care_pipeline(tmp_path: Path) -> None:
    start, end = date(2020, 1, 1), date(2020, 12, 31)
    base = "https://example.test"

    async with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get("/company-screener").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "BIG",
                        "companyName": "Big Pharma Inc",
                        "marketCap": 5e10,
                        "sector": "Healthcare",
                        "industry": "Pharmaceuticals",
                        "exchangeShortName": "NYSE",
                        "isActivelyTrading": True,
                    },
                    {
                        "symbol": "MED",
                        "companyName": "Mid Med Inc",
                        "marketCap": 2e10,
                        "sector": "Healthcare",
                        "industry": "Medical Devices",
                        "exchangeShortName": "NASDAQ",
                        "isActivelyTrading": True,
                    },
                ],
            )
        )
        mock.get("/delisted-companies").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "GONE",
                        "companyName": "Gone Pharma",
                        "exchange": "NASDAQ",
                        "ipoDate": "2010-05-01",
                        "delistedDate": "2020-08-15",
                    }
                ],
            )
        )
        mock.get("/symbol-change").mock(
            return_value=httpx.Response(
                200,
                json=[{"date": "2020-03-01", "name": "MED", "oldSymbol": "OLDMED", "newSymbol": "MED"}],
            )
        )
        mock.get("/historical-sp-500").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/profile", params={"symbol": "GONE"}).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "GONE",
                        "sector": "Healthcare",
                        "industry": "Pharmaceuticals",
                        "exchangeShortName": "NASDAQ",
                        "ipoDate": "2010-05-01",
                    }
                ],
            )
        )
        # Price/mcap routes for each ticker the universe will request.
        for sym, base_price in (("BIG", 100.0), ("MED", 50.0), ("GONE", 25.0), ("OLDMED", 50.0)):
            mock.get(
                "/historical-price-eod/dividend-adjusted",
                params={"symbol": sym},
            ).mock(
                return_value=httpx.Response(
                    200, json=_price_series(start, end, base_price, drift=0.0005)
                )
            )
            mock.get(
                "/historical-market-capitalization",
                params={"symbol": sym},
            ).mock(
                return_value=httpx.Response(
                    200, json=_mcap_series(start, end, 5e10 if sym == "BIG" else 2e10)
                )
            )

        async with FmpClient(_settings(tmp_path)) as client:
            request = UniverseRequest(
                sector="Healthcare",
                industry=None,
                start=start,
                end=end,
                exchanges=("NYSE", "NASDAQ", "AMEX"),
                include_delisted=True,
            )
            constituents = await build_universe(request, client)
            assert constituents

            prices: dict[str, dict[date, float]] = {}
            mcaps: dict[str, dict[date, float]] = {}
            for c in constituents:
                for span in c.ticker_history:
                    rows = await ep.historical_prices(client, span.ticker, start=start, end=end)
                    for r in rows:
                        if r.price is not None:
                            prices.setdefault(c.entity_id, {}).setdefault(r.date, r.price)
                    mrows = await ep.historical_market_cap(
                        client, span.ticker, start=start, end=end
                    )
                    for m in mrows:
                        mcaps.setdefault(c.entity_id, {}).setdefault(m.date, m.market_cap)

            engine = IndexEngine(
                constituents,
                prices,
                mcaps,
                EngineConfig(
                    weighting="market-cap",
                    start=start,
                    end=end,
                    rebalance="quarterly",
                    base_value=1000.0,
                    sector="Healthcare",
                ),
            )
            result = engine.run()

    # Spec invariants.
    assert not result.series.empty
    assert result.series["index_level"].iloc[0] == 1000.0
    assert result.series["n_constituents"].max() >= 1
    # Weights at each rebalance date should sum to ~1.0.
    rebal = result.constituents[result.constituents["is_rebalance_date"]]
    for _, grp in rebal.groupby("date"):
        assert abs(grp["weight"].sum() - 1.0) < 1e-6

    # Output writers produce the right files.
    series_path, const_path = write_csv(result, tmp_path / "hc.csv")
    assert series_path.exists()
    assert const_path.exists()
    json_path = write_json(
        result,
        tmp_path / "hc.json",
        {
            "sector": "Healthcare",
            "industry": None,
            "weighting": "market-cap",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "rebalance": "quarterly",
            "base_value": 1000.0,
        },
    )
    assert json_path.exists()
