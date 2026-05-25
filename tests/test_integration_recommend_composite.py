"""End-to-end integration: inline `miu build` + ETF discovery + composite fit.

This test extends the existing `test_integration_health_care.py` pattern to
exercise the full pipeline: build a tiny Healthcare index from FMP fixtures,
auto-discover Healthcare ETFs via `/company-screener?isEtf=true`, fit a
composite, and assert the composite TE is no worse than any single ETF's TE
(which must hold by construction since each unit vector is feasible).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import respx
from typer.testing import CliRunner

from miu.cli import app


def _settings_env(base: str) -> dict[str, str]:
    return {"FMP_BASE_URL": base, "FMP_API_KEY": "test"}


def _price_series(start: date, end: date, base: float, drift: float) -> list[dict]:
    days = pd.date_range(start, end, freq="B")
    return [
        {"date": d.date().isoformat(), "adjClose": base * (1 + drift) ** i}
        for i, d in enumerate(days)
    ]


def _mcap_series(start: date, end: date, base: float) -> list[dict]:
    days = pd.date_range(start, end, freq="B")
    return [{"date": d.date().isoformat(), "marketCap": base} for d in days]


def test_inline_recommend_discovers_and_ranks_etfs(tmp_path: Path) -> None:
    """Inline-mode recommend: build a Healthcare index, auto-discover XLV/IBB, rank."""
    start, end = date(2020, 1, 1), date(2020, 12, 31)
    base = "https://example.test"

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        # Equity universe (Healthcare) — two constituents, mcap-weighted.
        mock.get("/company-screener", params={"isEtf": "true"}).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "XLV",
                        "companyName": "Health Care SPDR",
                        "marketCap": 4e10,
                        "sector": "Healthcare",
                        "exchangeShortName": "NYSE",
                        "isEtf": True,
                    },
                    {
                        "symbol": "IBB",
                        "companyName": "iShares Biotech",
                        "marketCap": 1e10,
                        "sector": "Healthcare",
                        "exchangeShortName": "NASDAQ",
                        "isEtf": True,
                    },
                ],
            )
        )
        mock.get("/company-screener").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "symbol": "BIG",
                        "companyName": "Big Pharma",
                        "marketCap": 5e10,
                        "sector": "Healthcare",
                        "industry": "Pharmaceuticals",
                        "exchangeShortName": "NYSE",
                        "isActivelyTrading": True,
                    },
                    {
                        "symbol": "MED",
                        "companyName": "Mid Med",
                        "marketCap": 2e10,
                        "sector": "Healthcare",
                        "industry": "Medical Devices",
                        "exchangeShortName": "NASDAQ",
                        "isActivelyTrading": True,
                    },
                ],
            )
        )
        mock.get("/delisted-companies").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/symbol-change").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/historical-sp-500").mock(return_value=httpx.Response(200, json=[]))
        mock.get("/mergers-acquisitions").mock(return_value=httpx.Response(200, json=[]))

        # Constituent + ETF prices: give XLV identical drift to BIG/MED-blended
        # so the recommended ETF is XLV; IBB drifts differently so its TE is larger.
        for sym, base_p, drift in (
            ("BIG", 100.0, 0.0005),
            ("MED", 50.0, 0.0005),
            ("XLV", 100.0, 0.0005),
            ("IBB", 80.0, 0.0010),
        ):
            mock.get(
                "/historical-price-eod/dividend-adjusted",
                params={"symbol": sym},
            ).mock(
                return_value=httpx.Response(
                    200, json=_price_series(start, end, base_p, drift)
                )
            )

        for sym, mcap in (("BIG", 5e10), ("MED", 2e10)):
            mock.get(
                "/historical-market-capitalization",
                params={"symbol": sym},
            ).mock(return_value=httpx.Response(200, json=_mcap_series(start, end, mcap)))

        # ETF info: all fallbacks 200/empty so the implicit /profile fallback engages.
        for sym in ("XLV", "IBB"):
            mock.get("/etf/info", params={"symbol": sym}).mock(
                return_value=httpx.Response(200, json=[])
            )
            mock.get("/etf-info", params={"symbol": sym}).mock(
                return_value=httpx.Response(200, json=[])
            )
            mock.get("/profile", params={"symbol": sym}).mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {
                            "symbol": sym,
                            "companyName": f"{sym} ETF",
                            "sector": "Healthcare",
                            "exchangeShortName": "NYSE",
                            "isEtf": True,
                        }
                    ],
                )
            )

        result = CliRunner().invoke(
            app,
            [
                "recommend",
                "--sector",
                "Healthcare",
                "--weighting",
                "market-cap",
                "--start",
                start.isoformat(),
                "--end",
                end.isoformat(),
                "--rebalance",
                "quarterly",
                "--min-overlap",
                "30",
                "--format",
                "json",
                "--output",
                str(tmp_path / "recs.json"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ],
            env=_settings_env(base),
        )

    assert result.exit_code == 0, result.output
    body = json.loads((tmp_path / "recs.json").read_text())
    assert body["winner"] == "XLV"
    by_ticker = {row["ticker"]: row for row in body["table"]}
    assert by_ticker["XLV"]["te"] < by_ticker["IBB"]["te"]


def test_composite_with_user_candidates_beats_or_matches_single_te(tmp_path: Path) -> None:
    """Composite annualized TE must be ≤ the best single-candidate TE."""
    rng = np.random.default_rng(13)
    n = 250
    days = pd.bdate_range("2020-01-02", periods=n + 1)
    a_rets = rng.normal(0.0005, 0.012, size=n).tolist()
    b_rets = rng.normal(0.0004, 0.015, size=n).tolist()
    idx_rets = [0.7 * a + 0.3 * b for a, b in zip(a_rets, b_rets, strict=True)]

    levels = [1000.0]
    for r in idx_rets:
        levels.append(levels[-1] * (1 + r))
    idx_path = tmp_path / "idx.csv"
    pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in days],
            "index_level": levels,
            "daily_return": [0.0] + idx_rets,
            "n_constituents": [2] * len(days),
        }
    ).to_csv(idx_path, index=False)

    base = "https://example.test"

    def price_series(base_p: float, returns: list[float]) -> list[dict]:
        prices = [base_p]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return [
            {"date": d.date().isoformat(), "adjClose": p}
            for d, p in zip(days, prices, strict=True)
        ]

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "A"}
        ).mock(return_value=httpx.Response(200, json=price_series(100.0, a_rets)))
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "B"}
        ).mock(return_value=httpx.Response(200, json=price_series(50.0, b_rets)))
        for sym in ("A", "B"):
            mock.get("/etf/info", params={"symbol": sym}).mock(
                return_value=httpx.Response(200, json=[])
            )
            mock.get("/etf-info", params={"symbol": sym}).mock(
                return_value=httpx.Response(200, json=[])
            )
            mock.get("/profile", params={"symbol": sym}).mock(
                return_value=httpx.Response(200, json=[{"symbol": sym, "isEtf": True}])
            )

        result = CliRunner().invoke(
            app,
            [
                "composite",
                "--index",
                str(idx_path),
                "--candidates",
                "A,B",
                "--top-k",
                "2",
                "--min-overlap",
                "30",
                "--output",
                str(tmp_path / "comp.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ],
            env=_settings_env(base),
        )

    assert result.exit_code == 0, result.output
    meta = json.loads((tmp_path / "comp_meta.json").read_text())
    composite_te = meta["fit"]["te"]
    # Compare to the worst single-candidate TE (we know A's mix is closer to the index).
    a_arr = np.array(a_rets)
    b_arr = np.array(b_rets)
    y = np.array(idx_rets)
    a_te = float((a_arr - y).std(ddof=0) * math.sqrt(252))
    b_te = float((b_arr - y).std(ddof=0) * math.sqrt(252))
    assert composite_te <= min(a_te, b_te) + 1e-9
