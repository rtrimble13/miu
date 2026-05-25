"""CLI tests for `miu composite`."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import respx
from typer.testing import CliRunner

from miu.cli import app


def _write_index_csv(path: Path, n: int = 250) -> Path:
    days = pd.bdate_range("2020-01-02", periods=n)
    # Use a non-trivial daily_return so the composite optimization has signal.
    import numpy as np
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=n)
    levels = 1000.0 * (1 + rets).cumprod()
    df = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in days],
            "index_level": levels,
            "daily_return": rets,
            "n_constituents": [10] * n,
        }
    )
    df.to_csv(path, index=False)
    return path


def _price_series_from_returns(
    days: pd.DatetimeIndex, base: float, returns: list[float]
) -> list[dict]:
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return [
        {"date": d.date().isoformat(), "adjClose": p}
        for d, p in zip(days, prices, strict=True)
    ]


def test_composite_rejects_bad_weight_bounds(tmp_path: Path) -> None:
    idx = _write_index_csv(tmp_path / "idx.csv")
    result = CliRunner().invoke(
        app,
        [
            "composite",
            "--index",
            str(idx),
            "--candidates",
            "XLE,VDE",
            "--min-weight",
            "0.6",
            "--max-weight",
            "0.5",
            "--output",
            str(tmp_path / "out.csv"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--api-key",
            "test",
        ],
    )
    assert result.exit_code != 0


def test_composite_recovers_known_mix(tmp_path: Path) -> None:
    """End-to-end: target = 60% TIGHT + 40% LOOSE. The composite should recover that."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 250
    days = pd.bdate_range("2020-01-02", periods=n + 1)  # +1 because returns->prices
    tight_rets = rng.normal(0.0005, 0.011, size=n).tolist()
    loose_rets = rng.normal(0.0003, 0.014, size=n).tolist()
    idx_rets = [0.6 * a + 0.4 * b for a, b in zip(tight_rets, loose_rets, strict=True)]

    # Write the index CSV from the synthesized daily returns.
    idx_levels = [1000.0]
    for r in idx_rets:
        idx_levels.append(idx_levels[-1] * (1 + r))
    idx_path = tmp_path / "idx.csv"
    pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in days],
            "index_level": idx_levels,
            "daily_return": [0.0] + idx_rets,
            "n_constituents": [3] * len(days),
        }
    ).to_csv(idx_path, index=False)

    base = "https://example.test"
    with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "TIGHT"}
        ).mock(
            return_value=httpx.Response(
                200, json=_price_series_from_returns(days, 100.0, tight_rets)
            )
        )
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "LOOSE"}
        ).mock(
            return_value=httpx.Response(
                200, json=_price_series_from_returns(days, 50.0, loose_rets)
            )
        )
        for sym in ("TIGHT", "LOOSE"):
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
                "TIGHT,LOOSE",
                "--top-k",
                "2",
                "--min-overlap",
                "30",
                "--output",
                str(tmp_path / "comp.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--api-key",
                "test",
            ],
            env={"FMP_BASE_URL": base},
        )

    assert result.exit_code == 0, result.output
    comp_csv = tmp_path / "comp.csv"
    meta_path = tmp_path / "comp_meta.json"
    assert comp_csv.exists()
    assert meta_path.exists()
    weights = pd.read_csv(comp_csv)
    by_ticker = dict(zip(weights["ticker"], weights["weight"], strict=True))
    assert by_ticker["TIGHT"] == pytest.approx(0.6, abs=1e-2)
    assert by_ticker["LOOSE"] == pytest.approx(0.4, abs=1e-2)
    assert sum(by_ticker.values()) == pytest.approx(1.0, abs=1e-9)

    meta = json.loads(meta_path.read_text())
    assert meta["fit"]["converged"] is True
    assert meta["fit"]["te_bps"] < 100.0  # near-perfect fit on synthetic data


# pytest is imported lazily via the assertion above
import pytest  # noqa: E402
