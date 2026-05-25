"""CLI tests for `miu recommend` (and shared validation logic)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import respx
from typer.testing import CliRunner

from miu.cli import app


def _write_index_csv(path: Path) -> Path:
    days = pd.bdate_range("2020-01-02", periods=200)
    df = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in days],
            "index_level": [1000.0 * (1 + 0.0005) ** i for i in range(len(days))],
            "daily_return": [0.0] + [0.0005] * (len(days) - 1),
            "n_constituents": [10] * len(days),
        }
    )
    df.to_csv(path, index=False)
    return path


def _price_series(days: pd.DatetimeIndex, base: float, drift: float) -> list[dict]:
    return [
        {"date": d.date().isoformat(), "adjClose": base * (1 + drift) ** i}
        for i, d in enumerate(days)
    ]


def test_recommend_rejects_mutually_exclusive_flags(tmp_path: Path) -> None:
    idx = _write_index_csv(tmp_path / "idx.csv")
    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "--index",
            str(idx),
            "--sector",
            "Energy",
            "--candidates",
            "XLE",
            "--output",
            str(tmp_path / "out.csv"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        env={"FMP_API_KEY": "test"},
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_recommend_requires_sector_or_index(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "--weighting",
            "equal",
            "--start",
            "2020-01-01",
            "--output",
            str(tmp_path / "out.csv"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        env={"FMP_API_KEY": "test"},
    )
    assert result.exit_code != 0


def test_recommend_with_index_file_and_user_candidates(tmp_path: Path) -> None:
    idx = _write_index_csv(tmp_path / "idx.csv")
    days = pd.bdate_range("2020-01-02", periods=200)
    base = "https://example.test"

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        # ETF prices: TIGHT mirrors the index, LOOSE is noisier.
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "TIGHT"}
        ).mock(return_value=httpx.Response(200, json=_price_series(days, 100.0, 0.0005)))
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "LOOSE"}
        ).mock(return_value=httpx.Response(200, json=_price_series(days, 50.0, 0.0010)))
        # ETF info endpoints — both fallback paths return empty so the code
        # exercises the final /profile fallback.
        for sym in ("TIGHT", "LOOSE"):
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
                            "sector": "Energy",
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
                "--index",
                str(idx),
                "--candidates",
                "TIGHT,LOOSE",
                "--min-overlap",
                "30",
                "--output",
                str(tmp_path / "recs.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--api-key",
                "test",
            ],
            env={"FMP_BASE_URL": base},
        )

    assert result.exit_code == 0, result.output
    out_csv = tmp_path / "recs.csv"
    out_meta = tmp_path / "recs_meta.json"
    assert out_csv.exists()
    assert out_meta.exists()
    df = pd.read_csv(out_csv)
    assert set(df["ticker"]) == {"TIGHT", "LOOSE"}
    winner_row = df[df["is_winner"]]
    assert len(winner_row) == 1
    assert winner_row.iloc[0]["ticker"] == "TIGHT"
    meta = json.loads(out_meta.read_text())
    assert meta["winner"] == "TIGHT"
    assert meta["candidate_source"] == "user-supplied"


def test_recommend_json_format(tmp_path: Path) -> None:
    idx = _write_index_csv(tmp_path / "idx.csv")
    days = pd.bdate_range("2020-01-02", periods=200)
    base = "https://example.test"

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.get(
            "/historical-price-eod/dividend-adjusted", params={"symbol": "XLE"}
        ).mock(return_value=httpx.Response(200, json=_price_series(days, 100.0, 0.0005)))
        mock.get("/etf/info", params={"symbol": "XLE"}).mock(
            return_value=httpx.Response(200, json=[])
        )
        mock.get("/etf-info", params={"symbol": "XLE"}).mock(
            return_value=httpx.Response(200, json=[])
        )
        mock.get("/profile", params={"symbol": "XLE"}).mock(
            return_value=httpx.Response(200, json=[{"symbol": "XLE", "isEtf": True}])
        )
        result = CliRunner().invoke(
            app,
            [
                "recommend",
                "--index",
                str(idx),
                "--candidates",
                "XLE",
                "--min-overlap",
                "30",
                "--format",
                "json",
                "--output",
                str(tmp_path / "recs.json"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--api-key",
                "test",
            ],
            env={"FMP_BASE_URL": base},
        )
    assert result.exit_code == 0, result.output
    body = json.loads((tmp_path / "recs.json").read_text())
    assert body["winner"] == "XLE"
    assert isinstance(body["table"], list)
    assert body["table"][0]["ticker"] == "XLE"


def test_validate_command_still_works_after_refactor(tmp_path: Path) -> None:
    """Sanity check that refactoring _build_async didn't break the wider flow."""
    # `miu validate` only requires --reference and constructs its own pipeline.
    # We just ensure the CLI doesn't crash on --help.
    result = CliRunner().invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
