"""CLI smoke tests via typer's CliRunner."""

from __future__ import annotations

import inspect
import json
import math
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from miu.cli import _resolve_mna, app, build
from miu.fmp.models import MnaEvent
from miu.universe import Constituent, TickerSpan


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "miu " in result.output


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("build", "list-sectors", "validate", "cache"):
        assert cmd in result.output


def test_build_requires_sector_or_industry(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "build",
            "--weighting",
            "equal",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
            "--output",
            str(tmp_path / "x.csv"),
        ],
        env={"FMP_API_KEY": "test"},
    )
    assert result.exit_code != 0


def test_build_rejects_both_sector_and_industry(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "build",
            "--sector",
            "Healthcare",
            "--industry",
            "Pharma",
            "--weighting",
            "equal",
            "--start",
            "2020-01-01",
            "--output",
            str(tmp_path / "x.csv"),
        ],
        env={"FMP_API_KEY": "test"},
    )
    assert result.exit_code != 0


def test_build_missing_api_key(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "build",
            "--sector",
            "Healthcare",
            "--weighting",
            "equal",
            "--start",
            "2020-01-01",
            "--end",
            "2020-06-30",
            "--output",
            str(tmp_path / "x.csv"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        env={"FMP_API_KEY": ""},
    )
    assert result.exit_code == 2
    assert "FMP API key" in result.output


def test_cache_info(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    result = CliRunner().invoke(app, ["cache", "info", "--cache-dir", str(cache)])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["cache_dir"] == str(cache)
    assert body["entries"] == 0


def test_cache_clear(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "bucket").mkdir()
    (cache / "bucket" / "a.json.gz").write_bytes(b"x")
    result = CliRunner().invoke(app, ["cache", "clear", "--cache-dir", str(cache)])
    assert result.exit_code == 0
    assert "removed" in result.output


def test_build_cache_dir_flag_defaults_to_none() -> None:
    sig = inspect.signature(build)
    assert sig.parameters["cache_dir"].default.default is None


def _const(eid: str) -> Constituent:
    return Constituent(
        entity_id=eid,
        ticker_history=[TickerSpan(ticker=eid, start=None, end=None)],
        ipo_date=date(2000, 1, 1),
        sector="Healthcare",
    )


def test_resolve_mna_builds_stock_deal_resolution() -> None:
    """Stock-for-stock deal where the acquirer is in the universe: ratio is
    p_target/p_acquirer at the transaction date."""
    constituents = [_const("ACQ"), _const("TGT")]
    prices = {
        "TGT": {date(2022, 1, 5): 100.0},
        "ACQ": {date(2022, 1, 5): 200.0},
    }
    events = [
        MnaEvent(
            symbol="ACQ",
            targetedSymbol="TGT",
            targetedCompanyName="Target Co",
            transactionDate="2022-01-05",
            consideration="stock",
            dealType="acquisition",
        )
    ]
    out = _resolve_mna(events, constituents, prices)
    assert "TGT" in out
    assert out["TGT"].acquirer_id == "ACQ"
    assert math.isclose(out["TGT"].ratio, 0.5, rel_tol=1e-9)
    assert out["TGT"].cash_value is None


def test_resolve_mna_skips_cash_deals_and_unknown_acquirers() -> None:
    """Cash deals carry no value from FMP; deals with unknown acquirers
    cannot redistribute. Both should be left out of the resolution dict."""
    constituents = [_const("TGT")]
    prices = {"TGT": {date(2022, 1, 5): 100.0}}
    events = [
        MnaEvent(  # cash deal → skip
            symbol="ACQ",
            targetedSymbol="TGT",
            transactionDate="2022-01-05",
            consideration="cash",
            dealType="merger",
        ),
        MnaEvent(  # stock deal, acquirer not in universe → skip
            symbol="UNKNOWN",
            targetedSymbol="TGT",
            transactionDate="2022-01-05",
            consideration="stock",
            dealType="merger",
        ),
    ]
    out = _resolve_mna(events, constituents, prices)
    assert out == {}
