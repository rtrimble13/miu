"""CLI smoke tests via typer's CliRunner."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from typer.testing import CliRunner

from miu.cli import app, build


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
