"""Config precedence and error types."""

from __future__ import annotations

from pathlib import Path

import pytest

from miu.config import MiuApiError, MiuConfigError, Settings, _redact


def test_settings_uses_env_when_no_cli(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FMP_API_KEY", "env-key")
    s = Settings.load(api_key=None, cache_dir=tmp_path, config_file=tmp_path / "missing.toml")
    assert s.api_key == "env-key"


def test_settings_cli_beats_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FMP_API_KEY", "env-key")
    s = Settings.load(api_key="cli-key", cache_dir=tmp_path)
    assert s.api_key == "cli-key"


def test_settings_reads_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('api_key = "file-key"\ncache_dir = "/tmp/elsewhere"\n')
    s = Settings.load(cache_dir=None, config_file=cfg)
    assert s.api_key == "file-key"
    assert str(s.cache_dir) == "/tmp/elsewhere"


def test_require_api_key_raises_without_key(tmp_path: Path) -> None:
    s = Settings(api_key=None, cache_dir=tmp_path)
    with pytest.raises(MiuConfigError, match="FMP API key"):
        s.require_api_key()


def test_redact_hides_api_key() -> None:
    assert _redact({"apikey": "secret", "sym": "AAPL"}) == {"apikey": "***", "sym": "AAPL"}


def test_miu_api_error_redacts() -> None:
    exc = MiuApiError("/foo", {"apikey": "secret", "sym": "AAPL"}, 500, "boom")
    assert "secret" not in str(exc)
    assert "AAPL" in str(exc)
