"""Configuration, settings, and shared error types.

Precedence (spec §9): CLI flags > env vars > ~/.miu/config.toml > defaults.
CLI flags don't live here; cli.py passes CLI values into Settings.load(...).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".miu" / "cache"
DEFAULT_CONFIG_FILE = Path.home() / ".miu" / "config.toml"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


class MiuError(Exception):
    """Base for all miu-raised errors."""


class MiuConfigError(MiuError):
    """Bad or missing configuration."""


class MiuApiError(MiuError):
    """FMP API call failed permanently.

    Carries enough context to debug without leaking the API key.
    """

    def __init__(self, endpoint: str, params: dict[str, Any], status: int | None, message: str):
        self.endpoint = endpoint
        self.params = _redact(params)
        self.status = status
        super().__init__(f"FMP {endpoint} (status={status}): {message} params={self.params}")


class MiuUniverseError(MiuError):
    """Universe construction failure."""


class MiuOptimizerError(MiuError):
    """Composite optimizer failed (infeasible bounds, non-convergence, etc.)."""


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    return {k: ("***" if k.lower() in {"apikey", "api_key"} else v) for k, v in params.items()}


@dataclass
class Settings:
    api_key: str | None
    cache_dir: Path
    base_url: str = FMP_BASE_URL
    verbose: bool = False
    log_format: str = "human"  # or "json"

    @classmethod
    def load(
        cls,
        *,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        config_file: Path | None = None,
        verbose: bool = False,
        log_format: str = "human",
    ) -> Settings:
        cfg = _read_config_file(config_file or DEFAULT_CONFIG_FILE)
        resolved_key = api_key or os.environ.get("FMP_API_KEY") or cfg.get("api_key")
        resolved_cache = cache_dir or _coerce_path(cfg.get("cache_dir")) or DEFAULT_CACHE_DIR
        resolved_base_url = (
            os.environ.get("FMP_BASE_URL") or cfg.get("base_url") or FMP_BASE_URL
        )
        return cls(
            api_key=resolved_key,
            cache_dir=resolved_cache,
            base_url=resolved_base_url,
            verbose=verbose,
            log_format=log_format,
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise MiuConfigError(
                "FMP API key is not set. Provide --api-key, export FMP_API_KEY=..., "
                "or add `api_key = \"...\"` to ~/.miu/config.toml."
            )
        return self.api_key


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MiuConfigError(f"Could not read config at {path}: {exc}") from exc
    section = data.get("miu", data)
    if not isinstance(section, dict):
        raise MiuConfigError(f"Config at {path} must be a TOML table")
    return section


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()
