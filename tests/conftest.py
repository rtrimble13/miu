"""Shared test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from miu.cache import DiskCache
from miu.config import Settings


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def settings(tmp_cache_dir: Path) -> Settings:
    return Settings(api_key="test-key", cache_dir=tmp_cache_dir, base_url="https://example.test")


@pytest.fixture
def disk_cache(tmp_cache_dir: Path) -> DiskCache:
    return DiskCache(tmp_cache_dir)


@pytest.fixture
def sample_dates() -> list[date]:
    """A small calendar — 10 business days in Jan 2020."""
    import pandas as pd

    return [d.date() for d in pd.date_range("2020-01-02", periods=10, freq="B")]
