"""Disk cache: TTL expiry and key determinism."""

from __future__ import annotations

import time

from miu.cache import TTL, DiskCache


def test_set_and_get_roundtrip(disk_cache: DiskCache) -> None:
    disk_cache.set("/foo", {"a": 1}, {"hello": "world"}, ttl=60)
    hit = disk_cache.get("/foo", {"a": 1})
    assert hit is not None
    assert hit.payload == {"hello": "world"}


def test_key_excludes_apikey(disk_cache: DiskCache) -> None:
    disk_cache.set("/foo", {"a": 1}, {"x": 1}, ttl=60)
    # Same logical params but with apikey added — should hit the same entry.
    hit = disk_cache.get("/foo", {"a": 1, "apikey": "different"})
    assert hit is not None
    assert hit.payload == {"x": 1}


def test_ttl_expiry(disk_cache: DiskCache) -> None:
    disk_cache.set("/foo", {}, {"x": 1}, ttl=0)
    time.sleep(0.01)
    assert disk_cache.get("/foo", {}) is None


def test_clear(disk_cache: DiskCache) -> None:
    disk_cache.set("/a", {}, {"x": 1}, ttl=60)
    disk_cache.set("/b", {}, {"y": 2}, ttl=60)
    n = disk_cache.clear()
    assert n >= 4  # body + meta per entry
    assert disk_cache.get("/a", {}) is None


def test_ttl_enum_values_match_spec() -> None:
    assert int(TTL.HISTORICAL) == 30 * 86400
    assert int(TTL.SNAPSHOT) == 1 * 86400
    assert int(TTL.REFERENCE) == 7 * 86400
