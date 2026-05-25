"""On-disk TTL cache for FMP responses.

Keyed on sha256(endpoint + sorted params). Three TTL tiers (spec §3):
historical = 30d, snapshot = 1d, reference = 7d. Stored gzipped JSON
with a sidecar `.meta.json` containing fetched_at + ttl_seconds.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class TTL(int, Enum):
    HISTORICAL = 30 * 24 * 3600
    SNAPSHOT = 1 * 24 * 3600
    REFERENCE = 7 * 24 * 3600


@dataclass
class CacheEntry:
    payload: Any
    fetched_at: float
    ttl_seconds: int

    @property
    def expires_at(self) -> float:
        return self.fetched_at + self.ttl_seconds

    def is_fresh(self, now: float | None = None) -> bool:
        return (now or time.time()) < self.expires_at


class DiskCache:
    """Filesystem cache for serialized FMP responses.

    Layout: <root>/<endpoint_slug>/<hash>.json.gz + <hash>.meta.json
    The sidecar lets us check freshness without inflating the payload.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _slug(self, endpoint: str) -> str:
        return endpoint.strip("/").replace("/", "_") or "root"

    def _key(self, endpoint: str, params: dict[str, Any]) -> str:
        normalized = {k: params[k] for k in sorted(params) if k.lower() != "apikey"}
        blob = json.dumps([endpoint, normalized], separators=(",", ":"), default=str, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _paths(self, endpoint: str, params: dict[str, Any]) -> tuple[Path, Path]:
        bucket = self.root / self._slug(endpoint)
        bucket.mkdir(parents=True, exist_ok=True)
        key = self._key(endpoint, params)
        return bucket / f"{key}.json.gz", bucket / f"{key}.meta.json"

    def get(self, endpoint: str, params: dict[str, Any]) -> CacheEntry | None:
        body_path, meta_path = self._paths(endpoint, params)
        if not body_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            entry = CacheEntry(
                payload=None,
                fetched_at=float(meta["fetched_at"]),
                ttl_seconds=int(meta["ttl_seconds"]),
            )
            if not entry.is_fresh():
                return None
            with gzip.open(body_path, "rt") as fh:
                entry.payload = json.load(fh)
            return entry
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def set(self, endpoint: str, params: dict[str, Any], payload: Any, ttl: int) -> None:
        body_path, meta_path = self._paths(endpoint, params)
        # Write body + meta to sibling .tmp files first, then atomically
        # rename so a crash mid-write cannot leave a half-written entry.
        body_tmp = body_path.with_suffix(body_path.suffix + ".tmp")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with gzip.open(body_tmp, "wt") as fh:
            json.dump(payload, fh, default=str)
        meta_tmp.write_text(
            json.dumps({"fetched_at": time.time(), "ttl_seconds": int(ttl)}, separators=(",", ":"))
        )
        os.replace(body_tmp, body_path)
        os.replace(meta_tmp, meta_path)

    def clear(self) -> int:
        n = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                p.unlink()
                n += 1
        return n

    def stats(self) -> dict[str, int]:
        files = list(self.root.rglob("*.json.gz"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {"entries": len(files), "bytes": total_bytes}
