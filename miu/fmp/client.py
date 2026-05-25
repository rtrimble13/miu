"""Async HTTP client for FMP with retries, rate-limit awareness, and disk cache."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from miu.cache import TTL, DiskCache
from miu.config import MiuApiError, Settings

log = logging.getLogger("miu.fmp")

_RATE_LIMIT_FLOOR = 5
_DEFAULT_CONCURRENCY = 8


class _RetryableHTTPError(Exception):
    """Internal signal: server-side or rate-limit error worth retrying."""


class FmpClient:
    """Async FMP API client.

    Use as: `async with FmpClient(settings) as client: ...`
    """

    def __init__(
        self,
        settings: Settings,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
        client: httpx.AsyncClient | None = None,
        cache: DiskCache | None = None,
    ):
        self.settings = settings
        self._sem = asyncio.Semaphore(concurrency)
        self._client = client
        self._owns_client = client is None
        self.cache = cache or DiskCache(settings.cache_dir)

    async def __aenter__(self) -> FmpClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"User-Agent": "miu/0.1"},
            )
            self._owns_client = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        ttl: int = TTL.HISTORICAL,
    ) -> Any:
        """Cached GET. Returns parsed JSON (typically list[dict] or dict)."""
        params = dict(params or {})
        cache_params = {k: v for k, v in params.items() if k.lower() != "apikey"}

        hit = self.cache.get(endpoint, cache_params)
        if hit is not None:
            log.debug("cache hit: %s %s", endpoint, cache_params)
            return hit.payload

        params["apikey"] = self.settings.require_api_key()
        try:
            payload = await self._fetch_with_retry(endpoint, params)
        except RetryError as exc:
            inner = exc.last_attempt.exception() if exc.last_attempt else None
            raise MiuApiError(endpoint, params, None, f"giving up after retries: {inner}") from exc

        self.cache.set(endpoint, cache_params, payload, ttl)
        return payload

    async def _fetch_with_retry(self, endpoint: str, params: dict[str, Any]) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_RetryableHTTPError),
            reraise=True,
        ):
            with attempt:
                return await self._fetch_once(endpoint, params)
        raise RuntimeError("unreachable")

    async def _fetch_once(self, endpoint: str, params: dict[str, Any]) -> Any:
        assert self._client is not None
        async with self._sem:
            log.debug("GET %s params=%s", endpoint, {k: v for k, v in params.items() if k != "apikey"})
            try:
                resp = await self._client.get(endpoint, params=params)
            except httpx.HTTPError as exc:
                raise _RetryableHTTPError(f"transport: {exc}") from exc

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                if int(remaining) <= _RATE_LIMIT_FLOOR:
                    log.warning("rate limit low (%s remaining), sleeping 2s", remaining)
                    await asyncio.sleep(2.0)
            except ValueError:
                pass

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            raise _RetryableHTTPError(f"status {resp.status_code}: {resp.text[:200]}")

        if resp.status_code >= 400:
            raise MiuApiError(
                endpoint, params, resp.status_code, resp.text[:500] or "HTTP error"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise MiuApiError(endpoint, params, resp.status_code, f"non-JSON body: {exc}") from exc

        # FMP occasionally returns HTTP 200 with a JSON error envelope rather
        # than the expected list/object. Surface it as a hard failure so the
        # endpoint parser doesn't silently swallow `[{"Error Message": ...}]`.
        if isinstance(body, dict) and "Error Message" in body:
            raise MiuApiError(endpoint, params, resp.status_code, str(body["Error Message"]))
        return body
