"""FMP async client: retries, rate-limit handling, redacted errors."""

from __future__ import annotations

import httpx
import pytest
import respx

from miu.cache import TTL
from miu.config import MiuApiError, Settings
from miu.fmp.client import FmpClient


@pytest.fixture
def client_settings(tmp_path) -> Settings:
    return Settings(api_key="test-key", cache_dir=tmp_path / "cache", base_url="https://example.test")


@pytest.mark.asyncio
async def test_get_returns_json_and_caches(client_settings: Settings) -> None:
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        route = mock.get("/foo").mock(return_value=httpx.Response(200, json=[{"x": 1}]))
        async with FmpClient(client_settings) as client:
            payload = await client.get("/foo", {"a": 1}, ttl=int(TTL.HISTORICAL))
            assert payload == [{"x": 1}]
            # Second call should hit the cache and NOT call the network again.
            payload2 = await client.get("/foo", {"a": 1}, ttl=int(TTL.HISTORICAL))
            assert payload2 == [{"x": 1}]
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_get_retries_on_500(client_settings: Settings) -> None:
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        route = mock.get("/flaky").mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(500, text="boom"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        async with FmpClient(client_settings) as client:
            payload = await client.get("/flaky", {}, ttl=int(TTL.SNAPSHOT))
        assert payload == {"ok": True}
        assert route.call_count == 3


@pytest.mark.asyncio
async def test_get_retries_on_429(client_settings: Settings) -> None:
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        route = mock.get("/limited").mock(
            side_effect=[
                httpx.Response(429, text="slow down"),
                httpx.Response(200, json=[]),
            ]
        )
        async with FmpClient(client_settings) as client:
            payload = await client.get("/limited", {}, ttl=int(TTL.SNAPSHOT))
        assert payload == []
        assert route.call_count == 2


@pytest.mark.asyncio
async def test_get_redacts_api_key_in_error(client_settings: Settings) -> None:
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        mock.get("/bad").mock(return_value=httpx.Response(404, text="missing"))
        async with FmpClient(client_settings) as client:
            with pytest.raises(MiuApiError) as exc_info:
                await client.get("/bad", {"sym": "AAPL"}, ttl=int(TTL.SNAPSHOT))
        msg = str(exc_info.value)
        assert "test-key" not in msg
        assert "***" in msg
        assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_get_raises_on_error_message_envelope(client_settings: Settings) -> None:
    """FMP sometimes returns HTTP 200 with `{"Error Message": ...}`; surface as failure."""
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        mock.get("/bogus").mock(
            return_value=httpx.Response(200, json={"Error Message": "Invalid API KEY."})
        )
        async with FmpClient(client_settings) as client:
            with pytest.raises(MiuApiError) as exc_info:
                await client.get("/bogus", {}, ttl=int(TTL.SNAPSHOT))
    assert "Invalid API KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_rate_limit_header_triggers_sleep(monkeypatch, client_settings: Settings) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("miu.fmp.client.asyncio.sleep", fake_sleep)
    async with respx.mock(base_url="https://example.test", assert_all_called=False) as mock:
        mock.get("/p").mock(
            return_value=httpx.Response(200, json=[], headers={"X-RateLimit-Remaining": "2"})
        )
        async with FmpClient(client_settings) as client:
            await client.get("/p", {}, ttl=int(TTL.SNAPSHOT))
    assert any(s >= 1.0 for s in sleeps), "expected a low-rate-limit sleep"
