from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import baystfirm.candles as candle_module
from baystfirm.candles import (
    CandleNotFound,
    CandleService,
    aggregate_candles,
    fetch_source_candles,
    source_interval,
)
from baystfirm.config import Settings
from baystfirm.models import Candle
from baystfirm.storage import EventStore

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def _settings(tmp_path: Path, venue: str, symbol: str) -> Settings:
    return Settings(
        database_path=tmp_path / "candles.db",
        enabled_venues=(venue,),
        shadow_mode=True,
        symbols=(symbol,),
        solana_tokens_enabled=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("venue", "symbol", "fixture", "path"),
    [
        ("coinbase", "BTC-USD", "coinbase_candles.json", "/products/BTC-USD/candles"),
        ("kraken", "BTC-USD", "kraken_candles.json", "/0/public/OHLC"),
        ("bybit", "BTC-USDT", "bybit_candles.json", "/v5/market/kline"),
        ("okx", "BTC-USDT", "okx_candles.json", "/api/v5/market/candles"),
        ("binanceus", "BTC-USDT", "binanceus_candles.json", "/api/v3/klines"),
    ],
)
async def test_public_candle_responses_are_normalized_and_cached(
    tmp_path: Path, venue: str, symbol: str, fixture: str, path: str
) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    body = _fixture(fixture)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body)

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    result = await CandleService(
        _settings(tmp_path, venue, symbol), store, client_factory=client_factory
    ).get(venue, symbol, "1m", 2)

    assert len(requests) == 1
    assert requests[0].url.path == path
    assert result["venue"] == venue
    assert result["symbol"] == symbol
    assert result["interval"] == "1m"
    assert result["source_url_template"]
    assert result["stale"] is False
    candles = result["candles"]
    assert len(candles) == 2
    assert candles == sorted(candles, key=lambda item: item["open_time"])
    assert candles[0]["open_time"] < candles[1]["open_time"]
    cached, fetched_at = await store.candles(venue, symbol, "1m", 2)
    assert len(cached) == 2
    assert fetched_at is not None
    await store.close()


@pytest.mark.asyncio
async def test_okx_5000_candles_paginate_from_latest_to_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candle_module, "OKX_HISTORY_PAGE_DELAY_SECONDS", 0)
    requests: list[httpx.Request] = []
    next_index = 4999
    last_oldest: int | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal next_index, last_oldest
        requests.append(request)
        page = len(requests)
        if page == 1:
            assert request.url.path == "/api/v5/market/candles"
            assert "after" not in request.url.params
        else:
            assert request.url.path == "/api/v5/market/history-candles"
            assert request.url.params["after"] == str(last_oldest)

        limit = int(request.url.params["limit"])
        rows = []
        for _ in range(limit):
            timestamp = 1_700_000_000_000 + next_index * 3_600_000
            rows.append([str(timestamp), "100", "101", "99", "100", "1", "1", "1", "1"])
            next_index -= 1
        last_oldest = min(int(row[0]) for row in rows)
        return httpx.Response(200, json={"code": "0", "msg": "", "data": rows})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candles = await fetch_source_candles(client, "okx", "BTC-USDT", "1h", 5000)

    assert len(candles) == 5000
    assert len({candle.open_time for candle in candles}) == 5000
    assert candles == sorted(candles, key=lambda candle: candle.open_time)
    assert requests[0].url.path == "/api/v5/market/candles"
    assert all(request.url.path == "/api/v5/market/history-candles" for request in requests[1:])
    assert requests[0].url.params["limit"] == "300"
    assert requests[-1].url.params["limit"] == "200"
    assert len(requests) == 17


@pytest.mark.asyncio
async def test_okx_empty_history_page_stops_pagination() -> None:
    requests: list[httpx.Request] = []
    rows = [
        ["1700007200000", "100", "101", "99", "100", "1", "1", "1", "1"],
        ["1700003600000", "100", "101", "99", "100", "1", "1", "1", "1"],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert request.url.path == "/api/v5/market/candles"
            return httpx.Response(200, json={"code": "0", "msg": "", "data": rows})
        assert request.url.path == "/api/v5/market/history-candles"
        assert request.url.params["after"] == "1700003600000"
        return httpx.Response(200, json={"code": "0", "msg": "", "data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candles = await fetch_source_candles(client, "okx", "BTC-USDT", "1h", 5000)

    assert len(requests) == 2
    assert [candle.open_time for candle in candles] == [1_700_003_600_000, 1_700_007_200_000]


@pytest.mark.asyncio
async def test_fresh_candle_cache_skips_upstream_request(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_fixture("coinbase_candles.json"))

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    service = CandleService(
        _settings(tmp_path, "coinbase", "BTC-USD"),
        store,
        client_factory=client_factory,
    )
    await service.get("coinbase", "BTC-USD", "1m", 2)
    await service.get("coinbase", "BTC-USD", "1m", 2)

    assert calls == 1
    await store.close()


@pytest.mark.asyncio
async def test_unknown_pairs_and_spot_venue_perpetuals_are_not_found(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    settings = Settings(
        database_path=tmp_path / "candles.db",
        enabled_venues=("coinbase", "binanceus"),
        shadow_mode=True,
        symbols=("BTC-USD", "BTC-USDT-PERP"),
        solana_tokens_enabled=False,
    )
    service = CandleService(settings, store)

    with pytest.raises(CandleNotFound):
        await service.get("unknown", "BTC-USD", "1m", 2)
    with pytest.raises(CandleNotFound):
        await service.get("coinbase", "ETH-USD", "1m", 2)
    with pytest.raises(CandleNotFound):
        await service.get("binanceus", "BTC-USDT-PERP", "1m", 2)

    await store.close()


@pytest.mark.asyncio
async def test_exchange_unknown_symbol_response_is_not_found(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(404, json={"message": "NotFound"})
            ),
            **kwargs,
        )

    service = CandleService(
        _settings(tmp_path, "coinbase", "BTC-USD"),
        store,
        client_factory=client_factory,
    )
    with pytest.raises(CandleNotFound, match="does not list this symbol"):
        await service.get("coinbase", "BTC-USD", "1m", 2)

    await store.close()


def test_native_interval_selection_uses_largest_divisor() -> None:
    assert source_interval("coinbase", "4h") == "1h"
    assert source_interval("kraken", "6h") == "1h"
    assert source_interval("kraken", "12h") == "4h"
    assert source_interval("kraken", "1w") == "1d"
    assert source_interval("coinbase", "1w") == "1d"


@pytest.mark.asyncio
async def test_kraken_12h_aggregates_from_four_hour_candles(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
    rows = [
        [
            start + index * 4 * 3600,
            str(100 + index),
            str(110 + index),
            str(90 + index),
            str(101 + index),
            "100",
            str(index + 1),
            3,
        ]
        for index in range(3)
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"error": [], "result": {"XXBTZUSD": rows, "last": rows[-1][0]}},
        )

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    result = await CandleService(
        _settings(tmp_path, "kraken", "BTC-USD"),
        store,
        client_factory=client_factory,
    ).get("kraken", "BTC-USD", "12h", 1)

    assert requests[0].url.params["interval"] == "240"
    assert result["aggregated_from"] == "4h"
    assert result["candles"] == [
        {
            "open_time": start * 1000,
            "open": 100.0,
            "high": 112.0,
            "low": 90.0,
            "close": 103.0,
            "volume": 6.0,
        }
    ]
    await store.close()


def test_aggregates_1h_candles_into_4h() -> None:
    start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    candles = [
        Candle(
            open_time=start + index * 3_600_000,
            open=100 + index,
            high=105 + index,
            low=95 + index,
            close=101 + index,
            volume=2,
        )
        for index in range(4)
    ]

    aggregated = aggregate_candles(candles, "1h", "4h", now_ms=start + 4 * 3_600_000)

    assert len(aggregated) == 1
    assert aggregated[0].open_time == start
    assert aggregated[0].open == 100
    assert aggregated[0].high == 108
    assert aggregated[0].low == 95
    assert aggregated[0].close == 104
    assert aggregated[0].volume == 8


def test_aggregates_1m_candles_into_3m() -> None:
    start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    candles = [
        Candle(
            open_time=start + index * 60_000,
            open=10 + index,
            high=11 + index,
            low=9 + index,
            close=10.5 + index,
            volume=1,
        )
        for index in range(3)
    ]

    aggregated = aggregate_candles(candles, "1m", "3m", now_ms=start + 3 * 60_000)

    assert len(aggregated) == 1
    assert aggregated[0].open_time == start
    assert aggregated[0].open == 10
    assert aggregated[0].high == 13
    assert aggregated[0].low == 9
    assert aggregated[0].close == 12.5
    assert aggregated[0].volume == 3


def test_aggregates_daily_candles_into_monday_utc_week() -> None:
    monday = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    candles = [
        Candle(
            open_time=monday + index * 86_400_000,
            open=10 + index,
            high=11 + index,
            low=9 + index,
            close=10.5 + index,
            volume=1,
        )
        for index in range(7)
    ]

    aggregated = aggregate_candles(candles, "1d", "1w", now_ms=monday + 7 * 86_400_000)

    assert len(aggregated) == 1
    assert aggregated[0].open_time == monday
    assert datetime.fromtimestamp(aggregated[0].open_time / 1000, UTC).weekday() == 0


@pytest.mark.asyncio
async def test_coinbase_paginates_aggregated_source_range(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    result = await CandleService(
        _settings(tmp_path, "coinbase", "BTC-USD"),
        store,
        client_factory=client_factory,
    ).get("coinbase", "BTC-USD", "4h", 500)

    assert len(requests) == 7
    assert result["aggregated_from"] == "1h"
    assert result["candles"] == []
    assert result["truncated"] is True
    await store.close()


@pytest.mark.asyncio
async def test_stale_cache_is_returned_when_upstream_fails(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "candles.db")
    await store.open()
    old_candle = Candle(
        open_time=1_791_137_280_000,
        open=100,
        high=110,
        low=90,
        close=105,
        volume=7,
    )
    fetched_at = datetime.now(UTC) - timedelta(seconds=120)
    await store.store_candles("coinbase", "BTC-USD", "1m", [old_candle], fetched_at)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "temporarily unavailable"})

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    result = await CandleService(
        _settings(tmp_path, "coinbase", "BTC-USD"),
        store,
        client_factory=client_factory,
    ).get("coinbase", "BTC-USD", "1m", 1)

    assert result["stale"] is True
    assert "HTTP 503" in result["error"]
    assert result["candles"] == [old_candle.model_dump(mode="json")]
    await store.close()
