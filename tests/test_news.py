from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from baystfirm.config import Settings
from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest
from baystfirm.news import (
    NEWS_NOTE,
    MarketNewsObserver,
    NewsItem,
    NewsService,
    parse_official_feed,
)
from baystfirm.service import create_app
from baystfirm.storage import EventStore

FIXTURES = Path(__file__).parent / "fixtures" / "news"
USER_AGENT = "Baystfirm test contact@example.com"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _recent_fixture(name: str, published_at: datetime) -> bytes:
    return (
        _fixture(name)
        .replace(
            b"Fri, 03 Apr 2026 12:30:00 GMT",
            published_at.strftime("%a, %d %b %Y %H:%M:%S GMT").encode(),
        )
        .replace(
            b"2026-04-03T14:15:00Z",
            published_at.strftime("%Y-%m-%dT%H:%M:%SZ").encode(),
        )
        .replace(
            b"2026-04-03T16:45:00Z",
            published_at.strftime("%Y-%m-%dT%H:%M:%SZ").encode(),
        )
    )


def _item(
    item_id: str,
    *,
    kind: str = "market_event",
    source: str = "baystfirm",
    published_at: datetime | None = None,
    symbols: list[str] | None = None,
) -> NewsItem:
    return NewsItem(
        id=item_id,
        kind=kind,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        source_label="Baystfirm (measured)",
        title=f"Headline {item_id}",
        url=None,
        published_at=published_at or datetime(2026, 4, 3, tzinfo=UTC),
        symbols=symbols or [],
        details={},
    )


def test_news_settings_read_environment_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAYST_NEWS", raising=False)
    monkeypatch.delenv("BAYST_SEC_USER_AGENT", raising=False)
    defaults = Settings.from_env()
    assert defaults.news_enabled is True
    assert defaults.sec_user_agent == "Baystfirm/0.1 aubincorinaldiecooper@gmail.com"

    monkeypatch.setenv("BAYST_NEWS", "no")
    monkeypatch.setenv("BAYST_SEC_USER_AGENT", "Baystfirm test contact@example.com")
    configured = Settings.from_env()
    assert configured.news_enabled is False
    assert configured.sec_user_agent == USER_AGENT


def _market_event(
    event_type: EventType,
    observed_at: datetime,
    *,
    venue: str = "bybit",
    symbol: str = "BTC-USDT-PERP",
    price: float,
    size: float = 1.0,
    base_asset: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MarketEvent:
    parts = symbol.split("-")
    base = base_asset or parts[0]
    quote = parts[1] if len(parts) > 1 else "USD"
    return MarketEvent(
        venue=venue,
        symbol=symbol,
        native_symbol=symbol.replace("-", ""),
        base_asset=base,
        quote_asset=quote,
        instrument_kind=(InstrumentKind.PERPETUAL if "PERP" in parts else InstrumentKind.SPOT),
        event_type=event_type,
        exchange_timestamp=observed_at,
        received_timestamp=observed_at,
        price=price,
        size=size,
        payload_hash=payload_digest(f"{event_type}-{venue}-{symbol}-{observed_at.isoformat()}"),
        metadata=metadata or {},
    )


def test_official_feed_parsers_handle_rss_rdf_and_atom() -> None:
    fetched_at = datetime(2026, 4, 4, tzinfo=UTC)
    rss = parse_official_feed("sec", _fixture("sec-rss.xml"), fetched_at=fetched_at)
    rdf = parse_official_feed("bank_of_canada", _fixture("bank-rdf.xml"), fetched_at=fetched_at)
    atom = parse_official_feed(
        "federal_reserve",
        _fixture("federal-reserve-atom.xml"),
        fetched_at=fetched_at,
    )

    assert len(rss) == 1
    assert rss[0].source_label == "U.S. SEC"
    assert rss[0].title == "SEC announces new rules"
    assert rss[0].url == "https://www.sec.gov/news/press-release/example"
    assert rss[0].published_at == datetime(2026, 4, 3, 12, 30, tzinfo=UTC)
    assert rss[0].symbols == [] and rss[0].details == {}
    assert rdf[0].source_label == "Bank of Canada"
    assert rdf[0].published_at == datetime(2026, 4, 3, 14, 15, tzinfo=UTC)
    assert atom[0].source_label == "Federal Reserve"
    assert atom[0].url == "https://www.federalreserve.gov/newsevents/example.htm"
    assert atom[0].published_at == datetime(2026, 4, 3, 16, 45, tzinfo=UTC)


async def test_failed_official_feed_does_not_stop_other_sources(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "news.db")
    await store.open()
    recent_at = datetime.now(UTC) - timedelta(days=1)
    bodies = {
        "www.sec.gov": _recent_fixture("sec-rss.xml", recent_at),
        "www.cftc.gov": b"unused",
        "www.federalreserve.gov": _recent_fixture("federal-reserve-atom.xml", recent_at),
        "www.bankofcanada.ca": _recent_fixture("bank-rdf.xml", recent_at),
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "www.cftc.gov" in request.url.host:
            return httpx.Response(500, request=request)
        return httpx.Response(200, content=bodies[request.url.host], request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = NewsService(store, USER_AGENT, client=client)
    try:
        await service.poll_official_once()
        saved = await store.iter_news(limit=20)
        statuses = {item["source"]: item for item in service.official_sources()}
        assert len(saved) == 3
        assert statuses["cftc"]["last_success_at"] is None
        assert statuses["cftc"]["last_error"]
        assert all(
            statuses[source]["last_success_at"]
            for source in ("sec", "federal_reserve", "bank_of_canada")
        )
        assert all(request.headers["user-agent"] == USER_AGENT for request in requests)
    finally:
        await client.aclose()
        await store.close()


async def test_sec_filings_use_fixtures_and_keep_supported_amendments() -> None:
    tickers: dict[str, Any] = json.loads(_fixture("company-tickers.json"))
    submissions = json.loads(_fixture("aapl-submissions.json"))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json=tickers, request=request)
        if request.url.path == "/submissions/CIK0000320193.json":
            return httpx.Response(200, json=submissions, request=request)
        return httpx.Response(404, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = NewsService(_UnusedStore(), USER_AGENT, client=client)  # type: ignore[arg-type]
    try:
        filings, note = await service.filings_for_ticker("AAPL", 20)
        unknown, unknown_note = await service.filings_for_ticker("ZZZZ", 20)
        assert note is None
        assert unknown == []
        assert (
            unknown_note
            == "No SEC filer found for ZZZZ (non-US companies may not file with the SEC)."
        )
        assert [item.title for item in filings] == [
            "Apple Inc.: Form 8-K (Current report) · Items 2.02, 7.01",
            "Apple Inc.: Form 10-Q",
            "Apple Inc.: Form 8-K/A (Amended current report) · Items 1.01",
        ]
        assert filings[0].url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl-8k.htm"
        )
        assert filings[2].published_at == datetime(2026, 3, 25, tzinfo=UTC)
        assert all(item.symbols == ["AAPL"] for item in filings)
        assert len(requests) == 2
        assert all(request.headers["user-agent"] == USER_AGENT for request in requests)
    finally:
        await client.aclose()


class _UnusedStore:
    async def append_news(self, item: NewsItem) -> None:
        del item

    async def prune_news(self, older_than: datetime) -> None:
        del older_than


async def test_sec_failures_return_empty_items_and_a_note() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = NewsService(_UnusedStore(), USER_AGENT, client=client)  # type: ignore[arg-type]
    try:
        items, note = await service.filings_for_ticker("AAPL", 20)
        assert items == []
        assert note is not None and "SEC filings unavailable for AAPL" in note
    finally:
        await client.aclose()


async def test_news_storage_deduplicates_filters_exact_symbols_and_prunes(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "news-storage.db")
    await store.open()
    now = datetime(2026, 4, 3, tzinfo=UTC)
    await store.append_news(
        _item(
            "one", published_at=now, symbols=["BTC", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"]
        )
    )
    await store.append_news(_item("one", published_at=now + timedelta(days=1), symbols=["ETH"]))
    await store.append_news(
        _item(
            "two", kind="official", source="sec", published_at=now + timedelta(days=2), symbols=[]
        )
    )
    await store.append_news(_item("old", published_at=now - timedelta(days=31), symbols=["BTC"]))

    btc_items = await store.iter_news(symbol="BTC", kinds={"market_event"}, limit=20)
    mint_items = await store.iter_news(
        symbol="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        kinds={"market_event"},
        limit=20,
    )
    wrong_case = await store.iter_news(
        symbol="dezxaz8z7pnrnrjjz3wxboRgixCa6xjnB7YaB1pPB263",
        kinds={"market_event"},
        limit=20,
    )
    await store.prune_news(now - timedelta(days=30))
    after_prune = await store.iter_news(limit=20)

    assert [item.id for item in btc_items] == ["one", "old"]
    assert [item.id for item in mint_items] == ["one"]
    assert wrong_case == []
    assert {item.id for item in after_prune} == {"one", "two"}
    await store.close()


def test_liquidation_threshold_multiplier_and_cooldown_rules() -> None:
    observer = MarketNewsObserver()
    start = datetime(2026, 4, 3, tzinfo=UTC)
    multiplier_without = _market_event(
        EventType.LIQUIDATION,
        start,
        venue="okx",
        symbol="ETH-USDT-PERP",
        base_asset="ETH",
        price=1_000,
        size=100,
    )
    multiplier_with = multiplier_without.model_copy(
        update={"metadata": {"contract_multiplier": 3.0}}
    )
    assert observer.observe(multiplier_without) == []
    multiplied = observer.observe(multiplier_with)
    assert len(multiplied) == 1
    assert multiplied[0].details["notional_usd"] == 300_000
    assert "position liquidated on OKX" in multiplied[0].title

    cooldown_observer = MarketNewsObserver()
    schedule = (
        (249_999, 0),
        (250_000, 60),
        (250_000, 10 * 60),
        (250_000, 31 * 60),
    )
    items = [
        item
        for amount, offset in schedule
        for item in cooldown_observer.observe(
            _market_event(
                EventType.LIQUIDATION,
                start + timedelta(seconds=offset),
                venue="bybit",
                price=float(amount),
            )
        )
    ]
    assert len(items) == 2
    assert [item.published_at for item in items] == [
        start + timedelta(seconds=60),
        start + timedelta(seconds=31 * 60),
    ]
    assert all("long" not in item.title.casefold() for item in items)
    assert all("short" not in item.title.casefold() for item in items)


def test_liquidation_burst_uses_cross_venue_five_minute_window() -> None:
    start = datetime(2026, 4, 3, tzinfo=UTC)
    burst_observer = MarketNewsObserver()
    burst_observer.observe(
        _market_event(
            EventType.LIQUIDATION,
            start,
            venue="okx",
            symbol="SOL-USDT-PERP",
            base_asset="SOL",
            price=600_000,
        )
    )
    burst_items = burst_observer.observe(
        _market_event(
            EventType.LIQUIDATION,
            start + timedelta(minutes=4),
            venue="bybit",
            symbol="SOL-USDT-PERP",
            base_asset="SOL",
            price=600_000,
        )
    )
    burst = next(item for item in burst_items if item.details["rule"] == "liquidation_burst")
    assert burst.details["notional_usd"] == 1_200_000
    assert burst.details["venues"] == ["Bybit", "OKX"]
    assert "1.2M of SOL positions liquidated" in burst.title

    spread_observer = MarketNewsObserver()
    spread_observer.observe(
        _market_event(
            EventType.LIQUIDATION,
            start,
            venue="okx",
            symbol="SOL-USDT-PERP",
            base_asset="SOL",
            price=600_000,
        )
    )
    spread = spread_observer.observe(
        _market_event(
            EventType.LIQUIDATION,
            start + timedelta(minutes=6),
            venue="bybit",
            symbol="SOL-USDT-PERP",
            base_asset="SOL",
            price=600_000,
        )
    )
    assert not any(item.details["rule"] == "liquidation_burst" for item in spread)


def test_stablecoin_off_peg_requires_three_trades_and_rearms_after_calm_period() -> None:
    observer = MarketNewsObserver()
    start = datetime(2026, 4, 3, tzinfo=UTC)

    def trade(offset: int, price: float) -> list[NewsItem]:
        return observer.observe(
            _market_event(
                EventType.TRADE,
                start + timedelta(seconds=offset),
                venue="kraken",
                symbol="USDC-USD",
                price=price,
            )
        )

    assert trade(0, 0.99) == []
    assert trade(10, 0.99) == []
    first = trade(20, 0.99)
    assert len(first) == 1
    assert "USDC trading 1.00% below 1 USD on Kraken" in first[0].title
    assert first[0].details["trade_count"] == 3
    assert trade(30, 0.99) == []

    for offset in (180, 190, 200, *range(260, 3621, 60)):
        assert trade(offset, 1.0) == []
    second = [item for offset in (3630, 3640, 3650) for item in trade(offset, 0.98)]
    assert len(second) == 1
    assert second[0].details["rule"] == "stablecoin_off_peg"


def test_stablecoin_headline_reports_cross_market_medians_and_agreement() -> None:
    start = datetime(2026, 4, 3, tzinfo=UTC)

    def readings(prices: dict[str, float]) -> list[NewsItem]:
        observer = MarketNewsObserver()
        items: list[NewsItem] = []
        for venue, price in prices.items():
            for offset in (0, 10, 20):
                items.extend(
                    observer.observe(
                        _market_event(
                            EventType.TRADE,
                            start + timedelta(seconds=offset),
                            venue=venue,
                            symbol="USDC-USD",
                            price=price,
                        )
                    )
                )
        return items

    one_off = readings({"coinbase": 1.0, "binanceus": 1.0, "kraken": 0.99})
    assert len(one_off) == 1
    assert "; 1/3 venues off by ≥0.5%" in one_off[0].title
    assert one_off[0].details["cross_market_median"] == pytest.approx(1.0)
    assert one_off[0].details["venue_count"] == 3
    assert one_off[0].details["agreeing_count"] == 1
    assert {
        reading["venue"]: reading["median_price"]
        for reading in one_off[0].details["venue_readings"]
    } == {"Binance.US": 1.0, "Coinbase": 1.0, "Kraken": 0.99}
    assert {reading["trade_count"] for reading in one_off[0].details["venue_readings"]} == {3}

    all_off = readings({"coinbase": 0.99, "binanceus": 0.99, "kraken": 0.99})
    last = all_off[-1]
    assert "; 3/3 venues off by ≥0.5%" in last.title
    assert last.details["cross_market_median"] == pytest.approx(0.99)
    assert last.details["venue_count"] == 3
    assert last.details["agreeing_count"] == 3


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        database_path=tmp_path / "service.db",
        enabled_venues=(),
        shadow_mode=True,
        symbols=(),
        api_key=api_key,
        solana_tokens_enabled=False,
        news_enabled=False,
        sec_user_agent=USER_AGENT,
    )


def test_news_endpoint_reads_seeded_storage_without_polling(tmp_path: Path) -> None:
    async def seed() -> None:
        store = EventStore(tmp_path / "service.db")
        await store.open()
        await store.append_news(_item("seeded", symbols=["BTC"]))
        await store.close()

    asyncio.run(seed())
    with TestClient(create_app(_settings(tmp_path, api_key="secret"))) as client:
        unauthorized = client.get("/v1/news")
        response = client.get(
            "/v1/news?symbol=btc&kind=market_event&limit=10",
            headers={"Authorization": "Bearer secret"},
        )
        assert unauthorized.status_code == 401
        assert response.status_code == 200
        payload = response.json()
        assert payload["items"][0]["id"] == "seeded"
        assert len(payload["sources"]) == 4
        assert payload["note"] == NEWS_NOTE
        invalid_ticker = client.get(
            "/v1/news/filings?tickers=bad%21",
            headers={"Authorization": "Bearer secret"},
        )
        assert invalid_ticker.status_code == 422
