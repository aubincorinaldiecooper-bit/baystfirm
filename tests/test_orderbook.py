from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import MarketAdapter
from baystfirm.adapters.orderbook import LocalOrderBook, depth_within
from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest


def test_depth_within_sums_quote_notional_inside_bps_band() -> None:
    levels = [(99.9, 2.0), (99.8, 1.0), (100.05, 3.0), (99.0, 1.0)]
    depth = depth_within(levels, 100.0, 10)
    assert depth is not None
    assert abs(depth - 499.95) < 1e-9
    assert depth_within([(99.95, 2.0), (100.05, 3.0)], 100.0, 10) is None


def test_local_order_book_applies_updates_deletions_and_snapshots() -> None:
    book = LocalOrderBook(depth=3)
    book.update([(100, 1), (99, 2)], [(101, 1), (102, 2)], snapshot=True)
    book.update([(99, 0), (100.5, 3)], [(101, 0)], snapshot=False)
    assert book.top_n("bids") == [(100.5, 3), (100.0, 1)]
    assert book.top_n("asks") == [(102.0, 2)]

    book.update([(98, 4)], [(103, 5)], snapshot=True)
    assert book.top_n("bids") == [(98.0, 4)]
    assert book.top_n("asks") == [(103.0, 5)]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeAdapter(MarketAdapter):
    name = "test"
    websocket_url = "wss://invalid"

    def subscription_messages(self) -> list[dict[str, Any]]:
        return []

    def parse_message(self, raw: str) -> list[MarketEvent]:
        return []


def event(event_type: EventType, *, bid: float | None = None) -> MarketEvent:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    trade_values = {"price": 100.0, "size": 1.0} if event_type is EventType.TRADE else {}
    return MarketEvent(
        venue="test",
        symbol="BTC-USDT",
        native_symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        instrument_kind=InstrumentKind.PERPETUAL,
        event_type=event_type,
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        bid=bid,
        **trade_values,
        payload_hash=payload_digest(event_type.value),
    )


def test_adapter_throttles_only_selected_event_types_with_injected_clock() -> None:
    clock = FakeClock()
    adapter = _FakeAdapter(("BTC-USDT",), clock=clock)
    book = event(EventType.BOOK, bid=100)
    assert adapter.should_emit(book)
    clock.now = 0.999
    assert not adapter.should_emit(event(EventType.BOOK, bid=101))
    assert adapter.should_emit(event(EventType.QUOTE, bid=101))
    clock.now = 1.0
    latest = event(EventType.BOOK, bid=102)
    assert adapter.should_emit(latest)
    assert latest.bid == 102

    assert adapter.should_emit(event(EventType.TRADE))
    assert adapter.should_emit(event(EventType.TRADE))
    assert adapter.should_emit(event(EventType.LIQUIDATION))
    assert adapter.should_emit(event(EventType.LIQUIDATION))
