import asyncio
from datetime import UTC, datetime
from typing import Any

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest
from baystfirm.pipeline import IntelligencePipeline


class StoreStub:
    async def append_event(self, event: MarketEvent) -> None:
        pass

    async def append_classification(self, classification: Any) -> None:
        pass


class HubStub:
    async def publish(self, item: Any) -> None:
        pass


def market_event(event_type: EventType) -> MarketEvent:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    return MarketEvent(
        venue="bybit",
        symbol="BTC-USDT-PERP",
        native_symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        instrument_kind=InstrumentKind.PERPETUAL,
        event_type=event_type,
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        price=50_000 if event_type is EventType.TRADE else None,
        size=0.1 if event_type is EventType.TRADE else None,
        payload_hash=payload_digest(event_type.value),
    )


def test_pipeline_keeps_latest_event_by_type() -> None:
    async def ingest() -> IntelligencePipeline:
        pipeline = IntelligencePipeline(StoreStub(), HubStub(), MarketStateClassifier())  # type: ignore[arg-type]
        await pipeline.ingest(market_event(EventType.TRADE))
        await pipeline.ingest(market_event(EventType.BOOK))
        return pipeline

    pipeline = asyncio.run(ingest())
    assert len(pipeline.latest_events) == 2
    assert {key[2] for key in pipeline.latest_events} == {EventType.TRADE, EventType.BOOK}
