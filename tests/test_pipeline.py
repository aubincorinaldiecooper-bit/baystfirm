import asyncio
from datetime import UTC, datetime
from typing import Any

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.models import (
    Classification,
    EventType,
    InstrumentKind,
    MarketEvent,
    payload_digest,
)
from baystfirm.pipeline import IntelligencePipeline, default_classifiers


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
        pipeline = IntelligencePipeline(StoreStub(), HubStub(), [MarketStateClassifier()])  # type: ignore[arg-type]
        await pipeline.ingest(market_event(EventType.TRADE))
        await pipeline.ingest(market_event(EventType.BOOK))
        return pipeline

    pipeline = asyncio.run(ingest())
    assert len(pipeline.latest_events) == 2
    assert {key[2] for key in pipeline.latest_events} == {EventType.TRADE, EventType.BOOK}


def test_pipeline_keeps_latest_classification_per_horizon() -> None:
    class TwoHorizons:
        def __init__(self) -> None:
            self.label = "upward_momentum"

        def observe(self, event: MarketEvent) -> list[Classification]:
            return [
                Classification(
                    classifier="momentum_regime",
                    classifier_version="rules-0.1.0",
                    symbol=event.symbol,
                    label=self.label,
                    probability=0.8,
                    abstained=False,
                    horizon_seconds=horizon,
                    observed_at=event.exchange_timestamp,
                    evidence=[],
                    freshness_ms=event.latency_ms,
                )
                for horizon in (60, 300)
            ]

    async def ingest() -> IntelligencePipeline:
        classifier = TwoHorizons()
        pipeline = IntelligencePipeline(StoreStub(), HubStub(), [classifier])  # type: ignore[arg-type]
        await pipeline.ingest(market_event(EventType.TRADE))
        classifier.label = "range_bound"
        await pipeline.ingest(market_event(EventType.TRADE))
        return pipeline

    pipeline = asyncio.run(ingest())
    assert set(pipeline.latest_classifications) == {
        ("momentum_regime", "BTC-USDT-PERP", 60),
        ("momentum_regime", "BTC-USDT-PERP", 300),
    }
    assert all(item.label == "range_bound" for item in pipeline.latest_classifications.values())


def test_default_classifier_order_keeps_existing_rules_first() -> None:
    classifiers = default_classifiers(shadow=True)
    assert [type(classifier).__name__ for classifier in classifiers] == [
        "MarketStateClassifier",
        "MomentumRegimeClassifier",
    ]
