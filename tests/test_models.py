from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest


def make_trade() -> MarketEvent:
    exchange_time = datetime.now(UTC) - timedelta(milliseconds=25)
    return MarketEvent(
        venue="coinbase",
        symbol="BTC-USD",
        native_symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        instrument_kind=InstrumentKind.SPOT,
        event_type=EventType.TRADE,
        exchange_timestamp=exchange_time,
        price=60_000,
        size=0.1,
        payload_hash=payload_digest("{}"),
    )


def test_market_event_reports_observed_latency() -> None:
    event = make_trade()
    assert event.latency_ms >= 25


def test_trade_requires_price_and_size() -> None:
    trade = make_trade().model_dump()
    trade["price"] = None
    with pytest.raises(ValidationError):
        MarketEvent.model_validate(trade)


def test_crossed_quote_is_rejected() -> None:
    quote = make_trade().model_copy(
        update={"event_type": EventType.QUOTE, "bid": 61_000, "ask": 60_000}
    )
    with pytest.raises(ValidationError):
        MarketEvent.model_validate(quote.model_dump())
