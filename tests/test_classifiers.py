from datetime import UTC, datetime, timedelta

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest


def stablecoin_trade(venue: str, price: float, seconds: int = 0) -> MarketEvent:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return MarketEvent(
        venue=venue,
        symbol="USDC-USD",
        native_symbol="USDC-USD",
        base_asset="USDC",
        quote_asset="USD",
        instrument_kind=InstrumentKind.SPOT,
        event_type=EventType.TRADE,
        exchange_timestamp=timestamp,
        received_timestamp=timestamp + timedelta(milliseconds=20),
        price=price,
        size=1_000,
        payload_hash=payload_digest(f"{venue}-{price}"),
    )


def test_stablecoin_classifier_abstains_until_cross_venue_confirmation() -> None:
    classifier = MarketStateClassifier()
    result = classifier.observe(stablecoin_trade("coinbase", 1.0))[0]
    assert result.abstained
    assert result.label == "insufficient_cross_venue_data"


def test_stablecoin_classifier_detects_depeg() -> None:
    classifier = MarketStateClassifier()
    classifier.observe(stablecoin_trade("coinbase", 0.98))
    result = classifier.observe(stablecoin_trade("kraken", 0.981))[0]
    assert not result.abstained
    assert result.label == "depegged"
    assert result.shadow
