from datetime import UTC, datetime, timedelta

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.models import EventType, InstrumentKind, MarketEvent, payload_digest


def stablecoin_trade(
    venue: str, price: float, seconds: int = 0, symbol: str = "USDC-USD"
) -> MarketEvent:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return MarketEvent(
        venue=venue,
        symbol=symbol,
        native_symbol=symbol,
        base_asset=symbol.split("-")[0],
        quote_asset=symbol.split("-")[1],
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
    result = classifier.observe(stablecoin_trade("kraken", 0.981, seconds=1))[0]
    assert not result.abstained
    assert result.label == "depegged"
    assert result.shadow


def test_peg_classifier_throttles_emissions() -> None:
    classifier = MarketStateClassifier()
    assert classifier.observe(stablecoin_trade("coinbase", 1.0))
    assert classifier.observe(stablecoin_trade("kraken", 1.0)) == []


def test_stablecoin_quoted_pairs_are_converted_to_usd() -> None:
    classifier = MarketStateClassifier(emit_interval_seconds=0)
    classifier.observe(stablecoin_trade("coinbase", 0.99, symbol="USDT-USD"))
    classifier.observe(stablecoin_trade("kraken", 1.0, symbol="USDC-USD"))
    result = classifier.observe(stablecoin_trade("okx", 1.0, symbol="USDC-USDT"))[0]
    evidence = {item.metric: item.value for item in result.evidence}
    assert evidence["quote_converted_observations"] == 1
    assert evidence["cross_venue_median_usd_price"] == 0.995
