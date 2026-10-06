from datetime import UTC, datetime
from math import log
from pathlib import Path
from statistics import pstdev

import pytest

from baystfirm.bars import MAX_CLOSED_BARS, fill_minute_bars
from baystfirm.config import Settings
from baystfirm.models import Candle, EventType, InstrumentKind, MarketEvent, payload_digest
from baystfirm.regime import MomentumRegimeClassifier, classify_window
from baystfirm.service import _seed_minute_bars


def trade(minute: int, price: float, *, symbol: str = "BTC-USD") -> MarketEvent:
    timestamp = datetime.fromtimestamp(minute * 60, tz=UTC)
    base, quote = symbol.split("-", maxsplit=1)
    return MarketEvent(
        venue="coinbase",
        symbol=symbol,
        native_symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        instrument_kind=InstrumentKind.SPOT,
        event_type=EventType.TRADE,
        exchange_timestamp=timestamp,
        received_timestamp=timestamp,
        price=price,
        size=1,
        payload_hash=payload_digest(f"{minute}-{price}"),
    )


def seed_candles(start_minute: int, prices: list[float]) -> list[Candle]:
    return [
        Candle(
            open_time=(start_minute + index) * 60_000,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1,
        )
        for index, price in enumerate(prices)
    ]


def test_fill_minute_bars_fills_short_gaps_and_drops_older_long_gaps() -> None:
    candles = seed_candles(0, [10, 11])
    candles.extend(seed_candles(5, [12]))
    candles.extend(seed_candles(66, [20, 21, 22]))

    filled = fill_minute_bars(candles[:3])
    assert [bar.open_time for bar in filled] == [0, 60, 120, 180, 240, 300]
    assert [bar.filled for bar in filled] == [False, False, True, True, True, False]

    bars = fill_minute_bars(candles)

    assert [bar.open_time for bar in bars] == [66 * 60, 67 * 60, 68 * 60]


def test_classify_window_matches_live_closed_bar_classification() -> None:
    candles = seed_candles(100_000, [100 + index / 10 for index in range(350)])
    classifier = MomentumRegimeClassifier(shadow=False)
    assert classifier.seed("BTC-USD", candles)
    closed = classifier.bars.closed_bars("BTC-USD")

    live = [
        item
        for index, bar in enumerate(closed)
        for item in classifier._classify_closed_bar("BTC-USD", bar, 0.0)
    ]
    replay = [
        item
        for index in range(len(closed))
        for item in classify_window(
            "BTC-USD",
            closed[max(0, index - (MAX_CLOSED_BARS - 1)) : index + 1],
            shadow=False,
            freshness_ms=0.0,
        )
    ]

    assert [
        (item.label, item.probability, item.horizon_seconds, item.observed_at) for item in live
    ] == [(item.label, item.probability, item.horizon_seconds, item.observed_at) for item in replay]


def classify_next_bar(
    seed_prices: list[float], live_price: float, *, first_live_minute: int = 100_000
):
    classifier = MomentumRegimeClassifier(shadow=False)
    assert classifier.seed(
        "BTC-USD", seed_candles(first_live_minute - len(seed_prices), seed_prices)
    )
    assert classifier.observe(trade(first_live_minute, live_price)) == []
    return classifier.observe(trade(first_live_minute + 1, live_price))


def test_regime_warms_up_before_emitting() -> None:
    classifier = MomentumRegimeClassifier()
    assert classifier.observe(trade(100, 100)) == []
    assert classifier.observe(trade(101, 100)) == []
    assert any(item.horizon_seconds == 60 for item in classifier.observe(trade(102, 100)))


@pytest.mark.parametrize(
    ("last_price", "expected_label"),
    [(101, "upward_momentum"), (99, "downward_momentum"), (100.002, "range_bound")],
)
def test_regime_labels_use_hand_computed_return_and_volatility(
    last_price: float, expected_label: str
) -> None:
    result = classify_next_bar([100, 100], last_price)
    [classification] = [item for item in result if item.horizon_seconds == 60]
    return_value = log(last_price / 100)
    sigma = pstdev([0.0, return_value])
    threshold = max(sigma, 5e-4)
    score = abs(return_value) / threshold
    evidence = {item.metric: item for item in classification.evidence}
    expected_probability = (
        min(0.9, 0.5 + 0.2 * min(score, 2))
        if expected_label != "range_bound"
        else 0.5 + 0.4 * (1 - score)
    )

    assert classification.label == expected_label
    assert classification.probability == pytest.approx(expected_probability)
    assert evidence["lookback_return_bps"].value == pytest.approx(return_value * 10_000)
    assert evidence["lookback_return_bps"].threshold == pytest.approx(threshold * 10_000)
    assert evidence["realized_vol_1m_bps"].value == pytest.approx(sigma * 10_000)
    assert classification.horizon_seconds == 60
    assert classification.observed_at == datetime.fromtimestamp(100_001 * 60, tz=UTC)


def test_regime_evidence_uses_window_trade_ids_and_closing_trade_freshness() -> None:
    classifier = MomentumRegimeClassifier()
    first = trade(100, 100)
    second = trade(101, 101)
    closing = trade(102, 102)

    classifier.observe(first)
    classifier.observe(second)
    results = classifier.observe(closing)
    [classification] = [item for item in results if item.horizon_seconds == 60]

    assert classification.freshness_ms == closing.latency_ms
    assert all(
        evidence.source_event_ids == [first.event_id, second.event_id]
        for evidence in classification.evidence
    )


def test_flat_series_uses_five_basis_point_threshold_floor() -> None:
    [classification] = [
        item for item in classify_next_bar([100, 100], 100) if item.horizon_seconds == 60
    ]
    evidence = {item.metric: item for item in classification.evidence}
    assert classification.label == "range_bound"
    assert evidence["threshold_bps"].value == 5


def test_sixty_minute_horizon_emits_only_on_epoch_cadence() -> None:
    end_minute = 1_000_000 // 12 * 12

    def has_hourly_classification(closed_end_minute: int) -> bool:
        classifier = MomentumRegimeClassifier()
        first_live_minute = closed_end_minute - 1
        assert classifier.seed(
            "BTC-USD",
            seed_candles(first_live_minute - 60, [100] * 60),
        )
        classifier.observe(trade(first_live_minute, 100))
        results = classifier.observe(trade(first_live_minute + 1, 100))
        return any(item.horizon_seconds == 3600 for item in results)

    assert has_hourly_classification(end_minute)
    assert not has_hourly_classification(end_minute + 1)


def test_prepended_seed_enables_hourly_horizon_on_next_live_close() -> None:
    classifier = MomentumRegimeClassifier()
    classifier.observe(trade(106, 100))
    classifier.observe(trade(107, 100))
    assert classifier.seed("BTC-USD", seed_candles(46, [100] * 60))

    results = classifier.observe(trade(108, 100))

    assert any(item.horizon_seconds == 3600 for item in results)


def test_majority_filled_horizon_abstains() -> None:
    classifier = MomentumRegimeClassifier()
    start = 100_000
    assert classifier.seed("BTC-USD", seed_candles(start, [100]))

    result = classifier.observe(trade(start + 2, 101))
    [classification] = [item for item in result if item.horizon_seconds == 60]
    evidence = {item.metric: item for item in classification.evidence}
    assert classification.abstained
    assert classification.label == "range_bound"
    assert classification.probability == 0.5
    assert evidence["filled_bar_share"].value == 1


def test_stablecoin_base_is_skipped() -> None:
    classifier = MomentumRegimeClassifier()
    assert classifier.observe(trade(100, 1, symbol="USDC-USD")) == []
    assert not classifier.has_bars("USDC-USD")


@pytest.mark.asyncio
async def test_seeding_uses_stubbed_candle_source_and_only_closed_bars() -> None:
    settings = Settings(
        database_path=Path("unused.db"),
        enabled_venues=("coinbase", "kraken"),
        shadow_mode=True,
        symbols=("BTC-USD",),
        solana_tokens_enabled=False,
    )
    classifier = MomentumRegimeClassifier()
    minute_open_ms = int(datetime.now(UTC).timestamp() // 60 * 60_000)
    candles = [
        Candle(
            open_time=minute_open_ms - 120_000,
            open=10,
            high=10,
            low=10,
            close=10,
            volume=1,
        ),
        Candle(
            open_time=minute_open_ms - 60_000,
            open=11,
            high=11,
            low=11,
            close=11,
            volume=1,
        ),
        Candle(
            open_time=minute_open_ms + 60_000,
            open=12,
            high=12,
            low=12,
            close=12,
            volume=1,
        ),
    ]
    calls: list[tuple[str, str, str, int]] = []

    async def fetcher(client, venue: str, symbol: str, interval: str, limit: int):
        calls.append((venue, symbol, interval, limit))
        return candles

    await _seed_minute_bars(settings, classifier, fetcher=fetcher)
    await _seed_minute_bars(settings, classifier, fetcher=fetcher)

    assert calls == [("coinbase", "BTC-USD", "1m", 1500)]
    seeded = classifier.bars.closed_bars("BTC-USD")
    assert [bar.close for bar in seeded] == [10, 11]


@pytest.mark.asyncio
async def test_seeding_falls_back_in_venue_order_for_perpetuals() -> None:
    settings = Settings(
        database_path=Path("unused.db"),
        enabled_venues=("bybit", "okx", "coinbase"),
        shadow_mode=True,
        symbols=("BTC-USDT-PERP",),
        solana_tokens_enabled=False,
    )
    classifier = MomentumRegimeClassifier()
    now_minute_ms = int(datetime.now(UTC).timestamp() // 60 * 60_000)
    candle = Candle(
        open_time=now_minute_ms - 120_000,
        open=50_000,
        high=50_000,
        low=50_000,
        close=50_000,
        volume=1,
    )
    calls: list[tuple[str, str, str, int]] = []

    async def fetcher(client, venue: str, symbol: str, interval: str, limit: int):
        calls.append((venue, symbol, interval, limit))
        if venue == "okx":
            raise RuntimeError("upstream unavailable")
        return [candle]

    await _seed_minute_bars(settings, classifier, fetcher=fetcher)

    assert calls == [
        ("okx", "BTC-USDT-PERP", "1m", 1500),
        ("bybit", "BTC-USDT-PERP", "1m", 1500),
    ]
    assert classifier.has_bars("BTC-USDT-PERP")


@pytest.mark.asyncio
async def test_service_seeding_prepends_after_live_trades() -> None:
    settings = Settings(
        database_path=Path("unused.db"),
        enabled_venues=("coinbase",),
        shadow_mode=True,
        symbols=("BTC-USD",),
        solana_tokens_enabled=False,
    )
    classifier = MomentumRegimeClassifier()
    minute_open_ms = int(datetime.now(UTC).timestamp() // 60 * 60_000)
    current_minute = minute_open_ms // 60_000
    classifier.observe(trade(current_minute - 1, 10))
    classifier.observe(trade(current_minute, 11))
    existing = classifier.bars.closed_bars("BTC-USD")[0]
    candles = [
        Candle(
            open_time=minute_open_ms - 180_000,
            open=8,
            high=8,
            low=8,
            close=8,
            volume=1,
        ),
        Candle(
            open_time=minute_open_ms - 120_000,
            open=9,
            high=9,
            low=9,
            close=9,
            volume=1,
        ),
    ]
    calls: list[str] = []

    async def fetcher(client, venue: str, symbol: str, interval: str, limit: int):
        calls.append(venue)
        return candles

    await _seed_minute_bars(settings, classifier, fetcher=fetcher)

    seeded = classifier.bars.closed_bars("BTC-USD")
    assert calls == ["coinbase"]
    assert classifier.is_seeded("BTC-USD")
    assert [bar.open_time for bar in seeded[:2]] == [
        minute_open_ms // 1000 - 180,
        minute_open_ms // 1000 - 120,
    ]
    assert seeded[-1] is existing
