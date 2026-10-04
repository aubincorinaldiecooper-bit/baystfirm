from datetime import UTC, datetime, timedelta

from baystfirm.bars import MAX_CLOSED_BARS, MinuteBarSeries
from baystfirm.models import Candle, EventType, InstrumentKind, MarketEvent, payload_digest


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
        received_timestamp=timestamp + timedelta(milliseconds=10),
        price=price,
        size=1,
        payload_hash=payload_digest(f"{minute}-{price}"),
    )


def test_aggregates_trade_ohlc_and_closes_on_next_minute() -> None:
    series = MinuteBarSeries()
    assert series.observe(trade(100, 10)) == []
    assert series.observe(trade(100, 12)) == []

    [closed] = series.observe(trade(101, 11))
    assert (closed.open, closed.high, closed.low, closed.close) == (10, 12, 10, 12)
    assert closed.trade_count == 2
    assert not closed.filled
    assert closed.open_time == 100 * 60


def test_late_trade_is_ignored_and_counted() -> None:
    series = MinuteBarSeries()
    series.observe(trade(100, 10))
    series.observe(trade(101, 11))

    assert series.observe(trade(100, 5)) == []
    assert series.late_trades["BTC-USD"] == 1
    [closed] = series.observe(trade(102, 12))
    assert (closed.open, closed.close) == (11, 11)


def test_gap_minutes_are_filled_with_flat_bars() -> None:
    series = MinuteBarSeries()
    series.observe(trade(100, 10))

    closed = series.observe(trade(103, 12))
    assert [bar.open_time for bar in closed] == [6000, 6060, 6120]
    assert [bar.filled for bar in closed] == [False, True, True]
    assert [bar.trade_count for bar in closed] == [1, 0, 0]
    assert [(bar.open, bar.high, bar.low, bar.close) for bar in closed[1:]] == [
        (10, 10, 10, 10),
        (10, 10, 10, 10),
    ]


def test_gap_over_sixty_minutes_resets_warmup() -> None:
    series = MinuteBarSeries()
    series.observe(trade(100, 10))

    assert series.observe(trade(161, 20)) == []
    assert series.closed_bars("BTC-USD") == ()
    [closed] = series.observe(trade(162, 21))
    assert closed.open_time == 161 * 60


def test_closed_bar_series_is_capped() -> None:
    series = MinuteBarSeries()
    for minute in range(MAX_CLOSED_BARS + 2):
        series.observe(trade(minute, 100 + minute))

    bars = series.closed_bars("BTC-USD")
    assert len(bars) == MAX_CLOSED_BARS
    assert bars[0].open_time == 60


def test_seed_populates_only_an_empty_series() -> None:
    series = MinuteBarSeries()
    seed = [
        Candle(open_time=60_000, open=10, high=12, low=9, close=11, volume=3),
        Candle(open_time=120_000, open=11, high=13, low=10, close=12, volume=4),
    ]

    assert series.seed("BTC-USD", seed)
    assert not series.seed("BTC-USD", seed)
    bars = series.closed_bars("BTC-USD")
    assert [bar.open_time for bar in bars] == [60, 120]
    assert [(bar.open, bar.high, bar.low, bar.close) for bar in bars] == [
        (10, 12, 9, 11),
        (11, 13, 10, 12),
    ]
    assert all(bar.trade_count is None for bar in bars)


def test_seed_prepends_live_history_and_fills_to_the_first_live_bar() -> None:
    series = MinuteBarSeries()
    series.observe(trade(100, 10))
    seed = [
        Candle(open_time=96 * 60_000, open=8, high=8, low=8, close=8, volume=1),
        Candle(open_time=98 * 60_000, open=9, high=9, low=9, close=9, volume=1),
    ]

    assert series.seed("BTC-USD", seed)
    bars = series.closed_bars("BTC-USD")
    assert [bar.open_time for bar in bars] == [96 * 60, 97 * 60, 98 * 60, 99 * 60]
    assert [bar.filled for bar in bars] == [False, True, False, True]
    assert [bar.close for bar in bars] == [8, 8, 9, 9]

    [closed] = series.observe(trade(101, 11))
    assert closed.open_time == 100 * 60
    assert series.closed_bars("BTC-USD")[-1] is closed


def test_seed_gap_over_sixty_minutes_rejects_history_and_preserves_live_bars() -> None:
    series = MinuteBarSeries()
    series.observe(trade(100, 10))
    series.observe(trade(101, 11))
    live_bars = series.closed_bars("BTC-USD")
    seed = [Candle(open_time=30 * 60_000, open=5, high=5, low=5, close=5, volume=1)]

    assert not series.seed("BTC-USD", seed)
    assert series.is_seeded("BTC-USD")
    assert series.closed_bars("BTC-USD") == live_bars
    assert all(
        actual is expected
        for actual, expected in zip(series.closed_bars("BTC-USD"), live_bars, strict=True)
    )
    assert not series.seed("BTC-USD", seed)

    [closed] = series.observe(trade(102, 12))
    assert closed.open_time == 101 * 60
