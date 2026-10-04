from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from baystfirm.models import Candle, EventType, MarketEvent

BAR_SECONDS = 60
MAX_CLOSED_BARS = 1500
MAX_GAP_MINUTES = 60


@dataclass
class MinuteBar:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    trade_count: int | None
    filled: bool = False
    first_event_id: UUID | None = None
    last_event_id: UUID | None = None
    first_trade_time: datetime | None = None
    last_trade_time: datetime | None = None


@dataclass
class _SymbolBars:
    closed: deque[MinuteBar] = field(default_factory=lambda: deque(maxlen=MAX_CLOSED_BARS))
    current: MinuteBar | None = None
    late_trades: int = 0
    seeded: bool = False


class MinuteBarSeries:
    def __init__(self) -> None:
        self._symbols: dict[str, _SymbolBars] = {}

    @property
    def late_trades(self) -> dict[str, int]:
        return {symbol: state.late_trades for symbol, state in self._symbols.items()}

    def has_bars(self, symbol: str) -> bool:
        state = self._symbols.get(symbol)
        return state is not None and (bool(state.closed) or state.current is not None)

    def is_seeded(self, symbol: str) -> bool:
        state = self._symbols.get(symbol)
        return state is not None and state.seeded

    def closed_bars(self, symbol: str) -> tuple[MinuteBar, ...]:
        state = self._symbols.get(symbol)
        return tuple(state.closed) if state is not None else ()

    def seed(self, symbol: str, bars: list[Candle]) -> bool:
        state = self._symbols.setdefault(symbol, _SymbolBars())
        if state.seeded:
            return False

        earliest = (
            state.closed[0].open_time
            if state.closed
            else state.current.open_time
            if state.current is not None
            else None
        )
        previous_open_time: int | None = None
        seed_candles: list[Candle] = []
        for candle in bars:
            open_time = candle.open_time // 1000
            if previous_open_time is not None and open_time <= previous_open_time:
                raise ValueError("seed bars must be ordered oldest first")
            if earliest is None or open_time < earliest:
                seed_candles.append(candle)
            previous_open_time = open_time

        state.seeded = True
        seed_bars = fill_minute_bars(seed_candles)
        if not seed_bars:
            return False

        seeded_closed: deque[MinuteBar] = deque(seed_bars, maxlen=MAX_CLOSED_BARS)

        if earliest is not None:
            latest_seed = seeded_closed[-1]
            gap_minutes = (earliest - latest_seed.open_time) // BAR_SECONDS
            if gap_minutes > MAX_GAP_MINUTES:
                return False
            for offset in range(1, gap_minutes):
                seeded_closed.append(
                    _flat_bar(
                        latest_seed.open_time + offset * BAR_SECONDS,
                        latest_seed.close,
                    )
                )

        seeded_closed.extend(state.closed)
        state.closed = seeded_closed
        return True

    def observe(self, event: MarketEvent) -> list[MinuteBar]:
        if event.event_type is not EventType.TRADE or event.price is None:
            return []

        state = self._symbols.setdefault(event.symbol, _SymbolBars())
        minute_open = int(event.exchange_timestamp.timestamp()) // BAR_SECONDS * BAR_SECONDS
        if state.current is None:
            newly_closed: list[MinuteBar] = []
            if state.closed:
                latest = state.closed[-1]
                if minute_open <= latest.open_time:
                    state.late_trades += 1
                    return []
                gap_minutes = (minute_open - latest.open_time) // BAR_SECONDS
                if gap_minutes > MAX_GAP_MINUTES:
                    state.closed.clear()
                else:
                    previous_close = latest.close
                    for offset in range(1, gap_minutes):
                        filled = _flat_bar(latest.open_time + offset * BAR_SECONDS, previous_close)
                        state.closed.append(filled)
                        newly_closed.append(filled)
            state.current = _trade_bar(event, minute_open)
            return newly_closed

        current = state.current
        if minute_open < current.open_time:
            state.late_trades += 1
            return []
        if minute_open == current.open_time:
            _add_trade(current, event)
            return []

        gap_minutes = (minute_open - current.open_time) // BAR_SECONDS
        if gap_minutes > MAX_GAP_MINUTES:
            state.closed.clear()
            state.current = _trade_bar(event, minute_open)
            return []

        newly_closed = [current]
        state.closed.append(current)
        previous_close = current.close
        for offset in range(1, gap_minutes):
            filled = _flat_bar(current.open_time + offset * BAR_SECONDS, previous_close)
            state.closed.append(filled)
            newly_closed.append(filled)
        state.current = _trade_bar(event, minute_open)
        return newly_closed


def _trade_bar(event: MarketEvent, minute_open: int) -> MinuteBar:
    price = event.price
    assert price is not None
    return MinuteBar(
        open_time=minute_open,
        open=price,
        high=price,
        low=price,
        close=price,
        trade_count=1,
        first_event_id=event.event_id,
        last_event_id=event.event_id,
        first_trade_time=event.exchange_timestamp,
        last_trade_time=event.exchange_timestamp,
    )


def _add_trade(bar: MinuteBar, event: MarketEvent) -> None:
    price = event.price
    assert price is not None
    bar.high = max(bar.high, price)
    bar.low = min(bar.low, price)
    if bar.first_trade_time is None or event.exchange_timestamp < bar.first_trade_time:
        bar.open = price
        bar.first_trade_time = event.exchange_timestamp
        bar.first_event_id = event.event_id
    if bar.last_trade_time is None or event.exchange_timestamp >= bar.last_trade_time:
        bar.close = price
        bar.last_trade_time = event.exchange_timestamp
        bar.last_event_id = event.event_id
    bar.trade_count = (bar.trade_count or 0) + 1


def _flat_bar(open_time: int, price: float) -> MinuteBar:
    return MinuteBar(
        open_time=open_time,
        open=price,
        high=price,
        low=price,
        close=price,
        trade_count=0,
        filled=True,
    )


def fill_minute_bars(candles: Sequence[Candle]) -> list[MinuteBar]:
    bars: list[MinuteBar] = []
    previous_open_time: int | None = None
    for candle in candles:
        open_time = candle.open_time // 1000
        if previous_open_time is not None and open_time <= previous_open_time:
            raise ValueError("seed bars must be ordered oldest first")
        if bars:
            previous = bars[-1]
            gap_minutes = (open_time - previous.open_time) // BAR_SECONDS
            if gap_minutes > MAX_GAP_MINUTES:
                bars.clear()
            else:
                for offset in range(1, gap_minutes):
                    bars.append(
                        _flat_bar(
                            previous.open_time + offset * BAR_SECONDS,
                            previous.close,
                        )
                    )
        bars.append(
            MinuteBar(
                open_time=open_time,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                trade_count=None,
            )
        )
        previous_open_time = open_time
    return bars
