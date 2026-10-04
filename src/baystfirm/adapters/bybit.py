from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import STABLE_QUOTES, MarketAdapter, split_symbol
from baystfirm.adapters.orderbook import LocalOrderBook, depth_within
from baystfirm.models import (
    EventType,
    InstrumentKind,
    MarketEvent,
    Side,
    payload_digest,
)


def _float_or_none(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _datetime_ms_or_none(value: Any) -> datetime | None:
    return None if value in (None, "") else datetime.fromtimestamp(int(value) / 1000, tz=UTC)


class _BybitAdapter(MarketAdapter):
    name = "bybit"
    instrument_kind: InstrumentKind
    orderbook_depth = 1000

    def subscription_messages(self) -> list[dict[str, Any]]:
        topics: list[str] = []
        for symbol in self.symbols:
            topics.append(f"publicTrade.{symbol}")
            if self.instrument_kind is InstrumentKind.PERPETUAL:
                topics.extend((f"tickers.{symbol}", f"allLiquidation.{symbol}"))
            topics.append(f"orderbook.{self.orderbook_depth}.{symbol}")
        return [
            {"op": "subscribe", "args": topics[index : index + 10]}
            for index in range(0, len(topics), 10)
        ]

    def canonical_symbol(self, base: str, quote: str) -> str:
        return f"{base}-{quote}"

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        topic = str(payload.get("topic", ""))
        if topic.startswith("publicTrade."):
            return self._parse_trades(payload, raw)
        if topic.startswith("tickers.") and self.instrument_kind is InstrumentKind.PERPETUAL:
            return self._parse_ticker(payload, raw)
        if topic.startswith("allLiquidation.") and self.instrument_kind is InstrumentKind.PERPETUAL:
            return self._parse_liquidations(payload, raw)
        if topic.startswith("orderbook."):
            return self._parse_orderbook(payload, raw)
        return []

    def _native_parts(self, native_symbol: str) -> tuple[str, str] | None:
        quote = next((item for item in sorted(STABLE_QUOTES) if native_symbol.endswith(item)), None)
        if quote is None:
            return None
        return native_symbol.removesuffix(quote), quote

    def _parse_trades(self, payload: dict[str, Any], raw: str) -> list[MarketEvent]:
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for trade in payload.get("data", []):
            native_symbol = str(trade["s"]).upper()
            parts = self._native_parts(native_symbol)
            if parts is None:
                continue
            base, quote = parts
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=self.canonical_symbol(base, quote),
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=self.instrument_kind,
                    event_type=EventType.TRADE,
                    exchange_timestamp=datetime.fromtimestamp(int(trade["T"]) / 1000, tz=UTC),
                    received_timestamp=datetime.now(UTC),
                    sequence=trade.get("i"),
                    price=float(trade["p"]),
                    size=float(trade["v"]),
                    side=Side.BUY if trade.get("S") == "Buy" else Side.SELL,
                    payload_hash=digest,
                    metadata={"trade_id": trade.get("i"), "block_trade": trade.get("BT", False)},
                )
            )
        return events

    def _parse_ticker(self, payload: dict[str, Any], raw: str) -> list[MarketEvent]:
        data_items = payload.get("data", [])
        if isinstance(data_items, dict):
            data_items = [data_items]
        timestamp = datetime.fromtimestamp(int(payload["ts"]) / 1000, tz=UTC)
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for item in data_items:
            native_symbol = str(
                item.get("symbol") or payload["topic"].split(".", maxsplit=1)[1]
            ).upper()
            if native_symbol not in self.symbols:
                continue
            state = self._ticker_state.setdefault(native_symbol, {})
            if payload.get("type") == "snapshot":
                state.clear()
            state.update(item)
            parts = self._native_parts(native_symbol)
            if parts is None:
                continue
            base, quote = parts
            if any(
                key in item for key in ("fundingRate", "nextFundingTime", "markPrice", "indexPrice")
            ):
                events.append(
                    MarketEvent(
                        venue=self.name,
                        symbol=self.canonical_symbol(base, quote),
                        native_symbol=native_symbol,
                        base_asset=base,
                        quote_asset=quote,
                        instrument_kind=self.instrument_kind,
                        event_type=EventType.FUNDING,
                        exchange_timestamp=timestamp,
                        received_timestamp=datetime.now(UTC),
                        sequence=payload.get("cs"),
                        funding_rate=_float_or_none(state.get("fundingRate")),
                        next_funding_at=_datetime_ms_or_none(state.get("nextFundingTime")),
                        mark_price=_float_or_none(state.get("markPrice")),
                        index_price=_float_or_none(state.get("indexPrice")),
                        payload_hash=digest,
                        metadata={"ticker_type": payload.get("type")},
                    )
                )
            if any(key in item for key in ("openInterest", "openInterestValue")):
                events.append(
                    MarketEvent(
                        venue=self.name,
                        symbol=self.canonical_symbol(base, quote),
                        native_symbol=native_symbol,
                        base_asset=base,
                        quote_asset=quote,
                        instrument_kind=self.instrument_kind,
                        event_type=EventType.OPEN_INTEREST,
                        exchange_timestamp=timestamp,
                        received_timestamp=datetime.now(UTC),
                        sequence=payload.get("cs"),
                        open_interest=_float_or_none(state.get("openInterest")),
                        open_interest_value=_float_or_none(state.get("openInterestValue")),
                        payload_hash=digest,
                        metadata={"ticker_type": payload.get("type")},
                    )
                )
        return events

    def _parse_liquidations(self, payload: dict[str, Any], raw: str) -> list[MarketEvent]:
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for liquidation in payload.get("data", []):
            native_symbol = str(liquidation["s"]).upper()
            if native_symbol not in self.symbols:
                continue
            parts = self._native_parts(native_symbol)
            if parts is None:
                continue
            base, quote = parts
            position_side = str(liquidation.get("S", "")).lower()
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=self.canonical_symbol(base, quote),
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=self.instrument_kind,
                    event_type=EventType.LIQUIDATION,
                    exchange_timestamp=datetime.fromtimestamp(int(liquidation["T"]) / 1000, tz=UTC),
                    received_timestamp=datetime.now(UTC),
                    sequence=liquidation.get("seq"),
                    price=float(liquidation["p"]),
                    size=float(liquidation["v"]),
                    side=Side.BUY if position_side == "buy" else Side.SELL,
                    payload_hash=digest,
                    metadata={"position_side": position_side},
                )
            )
        return events

    def _parse_orderbook(self, payload: dict[str, Any], raw: str) -> list[MarketEvent]:
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        native_symbol = str(data["s"]).upper()
        if native_symbol not in self.symbols:
            return []
        parts = self._native_parts(native_symbol)
        if parts is None:
            return []
        base, quote = parts
        book = self._books.setdefault(native_symbol, LocalOrderBook(depth=self.orderbook_depth))
        snapshot = payload.get("type") == "snapshot"
        book.update(
            ((float(price), float(size)) for price, size in data.get("b", [])),
            ((float(price), float(size)) for price, size in data.get("a", [])),
            snapshot=snapshot,
        )
        bids = book.top_n("bids")
        asks = book.top_n("asks")
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        mid = (best_bid[0] + best_ask[0]) / 2 if best_bid and best_ask else 0.0
        depth_levels = min(len(bids), len(asks))
        return [
            MarketEvent(
                venue=self.name,
                symbol=self.canonical_symbol(base, quote),
                native_symbol=native_symbol,
                base_asset=base,
                quote_asset=quote,
                instrument_kind=self.instrument_kind,
                event_type=EventType.BOOK,
                exchange_timestamp=datetime.fromtimestamp(int(payload["ts"]) / 1000, tz=UTC),
                received_timestamp=datetime.now(UTC),
                sequence=data.get("u"),
                bid=best_bid[0] if best_bid else None,
                ask=best_ask[0] if best_ask else None,
                bid_size=best_bid[1] if best_bid else None,
                ask_size=best_ask[1] if best_ask else None,
                bid_depth_10bps=depth_within(bids, mid, 10) if mid else None,
                ask_depth_10bps=depth_within(asks, mid, 10) if mid else None,
                bid_depth_50bps=depth_within(bids, mid, 50) if mid else None,
                ask_depth_50bps=depth_within(asks, mid, 50) if mid else None,
                depth_levels=depth_levels,
                payload_hash=payload_digest(raw),
                metadata={
                    "book_snapshot": snapshot,
                    "update_id": data.get("u"),
                    "cross_sequence": data.get("seq"),
                },
            )
        ]

    def __init__(self, symbols: tuple[str, ...]) -> None:
        super().__init__(symbols)
        self._ticker_state: dict[str, dict[str, Any]] = {}
        self._books: dict[str, LocalOrderBook] = {}


class BybitLinearAdapter(_BybitAdapter):
    websocket_url = "wss://stream.bybit.com/v5/public/linear"
    instrument_kind = InstrumentKind.PERPETUAL

    def __init__(self, symbols: tuple[str, ...]) -> None:
        native_symbols = []
        for symbol in symbols:
            if not symbol.endswith("-PERP"):
                continue
            base, quote, _ = symbol.split("-")
            native_symbols.append(f"{base}{quote}")
        super().__init__(tuple(native_symbols))

    def canonical_symbol(self, base: str, quote: str) -> str:
        return f"{base}-{quote}-PERP"


class BybitSpotAdapter(_BybitAdapter):
    websocket_url = "wss://stream.bybit.com/v5/public/spot"
    instrument_kind = InstrumentKind.SPOT

    def __init__(self, symbols: tuple[str, ...]) -> None:
        native_symbols = []
        for symbol in symbols:
            if symbol.endswith("-PERP"):
                continue
            base, quote = split_symbol(symbol)
            if quote in STABLE_QUOTES:
                native_symbols.append(f"{base}{quote}")
        super().__init__(tuple(native_symbols))
