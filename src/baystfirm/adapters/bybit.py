from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import STABLE_QUOTES, MarketAdapter, split_symbol
from baystfirm.models import (
    EventType,
    InstrumentKind,
    MarketEvent,
    Side,
    payload_digest,
)


class _BybitAdapter(MarketAdapter):
    name = "bybit"
    instrument_kind: InstrumentKind

    def subscription_messages(self) -> list[dict[str, Any]]:
        return [{"op": "subscribe", "args": [f"publicTrade.{symbol}"]} for symbol in self.symbols]

    def canonical_symbol(self, base: str, quote: str) -> str:
        return f"{base}-{quote}"

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if not str(payload.get("topic", "")).startswith("publicTrade."):
            return []
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for trade in payload.get("data", []):
            native_symbol = str(trade["s"]).upper()
            quote = next(q for q in sorted(STABLE_QUOTES) if native_symbol.endswith(q))
            base = native_symbol.removesuffix(quote)
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
