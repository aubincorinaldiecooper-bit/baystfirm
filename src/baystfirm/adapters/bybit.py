from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import MarketAdapter
from baystfirm.models import (
    EventType,
    InstrumentKind,
    MarketEvent,
    Side,
    payload_digest,
)


class BybitLinearAdapter(MarketAdapter):
    name = "bybit"
    websocket_url = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        native_symbols = []
        for symbol in symbols:
            if not symbol.endswith("-PERP"):
                continue
            base, quote, _ = symbol.split("-")
            native_symbols.append(f"{base}{quote}")
        super().__init__(tuple(native_symbols))

    def subscription_messages(self) -> list[dict[str, Any]]:
        if not self.symbols:
            return []
        return [{"op": "subscribe", "args": [f"publicTrade.{symbol}" for symbol in self.symbols]}]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if not str(payload.get("topic", "")).startswith("publicTrade."):
            return []
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for trade in payload.get("data", []):
            native_symbol = str(trade["s"]).upper()
            quote = "USDT" if native_symbol.endswith("USDT") else "USDC"
            base = native_symbol.removesuffix(quote)
            timestamp = datetime.fromtimestamp(int(trade["T"]) / 1000, tz=UTC)
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=f"{base}-{quote}-PERP",
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.PERPETUAL,
                    event_type=EventType.TRADE,
                    exchange_timestamp=timestamp,
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
