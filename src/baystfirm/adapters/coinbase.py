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


class CoinbaseAdapter(MarketAdapter):
    name = "coinbase"
    websocket_url = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        super().__init__(tuple(symbol for symbol in symbols if not symbol.endswith("-PERP")))

    def subscription_messages(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "subscribe",
                "product_ids": [symbol],
                "channels": [channel],
            }
            for symbol in self.symbols
            for channel in ("matches", "ticker")
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        payload_type = payload.get("type")
        if payload_type in {"match", "last_match"}:
            symbol = str(payload["product_id"]).upper()
            if symbol not in self.symbols:
                return []
            base, quote = symbol.split("-", maxsplit=1)
            side = Side.SELL if payload.get("side") == "sell" else Side.BUY
            return [
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.SPOT,
                    event_type=EventType.TRADE,
                    exchange_timestamp=datetime.fromisoformat(
                        payload["time"].replace("Z", "+00:00")
                    ),
                    received_timestamp=datetime.now(UTC),
                    sequence=payload.get("sequence") or payload.get("trade_id"),
                    price=float(payload["price"]),
                    size=float(payload["size"]),
                    side=side,
                    payload_hash=payload_digest(raw),
                    metadata={"trade_id": payload.get("trade_id")},
                )
            ]
        if payload_type != "ticker":
            return []
        symbol = str(payload["product_id"]).upper()
        if symbol not in self.symbols:
            return []
        base, quote = symbol.split("-", maxsplit=1)
        return [
            MarketEvent(
                venue=self.name,
                symbol=symbol,
                native_symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                instrument_kind=InstrumentKind.SPOT,
                event_type=EventType.QUOTE,
                exchange_timestamp=datetime.fromisoformat(payload["time"].replace("Z", "+00:00")),
                received_timestamp=datetime.now(UTC),
                sequence=payload.get("sequence"),
                bid=float(payload["best_bid"]),
                ask=float(payload["best_ask"]),
                bid_size=float(payload["best_bid_size"]),
                ask_size=float(payload["best_ask_size"]),
                payload_hash=payload_digest(raw),
                metadata={"ticker_type": "ticker"},
            )
        ]
