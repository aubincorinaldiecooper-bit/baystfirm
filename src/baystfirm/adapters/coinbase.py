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
        if not self.symbols:
            return []
        return [
            {
                "type": "subscribe",
                "product_ids": list(self.symbols),
                "channels": ["matches"],
            }
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if payload.get("type") not in {"match", "last_match"}:
            return []
        symbol = str(payload["product_id"]).upper()
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
                exchange_timestamp=datetime.fromisoformat(payload["time"].replace("Z", "+00:00")),
                received_timestamp=datetime.now(UTC),
                sequence=payload.get("sequence") or payload.get("trade_id"),
                price=float(payload["price"]),
                size=float(payload["size"]),
                side=side,
                payload_hash=payload_digest(raw),
                metadata={"trade_id": payload.get("trade_id")},
            )
        ]
