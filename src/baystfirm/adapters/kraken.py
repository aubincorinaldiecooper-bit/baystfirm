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


class KrakenAdapter(MarketAdapter):
    name = "kraken"
    websocket_url = "wss://ws.kraken.com/v2"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        supported = []
        for symbol in symbols:
            if symbol.endswith("-PERP"):
                continue
            base, quote = symbol.split("-", maxsplit=1)
            supported.append(f"{base}/{quote}")
        super().__init__(tuple(supported))

    def subscription_messages(self) -> list[dict[str, Any]]:
        if not self.symbols:
            return []
        return [
            {
                "method": "subscribe",
                "params": {"channel": "trade", "symbol": list(self.symbols), "snapshot": False},
            }
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if payload.get("channel") != "trade" or payload.get("type") != "update":
            return []
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for trade in payload.get("data", []):
            native_symbol = str(trade["symbol"]).upper()
            base, quote = native_symbol.split("/", maxsplit=1)
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=f"{base}-{quote}",
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.SPOT,
                    event_type=EventType.TRADE,
                    exchange_timestamp=datetime.fromisoformat(
                        str(trade["timestamp"]).replace("Z", "+00:00")
                    ),
                    received_timestamp=datetime.now(UTC),
                    sequence=trade.get("trade_id"),
                    price=float(trade["price"]),
                    size=float(trade["qty"]),
                    side=Side(str(trade.get("side", "unknown")).lower()),
                    payload_hash=digest,
                    metadata={"trade_id": trade.get("trade_id")},
                )
            )
        return events
