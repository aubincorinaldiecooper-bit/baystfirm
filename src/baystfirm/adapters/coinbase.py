from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import MarketAdapter
from baystfirm.models import (
    EventType,
    InstrumentKind,
    MarketEvent,
    payload_digest,
)


class CoinbaseAdapter(MarketAdapter):
    name = "coinbase"
    websocket_url = "wss://advanced-trade-ws.coinbase.com"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        super().__init__(tuple(symbol for symbol in symbols if not symbol.endswith("-PERP")))

    def subscription_messages(self) -> list[dict[str, Any]]:
        subscriptions = [
            {"type": "subscribe", "product_ids": [symbol], "channel": "ticker"}
            for symbol in self.symbols
        ]
        subscriptions.append({"type": "subscribe", "channel": "heartbeats"})
        return subscriptions

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if payload.get("channel") != "ticker":
            return []
        timestamp = datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for update in payload.get("events", []):
            for ticker in update.get("tickers", []):
                symbol = str(ticker["product_id"]).upper()
                if symbol not in self.symbols:
                    continue
                base, quote = symbol.split("-", maxsplit=1)
                events.append(
                    MarketEvent(
                        venue=self.name,
                        symbol=symbol,
                        native_symbol=symbol,
                        base_asset=base,
                        quote_asset=quote,
                        instrument_kind=InstrumentKind.SPOT,
                        event_type=EventType.QUOTE,
                        exchange_timestamp=timestamp,
                        received_timestamp=datetime.now(UTC),
                        sequence=payload.get("sequence_num"),
                        bid=float(ticker["best_bid"]),
                        ask=float(ticker["best_ask"]),
                        bid_size=float(ticker["best_bid_quantity"]),
                        ask_size=float(ticker["best_ask_quantity"]),
                        payload_hash=digest,
                        metadata={"ticker_type": ticker.get("type")},
                    )
                )
        return events
