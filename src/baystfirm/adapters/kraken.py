from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import MarketAdapter
from baystfirm.adapters.orderbook import LocalOrderBook, depth_within
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
        self._books: dict[str, LocalOrderBook] = {}

    def subscription_messages(self) -> list[dict[str, Any]]:
        if not self.symbols:
            return []
        return [
            {
                "method": "subscribe",
                "params": {"channel": "trade", "symbol": list(self.symbols), "snapshot": False},
            },
            {
                "method": "subscribe",
                "params": {
                    "channel": "book",
                    "symbol": list(self.symbols),
                    "depth": 25,
                    "snapshot": True,
                },
            },
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        if payload.get("channel") == "book" and payload.get("type") in {"snapshot", "update"}:
            return self._parse_book(payload, raw)
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

    def _parse_book(self, payload: dict[str, Any], raw: str) -> list[MarketEvent]:
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        snapshot = payload["type"] == "snapshot"
        for item in payload.get("data", []):
            native_symbol = str(item["symbol"]).upper()
            base, quote = native_symbol.split("/", maxsplit=1)
            book = self._books.setdefault(native_symbol, LocalOrderBook(depth=25))
            book.update(
                ((float(level["price"]), float(level["qty"])) for level in item.get("bids", [])),
                ((float(level["price"]), float(level["qty"])) for level in item.get("asks", [])),
                snapshot=snapshot,
            )
            bids = book.top_n("bids")
            asks = book.top_n("asks")
            best_bid = bids[0] if bids else None
            best_ask = asks[0] if asks else None
            mid = (best_bid[0] + best_ask[0]) / 2 if best_bid and best_ask else 0.0
            depth_levels = min(len(bids), len(asks))
            depth_bids = bids[:depth_levels]
            depth_asks = asks[:depth_levels]
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=f"{base}-{quote}",
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.SPOT,
                    event_type=EventType.BOOK,
                    exchange_timestamp=datetime.fromisoformat(
                        str(item["timestamp"]).replace("Z", "+00:00")
                    ),
                    received_timestamp=datetime.now(UTC),
                    bid=best_bid[0] if best_bid else None,
                    ask=best_ask[0] if best_ask else None,
                    bid_size=best_bid[1] if best_bid else None,
                    ask_size=best_ask[1] if best_ask else None,
                    bid_depth_10bps=depth_within(depth_bids, mid, 10) if mid else None,
                    ask_depth_10bps=depth_within(depth_asks, mid, 10) if mid else None,
                    bid_depth_50bps=depth_within(depth_bids, mid, 50) if mid else None,
                    ask_depth_50bps=depth_within(depth_asks, mid, 50) if mid else None,
                    depth_levels=depth_levels,
                    payload_hash=digest,
                    metadata={
                        "book_snapshot": snapshot,
                        "checksum": item.get("checksum"),
                    },
                )
            )
        return events
