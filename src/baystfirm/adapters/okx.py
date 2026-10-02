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


class OkxSpotAdapter(MarketAdapter):
    name = "okx"
    websocket_url = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        supported = []
        for symbol in symbols:
            if symbol.endswith("-PERP"):
                continue
            _, quote = split_symbol(symbol)
            if quote in STABLE_QUOTES:
                supported.append(symbol)
        super().__init__(tuple(supported))

    def subscription_messages(self) -> list[dict[str, Any]]:
        return [
            {"op": "subscribe", "args": [{"channel": "trades", "instId": symbol}]}
            for symbol in self.symbols
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        if raw == "pong":
            return []
        payload = json.loads(raw)
        if payload.get("arg", {}).get("channel") != "trades" or "data" not in payload:
            return []
        digest = payload_digest(raw)
        events: list[MarketEvent] = []
        for trade in payload["data"]:
            symbol = str(trade["instId"]).upper()
            base, quote = split_symbol(symbol)
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.SPOT,
                    event_type=EventType.TRADE,
                    exchange_timestamp=datetime.fromtimestamp(int(trade["ts"]) / 1000, tz=UTC),
                    received_timestamp=datetime.now(UTC),
                    sequence=trade.get("seqId") or trade.get("tradeId"),
                    price=float(trade["px"]),
                    size=float(trade["sz"]),
                    side=Side.BUY if trade.get("side") == "buy" else Side.SELL,
                    payload_hash=digest,
                    metadata={"trade_id": trade.get("tradeId")},
                )
            )
        return events
