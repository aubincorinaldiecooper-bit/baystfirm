from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from baystfirm.adapters.base import MarketAdapter, split_symbol
from baystfirm.adapters.orderbook import depth_within
from baystfirm.models import (
    EventType,
    InstrumentKind,
    MarketEvent,
    Side,
    payload_digest,
)

BINANCE_US_SYMBOLS = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
    "BTC-USDT": "BTCUSDT",
    "ETH-USDT": "ETHUSDT",
    "SOL-USDT": "SOLUSDT",
    "USDT-USD": "USDTUSD",
    "USDC-USD": "USDCUSD",
    "USDC-USDT": "USDCUSDT",
}


class BinanceUSAdapter(MarketAdapter):
    name = "binanceus"
    websocket_url = "wss://stream.binance.us:9443"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self._native_to_canonical = {
            BINANCE_US_SYMBOLS[symbol]: symbol for symbol in symbols if symbol in BINANCE_US_SYMBOLS
        }
        super().__init__(tuple(self._native_to_canonical))

    def subscription_messages(self) -> list[dict[str, Any]]:
        return []

    def connection_url(self) -> str:
        streams = [
            f"{native_symbol.lower()}@{channel}"
            for native_symbol in self.symbols
            for channel in ("trade", "depth20@100ms")
        ]
        return f"{self.websocket_url}/stream?streams={'/'.join(streams)}"

    def parse_message(self, raw: str) -> list[MarketEvent]:
        payload = json.loads(raw)
        stream = str(payload.get("stream", "")).lower()
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        native_symbol = str(data.get("s") or stream.split("@", maxsplit=1)[0]).upper()
        symbol = self._native_to_canonical.get(native_symbol)
        if symbol is None:
            return []
        base, quote = split_symbol(symbol)
        digest = payload_digest(raw)

        if stream.endswith("@trade"):
            timestamp = data.get("T", data.get("E"))
            if timestamp is None:
                return []
            return [
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.SPOT,
                    event_type=EventType.TRADE,
                    exchange_timestamp=datetime.fromtimestamp(int(timestamp) / 1000, tz=UTC),
                    received_timestamp=datetime.now(UTC),
                    sequence=data.get("t"),
                    price=float(data["p"]),
                    size=float(data["q"]),
                    side=Side.SELL if data.get("m") is True else Side.BUY,
                    payload_hash=digest,
                    metadata={"buyer_is_maker": data.get("m")},
                )
            ]

        if not stream.endswith("@depth20@100ms"):
            return []
        bids = _levels(data.get("bids"))
        asks = _levels(data.get("asks"))
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        mid = (best_bid[0] + best_ask[0]) / 2 if best_bid and best_ask else 0.0
        timestamp = data.get("E")
        exchange_timestamp = (
            datetime.fromtimestamp(int(timestamp) / 1000, tz=UTC)
            if timestamp is not None
            else datetime.now(UTC)
        )
        return [
            MarketEvent(
                venue=self.name,
                symbol=symbol,
                native_symbol=native_symbol,
                base_asset=base,
                quote_asset=quote,
                instrument_kind=InstrumentKind.SPOT,
                event_type=EventType.BOOK,
                exchange_timestamp=exchange_timestamp,
                received_timestamp=datetime.now(UTC),
                sequence=data.get("lastUpdateId"),
                bid=best_bid[0] if best_bid else None,
                ask=best_ask[0] if best_ask else None,
                bid_size=best_bid[1] if best_bid else None,
                ask_size=best_ask[1] if best_ask else None,
                bid_depth_10bps=depth_within(bids, mid, 10) if mid else None,
                ask_depth_10bps=depth_within(asks, mid, 10) if mid else None,
                bid_depth_50bps=depth_within(bids, mid, 50) if mid else None,
                ask_depth_50bps=depth_within(asks, mid, 50) if mid else None,
                depth_levels=min(len(bids), len(asks)),
                payload_hash=digest,
                metadata={"partial_snapshot": True},
            )
        ]


def _levels(raw_levels: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        return []
    return [
        (float(level[0]), float(level[1]))
        for level in raw_levels
        if isinstance(level, list) and len(level) >= 2
    ]
