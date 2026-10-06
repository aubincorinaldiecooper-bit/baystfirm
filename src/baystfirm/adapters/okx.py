from __future__ import annotations

import json
from collections.abc import Mapping
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


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _float_or_none(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _datetime_ms_or_none(value: Any) -> datetime | None:
    return None if value in (None, "") else datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _timestamp(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _book_levels(raw_levels: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_levels, list):
        return []
    return [(float(level[0]), float(level[1])) for level in raw_levels if len(level) >= 2]


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
        self._books: dict[str, LocalOrderBook] = {}

    def subscription_messages(self) -> list[dict[str, Any]]:
        args = [
            {"channel": channel, "instId": symbol}
            for symbol in self.symbols
            for channel in ("trades", "books")
        ]
        return [
            {"op": "subscribe", "args": args[index : index + 20]}
            for index in range(0, len(args), 20)
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        if raw == "pong":
            return []
        payload = json.loads(raw)
        channel = payload.get("arg", {}).get("channel")
        digest = payload_digest(raw)
        if channel == "trades":
            return self._parse_trades(payload, digest)
        if channel == "books":
            symbols = {symbol: symbol for symbol in self.symbols}
            return _parse_books(payload, digest, InstrumentKind.SPOT, symbols, self._books)
        return []

    def _parse_trades(self, payload: dict[str, Any], digest: str) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for trade in _items(payload.get("data")):
            symbol = str(trade["instId"]).upper()
            if symbol not in self.symbols:
                continue
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
                    exchange_timestamp=_timestamp(trade["ts"]),
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


class OkxSwapAdapter(MarketAdapter):
    name = "okx"
    websocket_url = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        supported = []
        for symbol in symbols:
            if not symbol.endswith("-PERP"):
                continue
            base_quote = symbol.removesuffix("-PERP")
            _, quote = split_symbol(base_quote)
            if quote in STABLE_QUOTES:
                supported.append(symbol)
        super().__init__(tuple(supported))
        self._native_to_canonical = {
            symbol.removesuffix("-PERP") + "-SWAP": symbol for symbol in self.symbols
        }
        self._books: dict[str, LocalOrderBook] = {}
        self._derivative_state: dict[str, dict[str, Any]] = {}
        self._contract_multipliers: dict[str, float] = {}
        self._pending_books: dict[str, tuple[dict[str, Any], str, str | None]] = {}

    def subscription_messages(self) -> list[dict[str, Any]]:
        args: list[dict[str, str]] = []
        for native_symbol in self._native_to_canonical:
            args.extend(
                {"channel": channel, "instId": native_symbol}
                for channel in ("trades", "funding-rate", "open-interest", "mark-price", "books")
            )
            index_symbol = native_symbol.removesuffix("-SWAP")
            args.append({"channel": "index-tickers", "instId": index_symbol})
        if self._native_to_canonical:
            args.append({"channel": "liquidation-orders", "instType": "SWAP"})
        return [
            {"op": "subscribe", "args": args[index : index + 20]}
            for index in range(0, len(args), 20)
        ]

    def parse_message(self, raw: str) -> list[MarketEvent]:
        if raw == "pong":
            return []
        payload = json.loads(raw)
        channel = payload.get("arg", {}).get("channel")
        digest = payload_digest(raw)
        if channel == "trades":
            return self._parse_trades(payload, digest)
        if channel in {"funding-rate", "mark-price", "index-tickers"}:
            return self._parse_derivative_prices(payload, digest, channel)
        if channel == "open-interest":
            return self._parse_open_interest(payload, digest)
        if channel == "books":
            return self._parse_books(payload, digest)
        if channel == "liquidation-orders":
            return self._parse_liquidations(payload, digest)
        return []

    def _parse_books(self, payload: dict[str, Any], digest: str) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for item in _items(payload.get("data")):
            native_symbol = str(item.get("instId") or payload.get("arg", {}).get("instId")).upper()
            symbol = self._native_to_canonical.get(native_symbol)
            multiplier = self._contract_multipliers.get(native_symbol)
            if symbol is None:
                continue
            item_payload = {
                "action": payload.get("action"),
                "arg": {"instId": native_symbol},
                "data": [item],
            }
            if multiplier is None:
                self._pending_books[native_symbol] = (
                    item,
                    digest,
                    payload.get("action"),
                )
                events.extend(
                    _parse_books(
                        item_payload,
                        digest,
                        InstrumentKind.PERPETUAL,
                        {native_symbol: symbol},
                        self._books,
                        emit=False,
                    )
                )
                continue
            events.extend(
                _parse_books(
                    item_payload,
                    digest,
                    InstrumentKind.PERPETUAL,
                    {native_symbol: symbol},
                    self._books,
                    size_multiplier=multiplier,
                )
            )
        return events

    def _parse_trades(self, payload: dict[str, Any], digest: str) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for trade in _items(payload.get("data")):
            native_symbol = str(trade["instId"]).upper()
            symbol = self._native_to_canonical.get(native_symbol)
            if symbol is None:
                continue
            base, quote = split_symbol(symbol.removesuffix("-PERP"))
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.PERPETUAL,
                    event_type=EventType.TRADE,
                    exchange_timestamp=_timestamp(trade["ts"]),
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

    def _parse_derivative_prices(
        self,
        payload: dict[str, Any],
        digest: str,
        channel: str,
    ) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for item in _items(payload.get("data")):
            native_symbol = str(item.get("instId") or payload.get("arg", {}).get("instId")).upper()
            if channel == "index-tickers":
                native_symbol += "-SWAP"
            symbol = self._native_to_canonical.get(native_symbol)
            if symbol is None:
                continue
            state = self._derivative_state.setdefault(symbol, {})
            if channel == "funding-rate":
                state["funding_rate"] = _float_or_none(item.get("fundingRate"))
                state["next_funding_at"] = _datetime_ms_or_none(item.get("nextFundingTime"))
            elif channel == "mark-price":
                state["mark_price"] = _float_or_none(item.get("markPx"))
            else:
                state["index_price"] = _float_or_none(item.get("idxPx"))
            base, quote = split_symbol(symbol.removesuffix("-PERP"))
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.PERPETUAL,
                    event_type=EventType.FUNDING,
                    exchange_timestamp=_timestamp(item["ts"]),
                    received_timestamp=datetime.now(UTC),
                    funding_rate=state.get("funding_rate"),
                    next_funding_at=state.get("next_funding_at"),
                    mark_price=state.get("mark_price"),
                    index_price=state.get("index_price"),
                    payload_hash=digest,
                    metadata={"source_channel": channel},
                )
            )
        return events

    def _parse_open_interest(self, payload: dict[str, Any], digest: str) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for item in _items(payload.get("data")):
            native_symbol = str(item["instId"]).upper()
            symbol = self._native_to_canonical.get(native_symbol)
            if symbol is None:
                continue
            open_interest = _float_or_none(item.get("oi"))
            open_interest_currency = _float_or_none(item.get("oiCcy"))
            multiplier = (
                open_interest_currency / open_interest
                if open_interest and open_interest_currency is not None
                else None
            )
            if multiplier is not None:
                self._contract_multipliers[native_symbol] = multiplier
            base, quote = split_symbol(symbol.removesuffix("-PERP"))
            events.append(
                MarketEvent(
                    venue=self.name,
                    symbol=symbol,
                    native_symbol=native_symbol,
                    base_asset=base,
                    quote_asset=quote,
                    instrument_kind=InstrumentKind.PERPETUAL,
                    event_type=EventType.OPEN_INTEREST,
                    exchange_timestamp=_timestamp(item["ts"]),
                    received_timestamp=datetime.now(UTC),
                    open_interest=open_interest,
                    open_interest_value=_float_or_none(item.get("oiUsd")),
                    payload_hash=digest,
                    metadata={
                        "open_interest_currency": item.get("oiCcy"),
                        "contract_multiplier": multiplier,
                    },
                )
            )
            pending = self._pending_books.pop(native_symbol, None)
            if pending is not None and multiplier is not None:
                book_item, book_digest, action = pending
                events.extend(
                    _parse_books(
                        {
                            "action": action,
                            "arg": {"instId": native_symbol},
                            "data": [book_item],
                        },
                        book_digest,
                        InstrumentKind.PERPETUAL,
                        {native_symbol: symbol},
                        self._books,
                        size_multiplier=multiplier,
                    )
                )
        return events

    def _parse_liquidations(self, payload: dict[str, Any], digest: str) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for item in _items(payload.get("data")):
            native_symbol = str(item["instId"]).upper()
            symbol = self._native_to_canonical.get(native_symbol)
            if symbol is None or item.get("instType") != "SWAP":
                continue
            base, quote = split_symbol(symbol.removesuffix("-PERP"))
            for detail in _items(item.get("details")):
                side = str(detail.get("side", "")).lower()
                events.append(
                    MarketEvent(
                        venue=self.name,
                        symbol=symbol,
                        native_symbol=native_symbol,
                        base_asset=base,
                        quote_asset=quote,
                        instrument_kind=InstrumentKind.PERPETUAL,
                        event_type=EventType.LIQUIDATION,
                        exchange_timestamp=_timestamp(detail["ts"]),
                        received_timestamp=datetime.now(UTC),
                        sequence=detail.get("ts"),
                        price=float(detail["bkPx"]),
                        size=float(detail["sz"]),
                        side=Side.BUY if side == "buy" else Side.SELL,
                        payload_hash=digest,
                        metadata={
                            "position_side": detail.get("posSide"),
                            "price_basis": "bankruptcy",
                            "bankruptcy_loss": detail.get("bkLoss"),
                            "contract_multiplier": self._contract_multipliers.get(native_symbol),
                        },
                    )
                )
        return events


def _parse_books(
    payload: dict[str, Any],
    digest: str,
    instrument_kind: InstrumentKind,
    symbols: Mapping[str, str],
    books: dict[str, LocalOrderBook],
    *,
    size_multiplier: float = 1.0,
    emit: bool = True,
) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for item in _items(payload.get("data")):
        native_symbol = str(item.get("instId") or payload.get("arg", {}).get("instId")).upper()
        symbol = symbols.get(native_symbol)
        if symbol is None:
            continue
        base, quote = split_symbol(symbol.removesuffix("-PERP"))
        book = books.setdefault(native_symbol, LocalOrderBook(depth=400))
        book.update(
            _book_levels(item.get("bids")),
            _book_levels(item.get("asks")),
            snapshot=payload.get("action") == "snapshot",
        )
        if not emit:
            continue
        bids = book.top_n("bids")
        asks = book.top_n("asks")
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        mid = (best_bid[0] + best_ask[0]) / 2 if best_bid and best_ask else 0.0
        bid_depth_levels = [(price, size * size_multiplier) for price, size in bids]
        ask_depth_levels = [(price, size * size_multiplier) for price, size in asks]
        events.append(
            MarketEvent(
                venue="okx",
                symbol=symbol,
                native_symbol=native_symbol,
                base_asset=base,
                quote_asset=quote,
                instrument_kind=instrument_kind,
                event_type=EventType.BOOK,
                exchange_timestamp=_timestamp(item["ts"]),
                received_timestamp=datetime.now(UTC),
                sequence=item.get("seqId"),
                bid=best_bid[0] if best_bid else None,
                ask=best_ask[0] if best_ask else None,
                bid_size=best_bid[1] if best_bid else None,
                ask_size=best_ask[1] if best_ask else None,
                bid_depth_10bps=depth_within(bid_depth_levels, mid, 10) if mid else None,
                ask_depth_10bps=depth_within(ask_depth_levels, mid, 10) if mid else None,
                bid_depth_50bps=depth_within(bid_depth_levels, mid, 50) if mid else None,
                ask_depth_50bps=depth_within(ask_depth_levels, mid, 50) if mid else None,
                depth_levels=min(len(bids), len(asks)),
                payload_hash=digest,
                metadata={
                    "source_channel": "books",
                    "sequence_id": item.get("seqId"),
                    "size_unit": "contracts"
                    if instrument_kind is InstrumentKind.PERPETUAL
                    else "base",
                    "contract_multiplier": size_multiplier,
                },
            )
        )
    return events
