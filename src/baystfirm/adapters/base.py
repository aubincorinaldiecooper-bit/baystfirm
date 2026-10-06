from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from baystfirm.models import EventType, MarketEvent

logger = logging.getLogger(__name__)
EmitEvent = Callable[[MarketEvent], Awaitable[None]]


STABLE_QUOTES = frozenset({"USDT", "USDC"})
THROTTLED_EVENT_TYPES = frozenset(
    {EventType.BOOK, EventType.QUOTE, EventType.FUNDING, EventType.OPEN_INTEREST}
)


def split_symbol(symbol: str) -> tuple[str, str]:
    base, quote = symbol.split("-", maxsplit=1)
    return base, quote


class MarketAdapter(ABC):
    name: str
    websocket_url: str

    def __init__(
        self,
        symbols: tuple[str, ...],
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.symbols = symbols
        self._clock = clock or time.monotonic
        self._last_emitted_at: dict[tuple[str, str, EventType], float] = {}

    def should_emit(self, event: MarketEvent) -> bool:
        if event.event_type not in THROTTLED_EVENT_TYPES:
            return True
        key = (event.venue, event.symbol, event.event_type)
        now = self._clock()
        last = self._last_emitted_at.get(key)
        if last is not None and now - last < 1.0:
            return False
        self._last_emitted_at[key] = now
        return True

    @abstractmethod
    def subscription_messages(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def parse_message(self, raw: str) -> list[MarketEvent]:
        raise NotImplementedError

    def connection_url(self) -> str:
        return self.websocket_url

    async def run(self, emit: EmitEvent, stop: asyncio.Event) -> None:
        backoff_seconds = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(
                    self.connection_url(),
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=10_000,
                ) as websocket:
                    for message in self.subscription_messages():
                        await websocket.send(json.dumps(message))
                    backoff_seconds = 1.0
                    while not stop.is_set():
                        raw = await asyncio.wait_for(websocket.recv(), timeout=30)
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        try:
                            events = self.parse_message(raw)
                        except (KeyError, TypeError, ValueError):
                            logger.warning("%s sent an unparseable message", self.name)
                            continue
                        for event in events:
                            if self.should_emit(event):
                                await emit(event)
            except TimeoutError:
                logger.warning("%s market stream timed out", self.name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s market stream disconnected", self.name)
            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff_seconds)
                except TimeoutError:
                    pass
                backoff_seconds = min(backoff_seconds * 2, 30)
