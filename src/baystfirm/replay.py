from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from baystfirm.models import MarketEvent
from baystfirm.storage import EventStore

ReplayCallback = Callable[[MarketEvent], Awaitable[None]]


class ReplayRunner:
    def __init__(self, store: EventStore) -> None:
        self.store = store

    async def run(
        self,
        callback: ReplayCallback,
        *,
        symbol: str | None = None,
        speed: float = 0,
        limit: int | None = None,
    ) -> int:
        if speed < 0:
            raise ValueError("replay speed cannot be negative")
        prior: MarketEvent | None = None
        count = 0
        async for event in self.store.iter_events(symbol=symbol, limit=limit):
            if speed > 0 and prior is not None:
                delay = (
                    event.exchange_timestamp - prior.exchange_timestamp
                ).total_seconds() / speed
                if delay > 0:
                    await asyncio.sleep(delay)
            await callback(event)
            prior = event
            count += 1
        return count
