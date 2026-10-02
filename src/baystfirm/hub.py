from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from baystfirm.models import Classification, MarketEvent

StreamItem = MarketEvent | Classification


class EventHub:
    def __init__(self, queue_size: int = 2_000) -> None:
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue[StreamItem]] = set()
        self._lock = asyncio.Lock()
        self.dropped_events = 0

    async def publish(self, item: StreamItem) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers)
        for queue in subscribers:
            if queue.full():
                self.dropped_events += 1
                continue
            queue.put_nowait(item)

    async def subscribe(self) -> AsyncIterator[StreamItem]:
        queue: asyncio.Queue[StreamItem] = asyncio.Queue(self._queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
