from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Collection

from baystfirm.hub import EventHub, StreamItem
from baystfirm.models import MarketEvent

KEEPALIVE_SECONDS = 15.0


def sse_frame(item: StreamItem) -> str:
    name = "market_event" if isinstance(item, MarketEvent) else "classification"
    data = json.dumps(item.model_dump(mode="json"), separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


async def sse_frames(
    hub: EventHub,
    symbols: Collection[str] = (),
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Server-sent events of the live hub, optionally limited to some symbols."""
    wanted = {symbol.upper() for symbol in symbols}
    async with hub.open_queue() as queue:
        yield "retry: 2000\n\n"
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), keepalive_seconds)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if wanted and item.symbol not in wanted:
                continue
            yield sse_frame(item)
