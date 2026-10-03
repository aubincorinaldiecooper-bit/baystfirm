from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from baystfirm.adapters import (
    BybitLinearAdapter,
    BybitSpotAdapter,
    CoinbaseAdapter,
    KrakenAdapter,
    MarketAdapter,
    OkxSpotAdapter,
)
from baystfirm.config import Settings
from baystfirm.models import MarketEvent
from baystfirm.pipeline import IntelligencePipeline

logger = logging.getLogger(__name__)


def build_adapters(settings: Settings) -> list[MarketAdapter]:
    factories: dict[str, tuple[Callable[[tuple[str, ...]], MarketAdapter], ...]] = {
        "coinbase": (CoinbaseAdapter,),
        "kraken": (KrakenAdapter,),
        "bybit": (BybitSpotAdapter, BybitLinearAdapter),
        "okx": (OkxSpotAdapter,),
    }
    adapters: list[MarketAdapter] = []
    for venue in settings.enabled_venues:
        venue_factories = factories.get(venue)
        if venue_factories is None:
            logger.warning("ignoring unknown venue adapter: %s", venue)
            continue
        for factory in venue_factories:
            adapter = factory(settings.symbols)
            if adapter.symbols:
                adapters.append(adapter)
    return adapters


class IngestionSupervisor:
    def __init__(
        self,
        settings: Settings,
        pipeline: IntelligencePipeline,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.stop_event = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []
        self.events_received = 0
        self.last_event_by_venue: dict[str, MarketEvent] = {}

    async def start(self) -> None:
        for adapter in build_adapters(self.settings):
            self.tasks.append(
                asyncio.create_task(
                    adapter.run(self._emit, self.stop_event),
                    name=f"market-stream-{adapter.name}-{type(adapter).__name__}",
                )
            )

    async def stop(self) -> None:
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _emit(self, event: MarketEvent) -> None:
        await self.pipeline.ingest(event)
        self.events_received += 1
        self.last_event_by_venue[event.venue] = event
