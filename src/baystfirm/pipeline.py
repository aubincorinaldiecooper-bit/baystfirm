from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.hub import EventHub
from baystfirm.models import Classification, EventType, MarketEvent
from baystfirm.news import NewsItem
from baystfirm.regime import MomentumRegimeClassifier
from baystfirm.storage import EventStore

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def observe(self, event: MarketEvent) -> list[Classification]: ...


class NewsObserver(Protocol):
    def observe(self, event: MarketEvent) -> list[NewsItem]: ...


def default_classifiers(*, shadow: bool) -> list[Classifier]:
    return [
        MarketStateClassifier(shadow=shadow),
        MomentumRegimeClassifier(shadow=shadow),
    ]


class IntelligencePipeline:
    def __init__(
        self,
        store: EventStore,
        hub: EventHub,
        classifiers: Sequence[Classifier],
        *,
        news: NewsObserver | None = None,
    ) -> None:
        self.store = store
        self.hub = hub
        self.classifiers = tuple(classifiers)
        self.news = news
        self.classifications_generated = 0
        self.latest_events: dict[tuple[str, str, EventType], MarketEvent] = {}
        self.latest_classifications: dict[tuple[str, str, int], Classification] = {}

    async def ingest(self, event: MarketEvent) -> None:
        await self.store.append_event(event)
        if self.news is not None:
            try:
                news_items = self.news.observe(event)
            except Exception:
                logger.exception("News observer failed while ingesting %s", event.event_id)
            else:
                for item in news_items:
                    try:
                        await self.store.append_news(item)
                    except Exception:
                        logger.exception("News item could not be stored: %s", item.id)
        self.latest_events[(event.venue, event.symbol, event.event_type)] = event
        await self.hub.publish(event)
        for classifier in self.classifiers:
            for classification in classifier.observe(event):
                await self.store.append_classification(classification)
                key = (
                    classification.classifier,
                    classification.symbol,
                    classification.horizon_seconds,
                )
                self.latest_classifications[key] = classification
                await self.hub.publish(classification)
                self.classifications_generated += 1
