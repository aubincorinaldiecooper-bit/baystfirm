from __future__ import annotations

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.hub import EventHub
from baystfirm.models import Classification, MarketEvent
from baystfirm.storage import EventStore


class IntelligencePipeline:
    def __init__(
        self,
        store: EventStore,
        hub: EventHub,
        classifier: MarketStateClassifier,
    ) -> None:
        self.store = store
        self.hub = hub
        self.classifier = classifier
        self.classifications_generated = 0
        self.latest_events: dict[tuple[str, str], MarketEvent] = {}
        self.latest_classifications: dict[tuple[str, str], Classification] = {}

    async def ingest(self, event: MarketEvent) -> None:
        await self.store.append_event(event)
        self.latest_events[(event.venue, event.symbol)] = event
        await self.hub.publish(event)
        for classification in self.classifier.observe(event):
            await self.store.append_classification(classification)
            key = (classification.classifier, classification.symbol)
            self.latest_classifications[key] = classification
            await self.hub.publish(classification)
            self.classifications_generated += 1
