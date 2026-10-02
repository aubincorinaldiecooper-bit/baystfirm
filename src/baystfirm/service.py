from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.config import Settings
from baystfirm.evaluation import PromotionGate
from baystfirm.hub import EventHub
from baystfirm.ingestion import IngestionSupervisor
from baystfirm.pipeline import IntelligencePipeline
from baystfirm.storage import EventStore


@dataclass
class Runtime:
    settings: Settings
    store: EventStore
    hub: EventHub
    pipeline: IntelligencePipeline
    ingestion: IngestionSupervisor


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    store = EventStore(resolved.database_path)
    hub = EventHub()
    pipeline = IntelligencePipeline(
        store=store,
        hub=hub,
        classifier=MarketStateClassifier(shadow=resolved.shadow_mode),
    )
    runtime = Runtime(
        settings=resolved,
        store=store,
        hub=hub,
        pipeline=pipeline,
        ingestion=IngestionSupervisor(resolved, pipeline),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.store.open()
        await runtime.ingestion.start()
        try:
            yield
        finally:
            await runtime.ingestion.stop()
            await runtime.store.close()

    app = FastAPI(
        title="Baystfirm Crypto Intelligence",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "time": datetime.now(UTC).isoformat(),
            "shadow_mode": runtime.settings.shadow_mode,
            "enabled_venues": runtime.settings.enabled_venues,
            "subscribers": runtime.hub.subscriber_count,
            "dropped_events": runtime.hub.dropped_events,
            "stored_events": await runtime.store.event_count(),
            "stored_classifications": await runtime.store.classification_count(),
            "events_received": runtime.ingestion.events_received,
            "classifications_generated": runtime.pipeline.classifications_generated,
            "last_event_by_venue": {
                venue: event.received_timestamp.isoformat()
                for venue, event in runtime.ingestion.last_event_by_venue.items()
            },
        }

    @app.get("/v1/events")
    async def events(
        symbol: str | None = None,
        limit: int = Query(default=100, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        return [
            event.model_dump(mode="json")
            async for event in runtime.store.iter_events(symbol=symbol, limit=limit)
        ]

    @app.get("/v1/classifications")
    async def classifications(
        symbol: str | None = None,
        classifier: str | None = None,
        limit: int = Query(default=100, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            async for item in runtime.store.iter_classifications(
                symbol=symbol,
                classifier=classifier,
                limit=limit,
            )
        ]

    @app.get("/v1/evaluation/gate")
    async def evaluation_gate() -> dict[str, object]:
        gate = PromotionGate()
        return {
            "status": "shadow",
            "promotion_eligible": False,
            "reason": "no validated evaluation run has been registered",
            "thresholds": {
                "minimum_samples": gate.minimum_samples,
                "minimum_coverage": gate.minimum_coverage,
                "minimum_accuracy": gate.minimum_accuracy,
                "maximum_false_alert_rate": gate.maximum_false_alert_rate,
                "maximum_brier_score": gate.maximum_brier_score,
                "maximum_expected_calibration_error": gate.maximum_ece,
                "maximum_p95_latency_ms": gate.maximum_p95_latency_ms,
            },
        }

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async for item in runtime.hub.subscribe():
                await websocket.send_json(item.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    return app


app = create_app()
