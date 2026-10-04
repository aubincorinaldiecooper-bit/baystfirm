from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse

from baystfirm.candles import (
    NATIVE_INTERVALS,
    PERPETUAL_VENUES,
    CandleNotFound,
    CandleService,
    CandleUnavailable,
    fetch_source_candles,
)
from baystfirm.classifiers import STABLECOINS, MarketStateClassifier
from baystfirm.config import Settings
from baystfirm.evaluation import PromotionGate
from baystfirm.hub import EventHub
from baystfirm.ingestion import IngestionSupervisor
from baystfirm.models import Candle
from baystfirm.pipeline import IntelligencePipeline
from baystfirm.regime import MomentumRegimeClassifier
from baystfirm.sse import sse_frames
from baystfirm.storage import EventStore
from baystfirm.track_record import TrackRecordService

logger = logging.getLogger(__name__)
SEED_VENUE_ORDER = ("coinbase", "kraken", "okx", "binanceus", "bybit")
SEED_BAR_COUNT = 1500


SourceCandleFetcher = Callable[[httpx.AsyncClient, str, str, str, int], Awaitable[list[Candle]]]


@dataclass
class Runtime:
    settings: Settings
    store: EventStore
    hub: EventHub
    pipeline: IntelligencePipeline
    ingestion: IngestionSupervisor
    momentum_classifier: MomentumRegimeClassifier
    track_record: TrackRecordService
    seed_task: asyncio.Task[None] | None = None


def _authorized(expected: str | None, authorization: str | None) -> bool:
    if expected is None:
        return True
    scheme, _, token = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), expected)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    store = EventStore(resolved.database_path)
    hub = EventHub()
    momentum_classifier = MomentumRegimeClassifier(shadow=resolved.shadow_mode)
    track_record = TrackRecordService(store)
    pipeline = IntelligencePipeline(
        store=store,
        hub=hub,
        classifiers=(
            MarketStateClassifier(shadow=resolved.shadow_mode),
            momentum_classifier,
        ),
    )
    runtime = Runtime(
        settings=resolved,
        store=store,
        hub=hub,
        pipeline=pipeline,
        ingestion=IngestionSupervisor(resolved, pipeline),
        momentum_classifier=momentum_classifier,
        track_record=track_record,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.store.open()
        runtime.seed_task = asyncio.create_task(
            _seed_minute_bars(runtime.settings, runtime.momentum_classifier),
            name="seed-momentum-minute-bars",
        )
        try:
            await runtime.ingestion.start()
            yield
        finally:
            await runtime.ingestion.stop()
            if runtime.seed_task is not None:
                runtime.seed_task.cancel()
                try:
                    await runtime.seed_task
                except asyncio.CancelledError:
                    pass
            await runtime.store.close()

    app = FastAPI(
        title="Baystfirm Crypto Intelligence",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    async def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if not _authorized(runtime.settings.api_key, authorization):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid bearer API key is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @v1.get("/candles")
    async def candles(
        venue: str,
        symbol: str,
        interval: str = Query(pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|12h|1d|1w)$"),
        limit: int = Query(default=300, ge=1, le=500),
    ) -> dict[str, Any]:
        try:
            return await CandleService(runtime.settings, runtime.store).get(
                venue, symbol, interval, limit
            )
        except CandleNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CandleUnavailable as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

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

    @v1.get("/events")
    async def events(
        symbol: str | None = None,
        limit: int = Query(default=100, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        return [
            event.model_dump(mode="json")
            async for event in runtime.store.iter_events(symbol=symbol, limit=limit)
        ]

    @v1.get("/classifications")
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

    @v1.get("/evaluation/gate")
    async def evaluation_gate() -> dict[str, object]:
        gate = PromotionGate()
        return {
            "status": "shadow",
            "promotion": "manual approval required after an eligible run",
            "runs": await runtime.store.latest_evaluation_runs(),
            "thresholds": {
                "minimum_samples": gate.minimum_samples,
                "minimum_coverage": gate.minimum_coverage,
                "minimum_accuracy": gate.minimum_accuracy,
                "minimum_macro_recall": gate.minimum_macro_recall,
                "maximum_false_alert_rate": gate.maximum_false_alert_rate,
                "maximum_brier_score": gate.maximum_brier_score,
                "maximum_expected_calibration_error": gate.maximum_ece,
                "maximum_p95_latency_ms": gate.maximum_p95_latency_ms,
            },
        }

    @v1.get("/track-record")
    async def track_record(window_hours: int = Query(default=24, ge=1, le=168)) -> dict[str, Any]:
        return await runtime.track_record.get(window_hours)

    @v1.get("/snapshot")
    async def snapshot() -> dict[str, object]:
        events = sorted(
            runtime.pipeline.latest_events.values(),
            key=lambda event: (event.symbol, event.venue),
        )
        classifications = sorted(
            runtime.pipeline.latest_classifications.values(),
            key=lambda item: (item.classifier, item.symbol, item.horizon_seconds),
        )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "shadow_mode": runtime.settings.shadow_mode,
            "enabled_venues": runtime.settings.enabled_venues,
            "symbols": runtime.settings.symbols,
            "latest_events": [event.model_dump(mode="json") for event in events],
            "latest_classifications": [item.model_dump(mode="json") for item in classifications],
        }

    @v1.get("/stream/sse")
    async def stream_sse(symbols: str | None = None) -> StreamingResponse:
        wanted = [value.strip() for value in (symbols or "").split(",") if value.strip()]
        return StreamingResponse(
            sse_frames(runtime.hub, wanted),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    app.include_router(v1)

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        if not _authorized(runtime.settings.api_key, websocket.headers.get("authorization")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            async for item in runtime.hub.subscribe():
                await websocket.send_json(item.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    return app


async def _seed_minute_bars(
    settings: Settings,
    classifier: MomentumRegimeClassifier,
    *,
    fetcher: SourceCandleFetcher = fetch_source_candles,
) -> None:
    enabled_venues = set(settings.enabled_venues)
    async with httpx.AsyncClient(timeout=10.0) as client:
        for symbol in settings.symbols:
            if symbol.split("-", maxsplit=1)[0] in STABLECOINS:
                continue
            if classifier.has_bars(symbol):
                continue
            for venue in SEED_VENUE_ORDER:
                if venue not in enabled_venues or venue not in NATIVE_INTERVALS:
                    continue
                if symbol.endswith("-PERP") and venue not in PERPETUAL_VENUES:
                    continue
                try:
                    candles = await fetcher(client, venue, symbol, "1m", SEED_BAR_COUNT)
                    now_ms = int(datetime.now(UTC).timestamp() * 1000)
                    closed = sorted(
                        {
                            candle.open_time: candle
                            for candle in candles
                            if candle.open_time + 60_000 <= now_ms
                        }.values(),
                        key=lambda candle: candle.open_time,
                    )[-SEED_BAR_COUNT:]
                    if not closed:
                        raise ValueError("candle source returned no closed 1m bars")
                    if classifier.seed(symbol, closed):
                        logger.info(
                            "seeded %s with %d closed minute bars from %s",
                            symbol,
                            len(closed),
                            venue,
                        )
                    break
                except Exception:
                    logger.warning(
                        "could not seed minute bars for %s from %s",
                        symbol,
                        venue,
                        exc_info=True,
                    )


app = create_app()
