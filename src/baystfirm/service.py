from __future__ import annotations

import asyncio
import hmac
import logging
import math
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Annotated, Any

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

from baystfirm.bars import fill_minute_bars
from baystfirm.candles import (
    INTERVAL_SECONDS,
    NATIVE_INTERVALS,
    PERPETUAL_VENUES,
    CandleNotFound,
    CandleService,
    CandleUnavailable,
    aggregate_candles,
    fetch_source_candles,
    source_interval,
    validate_candle_request,
)
from baystfirm.classifiers import STABLECOINS, MarketStateClassifier
from baystfirm.config import Settings
from baystfirm.evaluation import PromotionGate
from baystfirm.hub import EventHub
from baystfirm.indicators import IndicatorSpec, compute_indicator, parse_indicator
from baystfirm.ingestion import IngestionSupervisor
from baystfirm.models import Candle
from baystfirm.news import (
    NEWS_NOTE,
    MarketNewsObserver,
    NewsKind,
    NewsService,
    normalize_news_symbol,
    parse_tickers,
)
from baystfirm.pipeline import IntelligencePipeline
from baystfirm.regime import MomentumRegimeClassifier
from baystfirm.signal_backtest import replay_momentum
from baystfirm.solana_tokens import (
    SOURCE_NAMES,
    NotTokenMint,
    SolanaTokenClient,
    SolanaTokenEngine,
    SourceError,
    market_value,
    select_main_pool,
    validate_mint,
)
from baystfirm.sse import sse_frames
from baystfirm.storage import ClassificationRow, EventStore
from baystfirm.strategies import (
    BATCH_NOTE,
    BacktestRequest,
    BacktestResult,
    BatchBacktestRequest,
    BatchInstrumentResult,
    backtest,
    summarize_batch,
)
from baystfirm.track_record import TrackRecordService, score_track_record

logger = logging.getLogger(__name__)
SEED_VENUE_ORDER = ("coinbase", "kraken", "okx", "binanceus", "bybit")
SEED_BAR_COUNT = 1500
BACKTEST_BAR_COUNT = 10_080
SIGNAL_BACKTEST_REFRESH_SECONDS = 6 * 60 * 60
STORAGE_RETENTION_INTERVAL_SECONDS = 10 * 60
BACKTEST_NOTE = (
    "Replayed on 1-minute candles from one exchange per instrument; live calls use trades merged "
    "from all venues, so results can differ. Backtests usually look better than live results. "
    "Overlapping predictions are scored individually, so intervals are optimistic. Probabilities, "
    "not investment advice."
)
BACKTEST_CACHE_TTL_SECONDS = 60.0
BACKTEST_CACHE_MAX_ENTRIES = 32
SOLANA_TOKEN_REFRESH_SECONDS = 60
SOLANA_TOKEN_WORKERS = 4
SOLANA_SEARCH_CACHE_TTL_SECONDS = 30
SOLANA_SEARCH_CACHE_MAX_ENTRIES = 256
SOLANA_PRICE_CACHE_TTL_SECONDS = 5
SOLANA_PRICE_CACHE_MAX_ENTRIES = 256
SOLANA_POOL_CACHE_TTL_SECONDS = 5 * 60
SOLANA_POOL_CACHE_MAX_ENTRIES = 256
SOLANA_CANDLE_CACHE_TTL_SECONDS = 60
SOLANA_CANDLE_CACHE_MAX_ENTRIES = 256
SOLANA_CANDLE_INTERVALS = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "4h": ("hour", 4),
    "1d": ("day", 1),
}
SOLANA_TOKENS_NOTE = (
    "Facts read from public sources at the times shown. Our own checks come first; RugCheck is "
    "shown as a second opinion. There is no overall safety verdict. Probabilities and facts, "
    "not investment advice."
)
SOLANA_SEARCH_NOTE = (
    "Results are pools found by DEX Screener search; they are not an endorsement of any token."
)


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
    backtest_candles: dict[tuple[str, str, str, int], tuple[float, list[Candle]]] = field(
        default_factory=dict
    )
    seed_task: asyncio.Task[None] | None = None
    news_service: NewsService | None = None
    news_task: asyncio.Task[None] | None = None
    signal_backtest: dict[str, Any] | None = None
    signal_backtest_task: asyncio.Task[None] | None = None
    retention_task: asyncio.Task[None] | None = None
    solana_tokens: SolanaTokenEngine | None = None
    solana_tokens_task: asyncio.Task[None] | None = None
    solana_http_client: httpx.AsyncClient | None = None
    solana_search_cache: OrderedDict[str, tuple[float, str, list[dict[str, Any]]]] = field(
        default_factory=OrderedDict
    )
    solana_pool_cache: OrderedDict[str, tuple[float, dict[str, Any] | None]] = field(
        default_factory=OrderedDict
    )
    solana_price_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )
    solana_candle_cache: OrderedDict[tuple[str, str, int], tuple[float, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )


def _authorized(expected: str | None, authorization: str | None) -> bool:
    if expected is None:
        return True
    scheme, _, token = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), expected)


def _parse_requested_indicators(indicator: list[str] | None) -> list[IndicatorSpec]:
    if not indicator:
        return []
    if len(indicator) > 6:
        raise HTTPException(
            status_code=422,
            detail="At most 6 indicator specs may be requested.",
        )
    try:
        return [parse_indicator(value) for value in indicator]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _apply_requested_indicators(
    result: dict[str, Any],
    specs: list[IndicatorSpec],
) -> dict[str, Any]:
    if not specs:
        return result
    candles = [Candle.model_validate(item) for item in result["candles"]]
    result["indicators"] = {spec.key: compute_indicator(spec, candles) for spec in specs}
    return result


def _parse_geckoterminal_candles(body: Any, limit: int) -> list[dict[str, Any]]:
    data = body.get("data") if isinstance(body, dict) else None
    attributes = data.get("attributes") if isinstance(data, dict) else None
    rows = attributes.get("ohlcv_list") if isinstance(attributes, dict) else None
    if not isinstance(rows, list):
        raise ValueError("GeckoTerminal did not return OHLCV rows.")
    candles: list[dict[str, Any]] = []
    for row in reversed(rows[:limit]):
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("GeckoTerminal returned an invalid OHLCV row.")
        try:
            timestamp, open_value, high, low, close, volume = (float(value) for value in row[:6])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("GeckoTerminal returned an invalid OHLCV row.") from error
        values = (timestamp, open_value, high, low, close, volume)
        if not all(math.isfinite(value) for value in values) or not math.isfinite(timestamp * 1000):
            raise ValueError("GeckoTerminal returned an invalid OHLCV row.")
        candles.append(
            Candle(
                open_time=int(timestamp * 1000),
                open=open_value,
                high=high,
                low=low,
                close=close,
                volume=volume,
            ).model_dump(mode="json")
        )
    return candles


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    store = EventStore(resolved.database_path)
    hub = EventHub()
    momentum_classifier = MomentumRegimeClassifier(shadow=resolved.shadow_mode)
    track_record_service = TrackRecordService(store)
    pipeline = IntelligencePipeline(
        store=store,
        hub=hub,
        classifiers=(
            MarketStateClassifier(shadow=resolved.shadow_mode),
            momentum_classifier,
        ),
        news=MarketNewsObserver(),
    )
    runtime = Runtime(
        settings=resolved,
        store=store,
        hub=hub,
        pipeline=pipeline,
        ingestion=IngestionSupervisor(resolved, pipeline),
        momentum_classifier=momentum_classifier,
        track_record=track_record_service,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.store.open()
        runtime.retention_task = asyncio.create_task(
            _storage_retention_loop(runtime),
            name="storage-retention",
        )
        runtime.news_service = NewsService(runtime.store, runtime.settings.sec_user_agent)
        if runtime.settings.news_enabled:
            runtime.news_task = asyncio.create_task(
                runtime.news_service.poll_official_forever(),
                name="official-news-poller",
            )
        runtime.seed_task = asyncio.create_task(
            _seed_minute_bars(runtime.settings, runtime.momentum_classifier),
            name="seed-momentum-minute-bars",
        )
        runtime.signal_backtest_task = asyncio.create_task(
            _signal_backtest_loop(runtime),
            name="momentum-signal-backtest",
        )
        runtime.solana_http_client = httpx.AsyncClient(timeout=10.0)
        runtime.solana_tokens = SolanaTokenEngine(
            SolanaTokenClient(runtime.settings, runtime.solana_http_client),
            news_sink=runtime.store.append_news,
        )
        if runtime.settings.solana_tokens_enabled:
            runtime.solana_tokens_task = asyncio.create_task(
                _solana_discovery_loop(runtime),
                name="solana-token-discovery",
            )
        try:
            await runtime.ingestion.start()
            yield
        finally:
            await runtime.ingestion.stop()
            for task in (
                runtime.seed_task,
                runtime.news_task,
                runtime.signal_backtest_task,
                runtime.retention_task,
                runtime.solana_tokens_task,
            ):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if runtime.solana_http_client is not None:
                await runtime.solana_http_client.aclose()
            if runtime.news_service is not None:
                await runtime.news_service.close()
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

    def solana_checker() -> SolanaTokenClient:
        if runtime.solana_tokens is not None:
            return runtime.solana_tokens.checker
        if runtime.solana_http_client is not None:
            return SolanaTokenClient(runtime.settings, runtime.solana_http_client)
        raise HTTPException(status_code=503, detail="Solana data client is unavailable.")

    @v1.get("/candles")
    async def candles(
        venue: str,
        symbol: str,
        interval: str = Query(pattern="^(1m|3m|5m|15m|30m|1h|2h|4h|6h|12h|1d|1w)$"),
        limit: int = Query(default=300, ge=1, le=500),
        indicator: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        specs = _parse_requested_indicators(indicator)
        try:
            result = await CandleService(runtime.settings, runtime.store).get(
                venue, symbol, interval, limit
            )
        except CandleNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except CandleUnavailable as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return _apply_requested_indicators(result, specs)

    async def load_backtest_candles(
        venue: str, symbol: str, interval: str, bars: int
    ) -> list[Candle]:
        try:
            venue, symbol = validate_candle_request(runtime.settings, venue, symbol, interval)
        except CandleNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        cache_key = (venue, symbol, interval, bars)
        cache_entry = runtime.backtest_candles.get(cache_key)
        if cache_entry is not None and monotonic() - cache_entry[0] < BACKTEST_CACHE_TTL_SECONDS:
            candles = cache_entry[1]
        else:
            try:
                source_name = source_interval(venue, interval)
                source_seconds = INTERVAL_SECONDS[source_name]
                target_seconds = INTERVAL_SECONDS[interval]
                source_limit = math.ceil(bars * target_seconds / source_seconds)
                async with httpx.AsyncClient(timeout=10.0) as client:
                    source_candles = await fetch_source_candles(
                        client,
                        venue,
                        symbol,
                        source_name,
                        source_limit,
                    )
                unique = {candle.open_time: candle for candle in source_candles}
                normalized = sorted(unique.values(), key=lambda candle: candle.open_time)
                now_ms = int(datetime.now(UTC).timestamp() * 1000)
                fetched = (
                    aggregate_candles(
                        normalized,
                        source_name,
                        interval,
                        now_ms=now_ms,
                    )
                    if source_name != interval
                    else normalized
                )
                interval_ms = target_seconds * 1000
                candles = [
                    candle for candle in fetched if candle.open_time + interval_ms <= now_ms
                ][-bars:]
            except Exception as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
            inserted_at = monotonic()
            for cached_key, (cached_at, _) in list(runtime.backtest_candles.items()):
                if inserted_at - cached_at >= BACKTEST_CACHE_TTL_SECONDS:
                    del runtime.backtest_candles[cached_key]
            runtime.backtest_candles[cache_key] = (inserted_at, candles)
            while len(runtime.backtest_candles) > BACKTEST_CACHE_MAX_ENTRIES:
                oldest_key = min(
                    runtime.backtest_candles,
                    key=lambda key: runtime.backtest_candles[key][0],
                )
                del runtime.backtest_candles[oldest_key]

        return candles

    @v1.post("/backtest")
    async def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        rule = request.rule
        candles = await load_backtest_candles(
            rule.venue,
            rule.symbol,
            rule.interval,
            request.bars,
        )
        result = await asyncio.to_thread(
            backtest,
            rule,
            candles,
            request.fee_bps,
            slippage_bps=request.slippage_bps,
            holdout_pct=request.holdout_pct,
        )
        response = result.model_dump(mode="json")
        response["bars_requested"] = request.bars
        response["truncated"] = len(candles) < request.bars
        return response

    @v1.post("/backtest/batch")
    async def run_batch_backtest(request: BatchBacktestRequest) -> dict[str, Any]:
        rule = request.rule
        instruments = [(rule.venue, rule.symbol)] + [
            (instrument.venue, instrument.symbol) for instrument in request.also
        ]
        validated: list[tuple[str, str, str | None]] = []
        seen: set[tuple[str, str]] = set()
        for raw_venue, raw_symbol in instruments:
            normalized_key = (raw_venue.lower(), raw_symbol.upper())
            if normalized_key in seen:
                raise HTTPException(status_code=422, detail="Duplicate instrument.")
            seen.add(normalized_key)
            try:
                venue, symbol = validate_candle_request(
                    runtime.settings,
                    raw_venue,
                    raw_symbol,
                    rule.interval,
                )
            except CandleNotFound as error:
                validated.append((*normalized_key, str(error)))
            else:
                validated.append((venue, symbol, None))

        semaphore = asyncio.Semaphore(4)

        async def run_instrument(
            venue: str, symbol: str, validation_error: str | None
        ) -> tuple[BatchInstrumentResult, BacktestResult | None]:
            if validation_error is not None:
                return (
                    BatchInstrumentResult(
                        venue=venue,
                        symbol=symbol,
                        error=validation_error,
                        bars_tested=None,
                        truncated=None,
                        stats=None,
                        baseline=None,
                        buy_and_hold=None,
                        total_return_pct=None,
                        max_drawdown_pct=None,
                        recent=None,
                    ),
                    None,
                )

            async with semaphore:
                try:
                    candles = await load_backtest_candles(
                        venue,
                        symbol,
                        rule.interval,
                        request.bars,
                    )
                except HTTPException as error:
                    return (
                        BatchInstrumentResult(
                            venue=venue,
                            symbol=symbol,
                            error=str(error.detail),
                            bars_tested=None,
                            truncated=None,
                            stats=None,
                            baseline=None,
                            buy_and_hold=None,
                            total_return_pct=None,
                            max_drawdown_pct=None,
                            recent=None,
                        ),
                        None,
                    )

                instrument_rule = rule.model_copy(update={"venue": venue, "symbol": symbol})
                result = await asyncio.to_thread(
                    backtest,
                    instrument_rule,
                    candles,
                    request.fee_bps,
                    slippage_bps=request.slippage_bps,
                    holdout_pct=request.holdout_pct,
                )
                return (
                    BatchInstrumentResult(
                        venue=venue,
                        symbol=symbol,
                        error=None,
                        bars_tested=result.bars_tested,
                        truncated=len(candles) < request.bars,
                        stats=result.stats,
                        baseline=result.baseline,
                        buy_and_hold=result.buy_and_hold,
                        total_return_pct=result.equity.total_return_pct,
                        max_drawdown_pct=result.equity.max_drawdown_pct,
                        recent=result.holdout.recent if result.holdout is not None else None,
                    ),
                    result,
                )

        outcomes = await asyncio.gather(
            *(run_instrument(venue, symbol, error) for venue, symbol, error in validated)
        )
        results = [outcome[0] for outcome in outcomes]
        successful = [outcome[1] for outcome in outcomes if outcome[1] is not None]
        summary = summarize_batch(successful).model_copy(
            update={"instruments_failed": sum(result.error is not None for result in results)}
        )
        return {
            "results": [result.model_dump(mode="json") for result in results],
            "summary": summary.model_dump(mode="json"),
            "note": BATCH_NOTE,
        }

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

    @v1.get("/news")
    async def news(
        symbol: str | None = None,
        kind: Annotated[list[NewsKind] | None, Query()] = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        items = await runtime.store.iter_news(
            symbol=normalize_news_symbol(symbol),
            kinds=set(kind) if kind is not None else None,
            limit=limit,
        )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "items": [item.model_dump(mode="json") for item in items],
            "sources": runtime.news_service.official_sources() if runtime.news_service else [],
            "note": NEWS_NOTE,
        }

    @v1.get("/news/filings")
    async def news_filings(
        tickers: str = Query(min_length=1),
        limit: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, Any]:
        try:
            requested = parse_tickers(tickers)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        filings: dict[str, Any] = {}
        notes: list[str] = []
        if runtime.news_service is not None:
            for ticker in requested:
                ticker_items, ticker_note = await runtime.news_service.filings_for_ticker(
                    ticker, limit
                )
                for item in ticker_items:
                    filings[item.id] = item
                if ticker_note is not None:
                    notes.append(ticker_note)
        ordered = sorted(
            filings.values(),
            key=lambda item: item.published_at,
            reverse=True,
        )[:limit]
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "items": [item.model_dump(mode="json") for item in ordered],
            "notes": notes,
            "source": runtime.news_service.sec_source() if runtime.news_service else {},
            "note": NEWS_NOTE,
        }

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
    async def track_record(window_hours: int = Query(default=24, ge=1, le=48)) -> dict[str, Any]:
        return await runtime.track_record.get(window_hours)

    @v1.get("/track-record/backtest")
    async def track_record_backtest() -> dict[str, Any]:
        if runtime.signal_backtest is not None:
            return runtime.signal_backtest
        return {
            "status": "computing",
            "computed_at": None,
            "span_start": None,
            "span_end": None,
            "sources": [],
            "groups": [],
            "note": BACKTEST_NOTE,
        }

    @v1.get("/solana/search")
    async def solana_search(q: str) -> dict[str, Any]:
        query = q.strip()
        if not 1 <= len(query) <= 32:
            raise HTTPException(status_code=422, detail="Search query must be 1–32 characters.")
        cache_key = query.casefold()
        cached = runtime.solana_search_cache.get(cache_key)
        now = monotonic()
        if cached is not None and now - cached[0] < SOLANA_SEARCH_CACHE_TTL_SECONDS:
            runtime.solana_search_cache.move_to_end(cache_key)
            return {
                "query": query,
                "tokens": cached[2],
                "source": "dexscreener",
                "fetched_at": cached[1],
                "note": SOLANA_SEARCH_NOTE,
            }
        try:
            tokens = await solana_checker().search_tokens(query)
        except SourceError as error:
            raise HTTPException(status_code=502, detail=error.detail) from error
        fetched_at = datetime.now(UTC).isoformat()
        runtime.solana_search_cache[cache_key] = (monotonic(), fetched_at, tokens)
        runtime.solana_search_cache.move_to_end(cache_key)
        while len(runtime.solana_search_cache) > SOLANA_SEARCH_CACHE_MAX_ENTRIES:
            runtime.solana_search_cache.popitem(last=False)
        return {
            "query": query,
            "tokens": tokens,
            "source": "dexscreener",
            "fetched_at": fetched_at,
            "note": SOLANA_SEARCH_NOTE,
        }

    @v1.get("/solana/tokens/new")
    async def solana_new_tokens(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        engine = runtime.solana_tokens
        if engine is None:
            sources = [
                {"name": source, "fetched_at": None, "ok": False, "error": None}
                for source in SOURCE_NAMES
            ]
            return {
                "status": "warming",
                "updated_at": None,
                "sources": sources,
                "tokens": [],
                "note": SOLANA_TOKENS_NOTE,
            }
        return {**engine.response(limit), "note": SOLANA_TOKENS_NOTE}

    @v1.get("/solana/tokens/{mint}/price")
    async def solana_token_price(mint: str) -> dict[str, Any]:
        try:
            validate_mint(mint)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        cached = runtime.solana_price_cache.get(mint)
        now = monotonic()
        if cached is not None and now - cached[0] < SOLANA_PRICE_CACHE_TTL_SECONDS:
            runtime.solana_price_cache.move_to_end(mint)
            result = dict(cached[1])
            result["stale"] = False
            return result

        try:
            pairs = await solana_checker().get_token_pairs(mint)
        except SourceError as error:
            if cached is None:
                raise HTTPException(status_code=502, detail=error.detail) from error
            runtime.solana_price_cache.move_to_end(mint)
            result = dict(cached[1])
            result["stale"] = True
            return result

        main = select_main_pool(pairs)
        if main is None:
            raise HTTPException(
                status_code=404,
                detail="No trading pool found for this token",
            )

        fetched_at = datetime.now(UTC).isoformat()
        result = {
            "mint": mint,
            "source": "dexscreener",
            "fetched_at": fetched_at,
            "stale": False,
            "market": market_value(pairs, pools_checked_at=fetched_at),
        }
        cached_at = monotonic()
        runtime.solana_price_cache[mint] = (cached_at, result)
        runtime.solana_price_cache.move_to_end(mint)
        while len(runtime.solana_price_cache) > SOLANA_PRICE_CACHE_MAX_ENTRIES:
            runtime.solana_price_cache.popitem(last=False)

        runtime.solana_pool_cache[mint] = (cached_at, main)
        runtime.solana_pool_cache.move_to_end(mint)
        while len(runtime.solana_pool_cache) > SOLANA_POOL_CACHE_MAX_ENTRIES:
            runtime.solana_pool_cache.popitem(last=False)
        return result

    @v1.get("/solana/tokens/{mint}/candles")
    async def solana_token_candles(
        mint: str,
        interval: str = Query(pattern="^(1m|5m|15m|1h|4h|1d)$"),
        limit: int = Query(default=300, ge=1, le=500),
        indicator: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        try:
            validate_mint(mint)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        specs = _parse_requested_indicators(indicator)
        checker = solana_checker()
        pool_entry = runtime.solana_pool_cache.get(mint)
        cached_main = pool_entry[1] if pool_entry is not None else None
        now = monotonic()
        if pool_entry is not None and now - pool_entry[0] < SOLANA_POOL_CACHE_TTL_SECONDS:
            runtime.solana_pool_cache.move_to_end(mint)
            main_pool = cached_main
        else:
            try:
                pairs = await checker.get_token_pairs(mint)
            except SourceError as error:
                if cached_main is None:
                    raise HTTPException(status_code=502, detail=error.detail) from error
                main_pool = cached_main
            else:
                main_pool = select_main_pool(pairs)
                runtime.solana_pool_cache[mint] = (monotonic(), main_pool)
                runtime.solana_pool_cache.move_to_end(mint)
                while len(runtime.solana_pool_cache) > SOLANA_POOL_CACHE_MAX_ENTRIES:
                    runtime.solana_pool_cache.popitem(last=False)
        if main_pool is None:
            raise HTTPException(
                status_code=404,
                detail="No trading pool found for this token",
            )
        pool_address = main_pool.get("pairAddress")
        if not isinstance(pool_address, str) or not pool_address:
            raise HTTPException(
                status_code=404,
                detail="No trading pool found for this token",
            )
        cache_key = (pool_address, interval, limit)
        cached_candles = runtime.solana_candle_cache.get(cache_key)
        now = monotonic()
        if cached_candles is not None and now - cached_candles[0] < SOLANA_CANDLE_CACHE_TTL_SECONDS:
            runtime.solana_candle_cache.move_to_end(cache_key)
            result = dict(cached_candles[1])
            result["stale"] = False
            return _apply_requested_indicators(result, specs)

        timeframe, aggregate = SOLANA_CANDLE_INTERVALS[interval]
        try:
            body = await checker.get_geckoterminal_ohlcv(
                pool_address,
                mint,
                timeframe,
                aggregate,
                limit,
                priority="interactive",
            )
            candle_rows = _parse_geckoterminal_candles(body, limit)
        except (SourceError, ValueError) as error:
            if cached_candles is not None:
                runtime.solana_candle_cache.move_to_end(cache_key)
                result = dict(cached_candles[1])
                result["stale"] = True
                return _apply_requested_indicators(result, specs)
            raise HTTPException(
                status_code=502,
                detail="GeckoTerminal is rate-limited or unavailable; try again shortly",
            ) from error

        base_token = main_pool.get("baseToken")
        base_token = base_token if isinstance(base_token, dict) else {}
        symbol = base_token.get("symbol")
        result = {
            "venue": "geckoterminal",
            "symbol": symbol if isinstance(symbol, str) and symbol else mint,
            "interval": interval,
            "source_url_template": "https://www.geckoterminal.com/solana/pools/{pool}",
            "fetched_at": datetime.now(UTC).isoformat(),
            "aggregated_from": None,
            "candles": candle_rows,
            "stale": False,
            "truncated": len(candle_rows) < limit,
            "mint": mint,
            "pool_address": pool_address,
            "dex_id": str(main_pool.get("dexId") or "unknown"),
            "price_currency": "usd",
        }
        runtime.solana_candle_cache[cache_key] = (monotonic(), result)
        runtime.solana_candle_cache.move_to_end(cache_key)
        while len(runtime.solana_candle_cache) > SOLANA_CANDLE_CACHE_MAX_ENTRIES:
            runtime.solana_candle_cache.popitem(last=False)
        return _apply_requested_indicators(dict(result), specs)

    @v1.get("/solana/tokens/{mint}")
    async def solana_token(mint: str) -> dict[str, Any]:
        try:
            validate_mint(mint)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        engine = runtime.solana_tokens
        if engine is not None:
            try:
                card = await engine.get_card(mint)
            except NotTokenMint as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                on_demand = SolanaTokenEngine(
                    SolanaTokenClient(runtime.settings, client),
                    news_sink=runtime.store.append_news,
                )
                try:
                    card = await on_demand.get_card(mint)
                except NotTokenMint as error:
                    raise HTTPException(status_code=404, detail=str(error)) from error
        return card.model_dump(mode="json")

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
            if classifier.is_seeded(symbol):
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


async def _signal_backtest_loop(
    runtime: Runtime,
    *,
    fetcher: SourceCandleFetcher | None = None,
) -> None:
    while True:
        try:
            runtime.signal_backtest = await _compute_signal_backtest(runtime, fetcher=fetcher)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("momentum signal backtest computation failed")
        await asyncio.sleep(SIGNAL_BACKTEST_REFRESH_SECONDS)


async def _storage_retention_loop(runtime: Runtime) -> None:
    while True:
        now = datetime.now(UTC)
        try:
            deleted = await runtime.store.prune_events(
                now - timedelta(hours=runtime.settings.event_retention_hours)
            )
            logger.info("Storage retention pruned %d market events", deleted)
        except Exception:
            logger.exception("Storage retention failed to prune market events")
        try:
            deleted = await runtime.store.prune_classifications(
                now - timedelta(days=runtime.settings.classification_retention_days)
            )
            logger.info("Storage retention pruned %d classifications", deleted)
        except Exception:
            logger.exception("Storage retention failed to prune classifications")
        await asyncio.sleep(STORAGE_RETENTION_INTERVAL_SECONDS)


async def _solana_discovery_cycle(engine: SolanaTokenEngine) -> None:
    engine.schedule_retries()
    await engine.discover_once()
    await engine.refresh_markets_once()
    engine.updated_at = datetime.now(UTC).isoformat()
    engine.ready = True


async def _solana_discovery_loop(runtime: Runtime) -> None:
    engine = runtime.solana_tokens
    if engine is None:
        return
    workers = [
        asyncio.create_task(engine.worker(), name=f"solana-token-worker-{index}")
        for index in range(SOLANA_TOKEN_WORKERS)
    ]
    try:
        while True:
            try:
                await _solana_discovery_cycle(engine)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Solana token discovery cycle failed")
            await asyncio.sleep(SOLANA_TOKEN_REFRESH_SECONDS)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


async def _compute_signal_backtest(
    runtime: Runtime,
    *,
    fetcher: SourceCandleFetcher | None = None,
) -> dict[str, Any]:
    fetcher = fetch_source_candles if fetcher is None else fetcher
    enabled_venues = set(runtime.settings.enabled_venues)
    sources: list[dict[str, str | int]] = []
    rows: list[ClassificationRow] = []
    span_starts: list[datetime] = []
    span_ends: list[datetime] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for symbol in runtime.settings.symbols:
            if symbol.split("-", maxsplit=1)[0] in STABLECOINS:
                continue
            found_source = False
            for venue in SEED_VENUE_ORDER:
                if venue not in enabled_venues or venue not in NATIVE_INTERVALS:
                    continue
                if symbol.endswith("-PERP") and venue not in PERPETUAL_VENUES:
                    continue
                try:
                    fetched = await fetcher(
                        client,
                        venue,
                        symbol,
                        "1m",
                        BACKTEST_BAR_COUNT,
                    )
                    now_ms = int(datetime.now(UTC).timestamp() * 1000)
                    closed = sorted(
                        {
                            candle.open_time: candle
                            for candle in fetched
                            if candle.open_time + 60_000 <= now_ms
                        }.values(),
                        key=lambda candle: candle.open_time,
                    )[-BACKTEST_BAR_COUNT:]
                    if not closed:
                        raise ValueError("candle source returned no closed 1m bars")

                    filled = fill_minute_bars(closed)
                    rows.extend(await asyncio.to_thread(replay_momentum, symbol, closed))
                    sources.append({"symbol": symbol, "venue": venue, "bars": len(closed)})
                    span_starts.append(datetime.fromtimestamp(filled[0].open_time, UTC))
                    span_ends.append(datetime.fromtimestamp(filled[-1].open_time + 60, UTC))
                    logger.info(
                        "replayed momentum backtest for %s with %d closed bars from %s",
                        symbol,
                        len(closed),
                        venue,
                    )
                    found_source = True
                    break
                except Exception:
                    logger.warning(
                        "could not replay momentum backtest for %s from %s",
                        symbol,
                        venue,
                        exc_info=True,
                    )
            if not found_source:
                logger.warning("skipping momentum backtest for %s after venue failures", symbol)

    if span_starts:
        span_start = min(span_starts)
        span_end = max(span_ends)
        groups = score_track_record(rows, span_start, span_end)
        groups = [group for group in groups if group["classifier"] == "momentum_regime"]
    else:
        span_start = None
        span_end = None
        groups = []

    return {
        "status": "ready",
        "computed_at": datetime.now(UTC).isoformat(),
        "span_start": span_start.isoformat() if span_start is not None else None,
        "span_end": span_end.isoformat() if span_end is not None else None,
        "sources": sources,
        "groups": groups,
        "note": BACKTEST_NOTE,
    }


app = create_app()
