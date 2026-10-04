from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from statistics import median

from baystfirm.models import Classification, EventType, Evidence, MarketEvent

STABLECOINS = frozenset({"DAI", "FDUSD", "PYUSD", "USDC", "USDE", "USDT"})
PEG_FRESHNESS = timedelta(seconds=15)
REFERENCE_FRESHNESS = timedelta(seconds=60)
MOMENTUM_HISTORY_SECONDS = 35
MAX_HISTORY_TRADES = 50_000


class MarketStateClassifier:
    def __init__(self, *, shadow: bool = True, emit_interval_seconds: float = 1.0) -> None:
        self.shadow = shadow
        self.emit_interval = timedelta(seconds=emit_interval_seconds)
        self._history: dict[str, deque[MarketEvent]] = defaultdict(deque)
        self._stablecoin_observations: dict[str, dict[tuple[str, str], MarketEvent]] = defaultdict(
            dict
        )
        self._last_emit: dict[tuple[str, str], datetime] = {}

    def observe(self, event: MarketEvent) -> list[Classification]:
        if event.event_type is not EventType.TRADE or event.price is None:
            return []
        history = self._history[event.symbol]
        history.append(event)
        cutoff = event.exchange_timestamp - timedelta(seconds=MOMENTUM_HISTORY_SECONDS)
        while history and history[0].exchange_timestamp < cutoff:
            history.popleft()
        while len(history) > MAX_HISTORY_TRADES:
            history.popleft()
        result: Classification | None = None
        if event.base_asset in STABLECOINS and (
            event.quote_asset == "USD" or event.quote_asset in STABLECOINS
        ):
            self._stablecoin_observations[event.base_asset][(event.venue, event.symbol)] = event
            key = ("stablecoin_peg", event.base_asset)
            if self._due(key, event):
                result = self._classify_stablecoin_peg(event)
        elif event.base_asset not in STABLECOINS:
            key = ("short_horizon_momentum", event.symbol)
            if self._due(key, event):
                result = self._classify_momentum(event)
        else:
            return []
        if result is None:
            return []
        self._last_emit[key] = event.exchange_timestamp
        return [result]

    def _due(self, key: tuple[str, str], event: MarketEvent) -> bool:
        last = self._last_emit.get(key)
        return last is None or event.exchange_timestamp - last >= self.emit_interval

    def _usd_price(self, asset: str, now: datetime) -> float | None:
        if asset == "USD":
            return 1.0
        cutoff = now - REFERENCE_FRESHNESS
        prices = [
            item.price
            for (_, symbol), item in self._stablecoin_observations.get(asset, {}).items()
            if symbol.endswith("-USD") and item.price is not None
            if item.received_timestamp >= cutoff
        ]
        return median(prices) if prices else None

    def _classify_stablecoin_peg(self, event: MarketEvent) -> Classification:
        now = event.received_timestamp
        cutoff = now - PEG_FRESHNESS
        implied: list[tuple[MarketEvent, float]] = []
        for item in self._stablecoin_observations[event.base_asset].values():
            if item.received_timestamp < cutoff or item.price is None:
                continue
            quote_usd = self._usd_price(item.quote_asset, now)
            if quote_usd is not None:
                implied.append((item, item.price * quote_usd))
        source_ids = [item.event_id for item, _ in implied]
        venue_count = len({item.venue for item, _ in implied})
        if venue_count < 2:
            return Classification(
                classifier="stablecoin_peg",
                classifier_version="rules-0.2.0",
                symbol=event.base_asset,
                label="insufficient_cross_venue_data",
                probability=0.5,
                abstained=True,
                horizon_seconds=30,
                observed_at=event.exchange_timestamp,
                evidence=[
                    Evidence(
                        metric="fresh_venue_count",
                        value=venue_count,
                        threshold=2,
                        source_event_ids=source_ids,
                    )
                ],
                shadow=self.shadow,
                freshness_ms=event.latency_ms,
            )
        observed_price = median(price for _, price in implied)
        deviation_bps = abs(observed_price - 1.0) * 10_000
        if deviation_bps <= 10:
            label = "pegged"
            probability = min(0.99, 0.7 + (10 - deviation_bps) / 50)
            threshold = 10.0
        elif deviation_bps <= 50:
            label = "peg_watch"
            probability = min(0.95, 0.55 + (deviation_bps - 10) / 100)
            threshold = 50.0
        else:
            label = "depegged"
            probability = min(0.99, 0.75 + (deviation_bps - 50) / 500)
            threshold = 50.0
        return Classification(
            classifier="stablecoin_peg",
            classifier_version="rules-0.2.0",
            symbol=event.base_asset,
            label=label,
            probability=probability,
            abstained=False,
            horizon_seconds=30,
            observed_at=event.exchange_timestamp,
            evidence=[
                Evidence(
                    metric="cross_venue_median_usd_price",
                    value=round(observed_price, 8),
                    source_event_ids=source_ids,
                ),
                Evidence(
                    metric="peg_deviation_bps",
                    value=round(deviation_bps, 3),
                    threshold=threshold,
                    source_event_ids=source_ids,
                ),
                Evidence(
                    metric="fresh_venue_count",
                    value=venue_count,
                    threshold=2,
                    source_event_ids=source_ids,
                ),
                Evidence(
                    metric="quote_converted_observations",
                    value=sum(item.quote_asset != "USD" for item, _ in implied),
                    source_event_ids=source_ids,
                ),
            ],
            shadow=self.shadow,
            freshness_ms=max(item.latency_ms for item, _ in implied),
        )

    def _classify_momentum(self, event: MarketEvent) -> Classification | None:
        history = self._history[event.symbol]
        if len(history) < 30:
            return None
        window_start = history[-1].exchange_timestamp - timedelta(seconds=30)
        window = [item for item in history if item.exchange_timestamp >= window_start]
        if len(window) < 30:
            return None
        if window[-1].exchange_timestamp - window[0].exchange_timestamp < timedelta(seconds=10):
            return None
        first_price = window[0].price
        last_price = window[-1].price
        if first_price is None or last_price is None or first_price <= 0:
            return None
        return_bps = (last_price / first_price - 1) * 10_000
        if return_bps >= 25:
            label = "upward_momentum"
        elif return_bps <= -25:
            label = "downward_momentum"
        else:
            label = "range_bound"
        probability = min(0.95, 0.55 + abs(return_bps) / 250)
        source_ids = [window[0].event_id, window[-1].event_id]
        return Classification(
            classifier="short_horizon_momentum",
            classifier_version="rules-0.2.1",
            symbol=event.symbol,
            label=label,
            probability=probability,
            abstained=False,
            horizon_seconds=30,
            observed_at=event.exchange_timestamp,
            evidence=[
                Evidence(
                    metric="thirty_second_return_bps",
                    value=round(return_bps, 3),
                    threshold=25.0,
                    source_event_ids=source_ids,
                ),
                Evidence(
                    metric="window_trade_count",
                    value=len(window),
                    threshold=30,
                    source_event_ids=source_ids,
                ),
            ],
            shadow=self.shadow,
            freshness_ms=event.latency_ms,
        )
