from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from statistics import median

from baystfirm.models import Classification, EventType, Evidence, MarketEvent

STABLECOINS = frozenset({"DAI", "FDUSD", "PYUSD", "USDC", "USDE", "USDT"})


class MarketStateClassifier:
    def __init__(self, *, shadow: bool = True) -> None:
        self.shadow = shadow
        self._history: dict[str, deque[MarketEvent]] = defaultdict(lambda: deque(maxlen=500))
        self._stablecoin_venues: dict[str, dict[str, MarketEvent]] = defaultdict(dict)

    def observe(self, event: MarketEvent) -> list[Classification]:
        if event.event_type is not EventType.TRADE or event.price is None:
            return []
        self._history[event.symbol].append(event)
        outputs: list[Classification] = []
        if event.base_asset in STABLECOINS and event.quote_asset in {"USD", "USDC", "USDT"}:
            outputs.append(self._classify_stablecoin_peg(event))
        elif event.base_asset not in STABLECOINS:
            momentum = self._classify_momentum(event)
            if momentum is not None:
                outputs.append(momentum)
        return outputs

    def _classify_stablecoin_peg(self, event: MarketEvent) -> Classification:
        venue_events = self._stablecoin_venues[event.base_asset]
        venue_events[event.venue] = event
        cutoff = event.received_timestamp - timedelta(seconds=15)
        fresh = [item for item in venue_events.values() if item.received_timestamp >= cutoff]
        source_ids = [item.event_id for item in fresh]
        if len(fresh) < 2:
            return Classification(
                classifier="stablecoin_peg",
                classifier_version="rules-0.1.0",
                symbol=event.base_asset,
                label="insufficient_cross_venue_data",
                probability=0.5,
                abstained=True,
                horizon_seconds=30,
                observed_at=event.exchange_timestamp,
                evidence=[
                    Evidence(
                        metric="fresh_venue_count",
                        value=len(fresh),
                        threshold=2,
                        source_event_ids=source_ids,
                    )
                ],
                shadow=self.shadow,
                freshness_ms=event.latency_ms,
            )
        observed_price = median(item.price for item in fresh if item.price is not None)
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
            classifier_version="rules-0.1.0",
            symbol=event.base_asset,
            label=label,
            probability=probability,
            abstained=False,
            horizon_seconds=30,
            observed_at=max(item.exchange_timestamp for item in fresh),
            evidence=[
                Evidence(
                    metric="cross_venue_median_price",
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
                    value=len(fresh),
                    threshold=2,
                    source_event_ids=source_ids,
                ),
            ],
            shadow=self.shadow,
            freshness_ms=max(item.latency_ms for item in fresh),
        )

    def _classify_momentum(self, event: MarketEvent) -> Classification | None:
        history = self._history[event.symbol]
        if len(history) < 30:
            return None
        newest = history[-1]
        window_start = newest.exchange_timestamp - timedelta(seconds=30)
        window = [item for item in history if item.exchange_timestamp >= window_start]
        if len(window) < 30 or window[-1].exchange_timestamp - window[
            0
        ].exchange_timestamp < timedelta(seconds=10):
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
            classifier_version="rules-0.1.0",
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
