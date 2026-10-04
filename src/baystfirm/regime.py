from __future__ import annotations

from datetime import UTC, datetime
from math import log, sqrt
from statistics import pstdev
from uuid import UUID

from baystfirm.bars import BAR_SECONDS, MinuteBar, MinuteBarSeries
from baystfirm.classifiers import STABLECOINS
from baystfirm.models import Candle, Classification, EventType, Evidence, MarketEvent

HORIZON_BARS = (1, 5, 15, 60, 240, 1440)
CLASSIFIER_VERSION = "rules-0.1.0"


class MomentumRegimeClassifier:
    """Classifies a trailing regime, scored against the regime over the following window."""

    def __init__(self, *, shadow: bool = True) -> None:
        self.shadow = shadow
        self.bars = MinuteBarSeries()

    def observe(self, event: MarketEvent) -> list[Classification]:
        if (
            event.event_type is not EventType.TRADE
            or event.price is None
            or event.base_asset in STABLECOINS
        ):
            return []

        results: list[Classification] = []
        for closed_bar in self.bars.observe(event):
            results.extend(self._classify_closed_bar(event.symbol, closed_bar, event.latency_ms))
        return results

    def has_bars(self, symbol: str) -> bool:
        return self.bars.has_bars(symbol)

    def is_seeded(self, symbol: str) -> bool:
        return self.bars.is_seeded(symbol)

    def seed(self, symbol: str, candles: list[Candle]) -> bool:
        return self.bars.seed(symbol, candles)

    def _classify_closed_bar(
        self, symbol: str, closed_bar: MinuteBar, freshness_ms: float
    ) -> list[Classification]:
        all_closed = self.bars.closed_bars(symbol)
        close_index = next(
            (index for index, bar in enumerate(all_closed) if bar is closed_bar),
            None,
        )
        if close_index is None:
            return []
        closed = all_closed[: close_index + 1]
        end_seconds = closed_bar.open_time + BAR_SECONDS
        minute = end_seconds // BAR_SECONDS
        observed_at = closed_bar_end(closed_bar)
        results: list[Classification] = []
        for n in HORIZON_BARS:
            cadence = max(1, n // 5)
            if minute % cadence != 0 or len(closed) < n + 1:
                continue

            horizon_window = closed[-(n + 1) :]
            sigma_return_count = min(max(n, 60), len(closed) - 1)
            if sigma_return_count < n:
                continue
            sigma_window = closed[-(sigma_return_count + 1) :]
            if any(bar.close <= 0 for bar in sigma_window):
                continue
            returns = [
                log(newer.close / older.close)
                for older, newer in zip(sigma_window, sigma_window[1:], strict=False)
            ]
            if not returns:
                continue

            oldest_close = horizon_window[0].close
            newest_close = horizon_window[-1].close
            if oldest_close <= 0 or newest_close <= 0:
                continue
            lookback_return = log(newest_close / oldest_close)
            sigma1 = pstdev(returns)
            threshold = max(sigma1 * sqrt(n), 5e-4)
            filled_share = sum(bar.filled for bar in horizon_window[-n:]) / n
            score = abs(lookback_return) / threshold

            abstained = filled_share > 0.5
            if abstained:
                label = "range_bound"
                probability = 0.5
            elif lookback_return >= threshold:
                label = "upward_momentum"
                probability = min(0.9, 0.5 + 0.2 * min(score, 2))
            elif lookback_return <= -threshold:
                label = "downward_momentum"
                probability = min(0.9, 0.5 + 0.2 * min(score, 2))
            else:
                label = "range_bound"
                probability = 0.5 + 0.4 * (1 - score)

            source_event_ids = _source_event_ids(horizon_window[0], horizon_window[-1])
            evidence = [
                Evidence(
                    metric="lookback_return_bps",
                    value=lookback_return * 10_000,
                    threshold=threshold * 10_000,
                    source_event_ids=source_event_ids,
                ),
                Evidence(
                    metric="threshold_bps",
                    value=threshold * 10_000,
                    source_event_ids=source_event_ids,
                ),
                Evidence(
                    metric="realized_vol_1m_bps",
                    value=sigma1 * 10_000,
                    source_event_ids=source_event_ids,
                ),
                Evidence(
                    metric="bars_used",
                    value=n,
                    threshold=n,
                    source_event_ids=source_event_ids,
                ),
                Evidence(
                    metric="filled_bar_share",
                    value=filled_share,
                    threshold=0.5,
                    source_event_ids=source_event_ids,
                ),
            ]
            results.append(
                Classification(
                    classifier="momentum_regime",
                    classifier_version=CLASSIFIER_VERSION,
                    symbol=symbol,
                    label=label,
                    probability=probability,
                    abstained=abstained,
                    horizon_seconds=n * BAR_SECONDS,
                    observed_at=observed_at,
                    evidence=evidence,
                    shadow=self.shadow,
                    freshness_ms=freshness_ms,
                )
            )
        return results


def closed_bar_end(bar: MinuteBar) -> datetime:
    return datetime.fromtimestamp(bar.open_time + BAR_SECONDS, tz=UTC)


def _source_event_ids(oldest: MinuteBar, newest: MinuteBar) -> list[UUID]:
    if oldest.first_event_id is None or newest.last_event_id is None:
        return []
    return [oldest.first_event_id, newest.last_event_id]
