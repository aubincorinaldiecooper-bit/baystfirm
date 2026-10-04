from __future__ import annotations

from datetime import UTC, datetime

from baystfirm.models import Candle
from baystfirm.regime import MomentumRegimeClassifier
from baystfirm.signal_backtest import replay_momentum
from baystfirm.track_record import score_track_record


def make_candles(closes: list[float], *, start: datetime | None = None) -> list[Candle]:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    return [
        Candle(
            open_time=start_ms + index * 60_000,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
        for index, close in enumerate(closes)
    ]


def test_replay_momentum_matches_live_closed_bar_classification() -> None:
    candles = make_candles([100 + index / 10 for index in range(350)])
    symbol = "BTC-USD"
    classifier = MomentumRegimeClassifier(shadow=True)
    assert classifier.seed(symbol, candles)

    live = [
        classification
        for bar in classifier.bars.closed_bars(symbol)
        for classification in classifier._classify_closed_bar(symbol, bar, 0.0)
    ]
    replay = replay_momentum(symbol, candles)

    live_fields = [
        (
            item.classifier,
            item.symbol,
            item.horizon_seconds,
            item.observed_at.isoformat(),
            item.label,
            item.probability,
        )
        for item in live
    ]
    replay_fields = [(row[0], row[1], row[2], row[3], row[4], row[5]) for row in replay]
    assert replay_fields == live_fields


def test_replay_scoring_counts_hits_baseline_and_pending_for_one_and_five_minutes() -> None:
    rising = [100 * 1.001**index for index in range(120)]
    closes = rising + [rising[-1]] * 240
    candles = make_candles(closes)
    rows = replay_momentum("BTC-USD", candles)
    start = datetime.fromtimestamp(candles[0].open_time / 1000, UTC)
    end = datetime.fromtimestamp((candles[-1].open_time + 60_000) / 1000, UTC)

    groups = score_track_record(rows, start, end)
    by_horizon = {group["horizon_seconds"]: group for group in groups}

    one_minute = by_horizon[60]
    assert one_minute["scored"] == 357
    assert one_minute["hits"] == 356
    assert one_minute["baseline_hit_rate"] == 239 / 357
    assert one_minute["pending"] == 2

    five_minutes = by_horizon[300]
    assert five_minutes["scored"] == 349
    assert five_minutes["hits"] == 344
    assert five_minutes["baseline_hit_rate"] == 235 / 349
    assert five_minutes["pending"] == 6
