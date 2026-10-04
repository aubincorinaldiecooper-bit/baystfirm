from __future__ import annotations

from collections.abc import Sequence

from baystfirm.bars import MAX_CLOSED_BARS, MinuteBar, fill_minute_bars
from baystfirm.models import Candle
from baystfirm.regime import classify_window
from baystfirm.storage import ClassificationRow


def replay_momentum(symbol: str, candles: Sequence[Candle]) -> list[ClassificationRow]:
    bars = fill_minute_bars(candles)
    rows: list[ClassificationRow] = []
    for index, _bar in enumerate(bars):
        window: Sequence[MinuteBar] = bars[max(0, index - (MAX_CLOSED_BARS - 1)) : index + 1]
        classifications = classify_window(
            symbol,
            window,
            shadow=True,
            freshness_ms=0.0,
        )
        rows.extend(
            (
                classification.classifier,
                classification.symbol,
                classification.horizon_seconds,
                classification.observed_at.isoformat(),
                classification.label,
                classification.probability,
                int(classification.abstained),
                classification.freshness_ms,
                classification.classifier_version,
                int(classification.shadow),
                classification.calibration_status.value,
            )
            for classification in classifications
        )
    return rows
