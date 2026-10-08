"""Baystfirm shadow chart-state probabilities for fixed momentum labels.

Labels follow short_horizon_momentum: over a 30-second window,
returns at or above +25 bps are upward_momentum, at or below -25 bps are
downward_momentum, and intermediate returns are range_bound.
"""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CHART_LABELS = ("upward_momentum", "downward_momentum", "range_bound")
NORMAL_LABEL = "range_bound"
CLASSIFIER = "baystfirm_chart_momentum"
WINDOW_SECONDS = 300
HORIZON_SECONDS = 30
MOMENTUM_BPS = 25.0
MIN_TRADES = 30
MIN_SPAN_SECONDS = 10
Y_RANGE_BPS = 150.0
IMAGE_SIZE = 448
VOLUME_BUCKET_SECONDS = 5
_MILLISECONDS = 1000
_PRICE_HEIGHT = IMAGE_SIZE * 3 // 4
_VOLUME_HEIGHT = IMAGE_SIZE - _PRICE_HEIGHT


@dataclass(frozen=True)
class Tick:
    ts_ms: int
    price: float
    size: float


@dataclass(frozen=True)
class ChartSample:
    symbol: str
    anchor_ms: int
    split: str
    label: str
    outcome_bps: float
    persistence_label: str | None
    persistence_bps: float | None
    evidence: dict[str, int | float | None]


def ticks_between(ticks: Sequence[Tick], start_ms: int, end_ms: int) -> Sequence[Tick]:
    """Ticks with start_ms <= ts_ms <= end_ms; ticks sorted by ts_ms (bisect)."""
    start = bisect_left(ticks, start_ms, key=lambda tick: tick.ts_ms)
    end = bisect_right(ticks, end_ms, key=lambda tick: tick.ts_ms)
    return ticks[start:end]


def momentum_label(ticks: Sequence[Tick], end_ms: int) -> tuple[str, float] | None:
    """Baystfirm short_horizon_momentum label for the 30 s window ending at end_ms."""
    window = ticks_between(ticks, end_ms - HORIZON_SECONDS * _MILLISECONDS, end_ms)
    if len(window) < MIN_TRADES:
        return None
    if window[-1].ts_ms - window[0].ts_ms < MIN_SPAN_SECONDS * _MILLISECONDS:
        return None
    first, last = window[0].price, window[-1].price
    if first <= 0:
        return None
    bps = (last / first - 1) * 10_000
    if bps >= MOMENTUM_BPS or abs(bps - MOMENTUM_BPS) <= 1e-9:
        return "upward_momentum", bps
    if bps <= -MOMENTUM_BPS or abs(bps + MOMENTUM_BPS) <= 1e-9:
        return "downward_momentum", bps
    return "range_bound", bps


def render_chart(ticks: Sequence[Tick], anchor_ms: int) -> Image.Image:
    """Render only observed prices and volumes in the five minutes ending at anchor_ms."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(image)
    start_ms = anchor_ms - WINDOW_SECONDS * _MILLISECONDS
    window = ticks_between(ticks, start_ms, anchor_ms)
    if not window:
        return image

    last_price = window[-1].price
    if last_price <= 0:
        return image
    for level in (0, -25, 25, -100, 100):
        y = _price_y(level)
        draw.line((0, y, IMAGE_SIZE - 1, y), fill=(220, 220, 220), width=1)

    prices_by_column: dict[int, float] = {}
    volume_by_bucket = [0.0] * (WINDOW_SECONDS // VOLUME_BUCKET_SECONDS)
    for tick in window:
        column = min(
            IMAGE_SIZE - 1,
            max(
                0,
                (tick.ts_ms - start_ms)
                * (IMAGE_SIZE - 1)
                // (WINDOW_SECONDS * _MILLISECONDS),
            ),
        )
        prices_by_column[column] = tick.price
        bucket = min(
            len(volume_by_bucket) - 1,
            max(0, (tick.ts_ms - start_ms) // (VOLUME_BUCKET_SECONDS * _MILLISECONDS)),
        )
        volume_by_bucket[bucket] += tick.size

    points = [
        (column, _price_y((price / last_price - 1) * 10_000))
        for column, price in sorted(prices_by_column.items())
    ]
    if len(points) > 1:
        draw.line(points, fill=(0, 0, 0), width=2)
    elif points:
        x, y = points[0]
        draw.point((x, y), fill=(0, 0, 0))

    maximum_volume = max(volume_by_bucket, default=0.0)
    if maximum_volume > 0:
        for bucket, volume in enumerate(volume_by_bucket):
            if volume <= 0:
                continue
            bar_height = max(1, round(volume / maximum_volume * (_VOLUME_HEIGHT - 1)))
            x0 = bucket * IMAGE_SIZE // len(volume_by_bucket)
            x1 = (bucket + 1) * IMAGE_SIZE // len(volume_by_bucket) - 1
            y0 = IMAGE_SIZE - bar_height
            draw.rectangle((x0, y0, x1, IMAGE_SIZE - 1), fill=(150, 150, 150))
    return image


def _price_y(bps: float) -> int:
    clipped = max(-Y_RANGE_BPS, min(Y_RANGE_BPS, bps))
    return round((Y_RANGE_BPS - clipped) * (_PRICE_HEIGHT - 1) / (2 * Y_RANGE_BPS))


def chart_evidence(
    ticks: Sequence[Tick], anchor_ms: int
) -> dict[str, int | float | None]:
    recent = ticks_between(
        ticks, anchor_ms - HORIZON_SECONDS * _MILLISECONDS, anchor_ms
    )
    persistence = momentum_label(ticks, anchor_ms)
    window = ticks_between(ticks, anchor_ms - WINDOW_SECONDS * _MILLISECONDS, anchor_ms)
    range_bps = (
        (max(tick.price for tick in window) / min(tick.price for tick in window) - 1)
        * 10_000
        if window and min(tick.price for tick in window) > 0
        else None
    )
    return {
        "trades_last_30s": len(recent),
        "return_30s_bps": persistence[1] if persistence else None,
        "range_5m_bps": range_bps,
    }


def load_ticks(path: str | Path) -> list[Tick]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"timestamp_ms", "price", "size"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have timestamp_ms,price,size columns")
        ticks = [
            Tick(int(row["timestamp_ms"]), float(row["price"]), float(row["size"]))
            for row in reader
        ]
    if any(left.ts_ms > right.ts_ms for left, right in zip(ticks, ticks[1:], strict=False)):
        raise ValueError(f"{path} ticks are not sorted by timestamp_ms")
    return ticks


def build_samples(
    ticks: Sequence[Tick],
    symbol: str,
    split: str,
    stride_seconds: int = 60,
    max_samples: int | None = None,
) -> list[ChartSample]:
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if max_samples is not None and max_samples <= 0:
        return []
    if not ticks:
        return []
    stride_ms = stride_seconds * _MILLISECONDS
    anchor_ms = ticks[0].ts_ms + WINDOW_SECONDS * _MILLISECONDS
    last_anchor = ticks[-1].ts_ms - HORIZON_SECONDS * _MILLISECONDS
    samples: list[ChartSample] = []
    while anchor_ms <= last_anchor:
        outcome = momentum_label(ticks, anchor_ms + HORIZON_SECONDS * _MILLISECONDS)
        if outcome is not None:
            persistence = momentum_label(ticks, anchor_ms)
            samples.append(
                ChartSample(
                    symbol=symbol,
                    anchor_ms=anchor_ms,
                    split=split,
                    label=outcome[0],
                    outcome_bps=outcome[1],
                    persistence_label=persistence[0] if persistence else None,
                    persistence_bps=persistence[1] if persistence else None,
                    evidence=chart_evidence(ticks, anchor_ms),
                )
            )
        anchor_ms += stride_ms
    if max_samples is not None and len(samples) > max_samples:
        step = (len(samples) + max_samples - 1) // max_samples
        samples = samples[::step]
    return samples


def sample_dict(sample: ChartSample) -> dict[str, Any]:
    return asdict(sample)
