from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from talipp.indicators import EMA, MACD, RSI, SMA

from baystfirm.models import Candle

IndicatorKind = Literal["sma", "ema", "rsi", "macd"]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    kind: IndicatorKind
    params: tuple[int, ...]
    key: str


def parse_indicator(spec: str) -> IndicatorSpec:
    simple = re.fullmatch(r"(sma|ema|rsi):([0-9]+)", spec)
    if simple is not None:
        kind = cast(IndicatorKind, simple.group(1))
        period = int(simple.group(2))
        if not 2 <= period <= 200:
            raise ValueError(f"{kind} period must be between 2 and 200.")
        return IndicatorSpec(kind=kind, params=(period,), key=f"{kind}:{period}")

    macd = re.fullmatch(r"macd:([0-9]+),([0-9]+),([0-9]+)", spec)
    if macd is not None:
        fast, slow, signal = (int(value) for value in macd.groups())
        if any(not 2 <= period <= 200 for period in (fast, slow, signal)):
            raise ValueError("macd periods must each be between 2 and 200.")
        if fast >= slow:
            raise ValueError("macd fast period must be less than its slow period.")
        return IndicatorSpec(
            kind="macd",
            params=(fast, slow, signal),
            key=f"macd:{fast},{slow},{signal}",
        )

    raise ValueError(
        f"Invalid indicator spec {spec!r}; expected sma:<period>, ema:<period>, "
        "rsi:<period>, or macd:<fast>,<slow>,<signal>."
    )


def compute_indicator(
    spec: IndicatorSpec | str, candles: Sequence[Candle]
) -> dict[str, list[float | None]]:
    """Return candle-aligned talipp values; None entries are expected during warm-up."""
    parsed = parse_indicator(spec) if isinstance(spec, str) else spec
    closes = [float(candle.close) for candle in candles]

    if parsed.kind == "sma":
        indicator = SMA(parsed.params[0])
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "ema":
        indicator = EMA(parsed.params[0])
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "rsi":
        indicator = RSI(parsed.params[0])
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}

    fast, slow, signal = parsed.params
    indicator = MACD(fast, slow, signal)
    for close in closes:
        indicator.add(close)
    values = indicator.output_values
    return {
        "macd": [_as_float(value.macd) if value is not None else None for value in values],
        "signal": [_as_float(value.signal) if value is not None else None for value in values],
        "histogram": [
            _as_float(value.histogram) if value is not None else None for value in values
        ],
    }


def indicator_outputs(spec: IndicatorSpec) -> frozenset[str]:
    if spec.kind == "macd":
        return frozenset({"macd", "signal", "histogram"})
    return frozenset({"value"})


def _as_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)
