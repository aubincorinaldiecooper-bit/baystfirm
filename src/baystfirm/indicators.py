from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from talipp.indicators import ATR, BB, EMA, MACD, RSI, SMA, Stoch
from talipp.ohlcv import OHLCV

from baystfirm.models import Candle

IndicatorKind = Literal["sma", "ema", "rsi", "macd", "bb", "atr", "stoch"]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    kind: IndicatorKind
    params: tuple[int | float, ...]
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

    bb = re.fullmatch(r"bb:([0-9]+),([0-9]+(?:\.[0-9]+)?)", spec)
    if bb is not None:
        period = int(bb.group(1))
        multiplier = float(bb.group(2))
        if not 2 <= period <= 200:
            raise ValueError("bb period must be between 2 and 200.")
        if not 0.5 <= multiplier <= 5:
            raise ValueError("bb multiplier must be between 0.5 and 5.")
        return IndicatorSpec(
            kind="bb",
            params=(period, multiplier),
            key=f"bb:{period},{multiplier:g}",
        )

    atr = re.fullmatch(r"atr:([0-9]+)", spec)
    if atr is not None:
        period = int(atr.group(1))
        if not 2 <= period <= 200:
            raise ValueError("atr period must be between 2 and 200.")
        return IndicatorSpec(kind="atr", params=(period,), key=f"atr:{period}")

    stoch = re.fullmatch(r"stoch:([0-9]+),([0-9]+)", spec)
    if stoch is not None:
        period, smoothing = (int(value) for value in stoch.groups())
        if any(not 2 <= value <= 200 for value in (period, smoothing)):
            raise ValueError("stoch periods must each be between 2 and 200.")
        return IndicatorSpec(
            kind="stoch",
            params=(period, smoothing),
            key=f"stoch:{period},{smoothing}",
        )

    raise ValueError(
        f"Invalid indicator spec {spec!r}; expected sma:<period>, ema:<period>, "
        "rsi:<period>, macd:<fast>,<slow>,<signal>, bb:<period>,<mult>, "
        "atr:<period>, or stoch:<period>,<smoothing>."
    )


def compute_indicator(
    spec: IndicatorSpec | str, candles: Sequence[Candle]
) -> dict[str, list[float | None]]:
    """Return candle-aligned talipp values; None entries are expected during warm-up."""
    parsed = parse_indicator(spec) if isinstance(spec, str) else spec
    closes = [float(candle.close) for candle in candles]

    if parsed.kind == "sma":
        indicator = SMA(int(parsed.params[0]))
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "ema":
        indicator = EMA(int(parsed.params[0]))
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "rsi":
        indicator = RSI(int(parsed.params[0]))
        for close in closes:
            indicator.add(close)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "bb":
        indicator = BB(int(parsed.params[0]), float(parsed.params[1]))
        for close in closes:
            indicator.add(close)
        values = indicator.output_values
        return {
            "upper": [_as_float(value.ub) if value is not None else None for value in values],
            "middle": [_as_float(value.cb) if value is not None else None for value in values],
            "lower": [_as_float(value.lb) if value is not None else None for value in values],
        }
    ohlcv = [
        OHLCV(
            open=float(candle.open),
            high=float(candle.high),
            low=float(candle.low),
            close=float(candle.close),
            volume=float(candle.volume),
        )
        for candle in candles
    ]
    if parsed.kind == "atr":
        indicator = ATR(int(parsed.params[0]))
        for candle in ohlcv:
            indicator.add(candle)
        return {"value": [_as_float(value) for value in indicator.output_values]}
    if parsed.kind == "stoch":
        indicator = Stoch(int(parsed.params[0]), int(parsed.params[1]))
        for candle in ohlcv:
            indicator.add(candle)
        values = indicator.output_values
        return {
            "k": [_as_float(value.k) if value is not None else None for value in values],
            "d": [_as_float(value.d) if value is not None else None for value in values],
        }

    fast, slow, signal = (int(value) for value in parsed.params)
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
    if spec.kind == "bb":
        return frozenset({"upper", "middle", "lower"})
    if spec.kind == "stoch":
        return frozenset({"k", "d"})
    return frozenset({"value"})


def _as_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)
