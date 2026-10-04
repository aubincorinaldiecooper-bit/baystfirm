from __future__ import annotations

import pytest

from baystfirm.indicators import compute_indicator, parse_indicator
from baystfirm.models import Candle


def candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(
            open_time=index * 60_000,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1,
        )
        for index, close in enumerate(closes)
    ]


def test_indicator_parsing_canonicalizes_integer_specs() -> None:
    assert parse_indicator("sma:03").key == "sma:3"
    assert parse_indicator("macd:2,3,9").params == (2, 3, 9)


@pytest.mark.parametrize(
    "spec",
    [
        "unknown:3",
        "sma:1",
        "ema:201",
        "rsi:2.5",
        "macd:3,3,2",
        "macd:2,201,3",
        "macd:2,3",
    ],
)
def test_invalid_indicator_specs_raise_clear_errors(spec: str) -> None:
    with pytest.raises(ValueError, match="indicator|period|macd"):
        parse_indicator(spec)


def test_talipp_sma_ema_and_rsi_are_aligned_with_warmup() -> None:
    data = candles([1, 2, 3, 4, 5])

    sma = compute_indicator("sma:3", data)["value"]
    ema = compute_indicator("ema:3", data)["value"]
    rsi = compute_indicator("rsi:3", data)["value"]

    assert sma == [None, None, 2.0, 3.0, 4.0]
    assert ema == [None, None, 2.0, 3.0, 4.0]
    assert rsi == [None, None, None, 100.0, 100.0]
    assert len(sma) == len(ema) == len(rsi) == len(data)


def test_talipp_macd_exposes_aligned_macd_signal_and_histogram() -> None:
    output = compute_indicator("macd:2,3,2", candles([1, 2, 3, 4, 5]))

    assert output == {
        "macd": [None, None, 0.5, 0.5, 0.5],
        "signal": [None, None, None, 0.5, 0.5],
        "histogram": [None, None, None, 0.0, 0.0],
    }


def test_indicator_computation_accepts_parsed_specs() -> None:
    spec = parse_indicator("sma:2")

    assert compute_indicator(spec, candles([1, 3])) == {"value": [None, 2.0]}
