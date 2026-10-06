from __future__ import annotations

import pytest

from baystfirm.indicators import compute_indicator, indicator_outputs, parse_indicator
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
    assert parse_indicator("bb:020,2.0").key == "bb:20,2"
    assert parse_indicator("atr:014").key == "atr:14"
    assert parse_indicator("stoch:014,03").key == "stoch:14,3"


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
        "bb:1,2",
        "bb:20,5.1",
        "atr:201",
        "stoch:201,2",
    ],
)
def test_invalid_indicator_specs_raise_clear_errors(spec: str) -> None:
    with pytest.raises(ValueError, match="indicator|period|macd|multiplier"):
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


def test_talipp_bollinger_bands_use_canonical_outputs() -> None:
    output = compute_indicator("bb:3,2.0", candles([1, 2, 3, 4]))

    assert output["middle"] == [None, None, 2.0, 3.0]
    assert output["upper"][2] == pytest.approx(3.632993)
    assert output["lower"][2] == pytest.approx(0.367007)
    assert all(len(values) == 4 for values in output.values())


def test_talipp_atr_uses_ohlcv_and_stochastic_outputs_stay_aligned() -> None:
    constant_range = [
        Candle(
            open_time=index * 60_000,
            open=1.5,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=1,
        )
        for index in range(4)
    ]
    atr = compute_indicator("atr:2", constant_range)["value"]
    stoch_candles = [
        Candle(
            open_time=index * 60_000,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        )
        for index, close in enumerate([1, 2, 3, 4, 5])
    ]
    stochastic = compute_indicator("stoch:3,2", stoch_candles)

    assert atr[1] == pytest.approx(1.5)
    assert len(atr) == len(constant_range)
    assert set(stochastic) == {"k", "d"}
    assert len(stochastic["k"]) == len(stochastic["d"]) == 5
    assert stochastic["k"][:2] == [None, None]
    assert stochastic["d"][:3] == [None, None, None]


def test_new_indicator_output_names_match_talipp_series() -> None:
    assert set(indicator_outputs(parse_indicator("bb:3,2"))) == {
        "upper",
        "middle",
        "lower",
    }
    assert set(indicator_outputs(parse_indicator("atr:2"))) == {"value"}
    assert set(indicator_outputs(parse_indicator("stoch:3,2"))) == {"k", "d"}
