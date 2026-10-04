from __future__ import annotations

import pytest
from pydantic import ValidationError

from baystfirm.models import Candle
from baystfirm.strategies import (
    BACKTEST_NOTE,
    Condition,
    ExitRule,
    IndicatorOperand,
    PriceOperand,
    Rule,
    ValueOperand,
    backtest,
)


def candle(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
) -> Candle:
    return Candle(
        open_time=index * 60_000,
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        volume=1,
    )


def rule(
    *,
    op: str = "above",
    threshold: float = 0,
    expect: str = "up",
    exit_data: dict[str, int | float] | None = None,
) -> Rule:
    return Rule.model_validate(
        {
            "name": "test rule",
            "venue": "coinbase",
            "symbol": "BTC-USD",
            "interval": "1m",
            "conditions": [
                {
                    "left": {"kind": "price"},
                    "op": op,
                    "right": {"kind": "value", "value": threshold},
                }
            ],
            "expect": expect,
            "exit": exit_data or {"after_bars": 1},
        }
    )


def test_rule_operands_are_discriminated_and_extra_fields_are_forbidden() -> None:
    item = Condition.model_validate(
        {
            "left": {"kind": "price"},
            "op": "above",
            "right": {"kind": "value", "value": 2},
        }
    )
    assert isinstance(item.left, PriceOperand)
    assert isinstance(item.right, ValueOperand)
    assert item.left.field == "close"

    with pytest.raises(ValidationError):
        Condition.model_validate(
            {
                "left": {"kind": "value", "value": 2, "extra": True},
                "op": "above",
                "right": {"kind": "price"},
            }
        )


def test_exit_rule_requires_at_least_one_exit_condition() -> None:
    with pytest.raises(ValidationError, match="at least one exit"):
        ExitRule.model_validate({})


def test_indicator_operand_validates_spec_and_output() -> None:
    operand = IndicatorOperand.model_validate({"kind": "indicator", "spec": "sma:03"})
    assert operand.spec == "sma:3"

    with pytest.raises(ValidationError, match="output"):
        IndicatorOperand.model_validate({"kind": "indicator", "spec": "sma:3", "output": "macd"})


def test_none_indicator_values_never_satisfy_a_condition() -> None:
    rule_with_warmup = Rule.model_validate(
        {
            "name": "indicator warmup",
            "venue": "coinbase",
            "symbol": "BTC-USD",
            "interval": "1m",
            "conditions": [
                {
                    "left": {"kind": "indicator", "spec": "sma:3"},
                    "op": "above",
                    "right": {"kind": "value", "value": 0},
                }
            ],
            "expect": "up",
            "exit": {"after_bars": 1},
        }
    )

    result = backtest(rule_with_warmup, [candle(0, 1), candle(1, 2)], fee_bps=0)

    assert result.stats.trades == 0
    assert result.open_trades == []


def test_crosses_above_fires_only_on_the_crossing_bar() -> None:
    result = backtest(
        rule(op="crosses_above", threshold=2),
        [candle(index, close) for index, close in enumerate([1, 2, 2, 3, 3, 3])],
        fee_bps=0,
    )

    assert result.stats.trades == 1
    assert result.trades[0].entry_time == 3 * 60_000
    assert result.trades[0].exit_reason == "time"


def test_one_strategy_position_is_allowed_and_baseline_overlaps() -> None:
    data = [candle(index, 10) for index in range(6)]

    result = backtest(
        rule(exit_data={"after_bars": 2}),
        data,
        fee_bps=0,
    )

    assert result.stats.trades == 2
    assert [trade.entry_time for trade in result.trades] == [0, 3 * 60_000]
    assert result.baseline.trades == 4


@pytest.mark.parametrize(
    ("high", "low", "expected_exit", "expected_reason"),
    [
        (106, 99, 105, "take_profit"),
        (101, 94, 95, "stop_loss"),
        (106, 94, 95, "stop_loss"),
    ],
)
def test_intrabar_exits_and_stop_loss_precedence(
    high: float, low: float, expected_exit: float, expected_reason: str
) -> None:
    result = backtest(
        rule(exit_data={"take_profit_pct": 5, "stop_loss_pct": 5}),
        [candle(0, 100), candle(1, 103, high=high, low=low)],
        fee_bps=0,
    )

    assert result.trades[0].exit == expected_exit
    assert result.trades[0].exit_reason == expected_reason


def test_time_exit_uses_close_and_fee_reduces_return() -> None:
    data = [candle(0, 100), candle(1, 102, high=103, low=99)]

    free = backtest(rule(), data, fee_bps=0)
    with_fee = backtest(rule(), data, fee_bps=10)

    assert free.trades[0].exit == 102
    assert free.trades[0].exit_reason == "time"
    assert free.trades[0].return_pct == pytest.approx(2)
    assert with_fee.trades[0].return_pct == pytest.approx(1.8)


def test_down_expectation_mirrors_take_profit_and_return() -> None:
    result = backtest(
        rule(expect="down", exit_data={"take_profit_pct": 5, "stop_loss_pct": 5}),
        [candle(0, 100), candle(1, 98, high=101, low=94)],
        fee_bps=0,
    )

    assert result.trades[0].exit == 95
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].return_pct == pytest.approx((100 / 95 - 1) * 100)


def test_open_trade_is_reported_but_excluded_from_statistics() -> None:
    result = backtest(
        rule(exit_data={"after_bars": 3}),
        [candle(0, 100), candle(1, 101)],
        fee_bps=0,
    )

    assert result.stats.trades == 0
    assert result.stats.win_rate is None
    assert result.stats.win_rate_ci95 is None
    assert len(result.open_trades) == 1
    assert result.open_trades[0].entry_time == 0
    assert result.note == BACKTEST_NOTE


def test_wilson_interval_and_worst_losing_streak_are_reported() -> None:
    result = backtest(
        rule(),
        [candle(index, close) for index, close in enumerate([100, 99, 98, 97, 100, 101])],
        fee_bps=0,
    )

    assert result.stats.trades == 3
    assert result.stats.win_rate_ci95 is not None
    assert result.stats.worst_losing_streak == 2


def test_baseline_enters_every_eligible_bar_without_rule_signals() -> None:
    result = backtest(
        rule(threshold=1000),
        [candle(index, 10) for index in range(5)],
        fee_bps=0,
    )

    assert result.stats.trades == 0
    assert result.baseline.trades == 4
    assert result.baseline.win_rate == 0
