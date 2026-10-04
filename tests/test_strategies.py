from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from baystfirm.models import Candle
from baystfirm.strategies import (
    BACKTEST_NOTE,
    MAX_HOLD_BARS,
    Condition,
    ConditionGroup,
    ExitRule,
    IndicatorOperand,
    PriceOperand,
    Rule,
    ValueOperand,
    backtest,
    summarize_batch,
)
from baystfirm.track_record import _wilson_interval


def candle(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1,
    open_time: int | None = None,
) -> Candle:
    return Candle(
        open_time=index * 60_000 if open_time is None else open_time,
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        volume=volume,
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


def price_condition(op: str, value: float, *, field: str = "close") -> dict[str, object]:
    return {
        "left": {"kind": "price", "field": field},
        "op": op,
        "right": {"kind": "value", "value": value},
    }


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
    assert PriceOperand.model_validate({"kind": "price", "field": "volume"}).field == "volume"

    with pytest.raises(ValidationError):
        Condition.model_validate(
            {
                "left": {"kind": "value", "value": 2, "extra": True},
                "op": "above",
                "right": {"kind": "price"},
            }
        )


def test_condition_groups_apply_local_and_rule_level_combinations() -> None:
    base = rule().model_dump()
    base["combine"] = "all"
    base["conditions"] = [
        {
            "combine": "any",
            "conditions": [
                price_condition("above", 5),
                price_condition("below", 2),
            ],
        },
        price_condition("above", 0),
    ]
    grouped_rule = Rule.model_validate(base)
    result = backtest(
        grouped_rule,
        [candle(index, close) for index, close in enumerate([1, 6, 0.5, 3, 1, 3])],
        fee_bps=0,
    )

    assert isinstance(grouped_rule.conditions[0], ConditionGroup)
    assert [trade.entry_time for trade in result.trades] == [0, 120_000, 240_000]


def test_condition_groups_reject_single_condition_and_extra_fields() -> None:
    base = rule().model_dump()
    one_condition = {
        "combine": "any",
        "conditions": [price_condition("above", 5)],
    }
    base["conditions"] = [one_condition]
    with pytest.raises(ValidationError):
        Rule.model_validate(base)

    two_conditions_with_extra = {
        "combine": "any",
        "conditions": [
            price_condition("above", 5),
            price_condition("below", 2),
        ],
        "extra": True,
    }
    base["conditions"] = [two_conditions_with_extra]
    with pytest.raises(ValidationError):
        Rule.model_validate(base)


def test_volume_operand_and_grouped_indicator_condition_fire() -> None:
    volume_rule_data = rule().model_dump()
    volume_rule_data["conditions"] = [price_condition("above", 10, field="volume")]
    volume_result = backtest(
        Rule.model_validate(volume_rule_data),
        [candle(index, 10, volume=volume) for index, volume in enumerate([1, 20, 1])],
        fee_bps=0,
    )

    indicator_group_data = rule().model_dump()
    indicator_group_data["conditions"] = [
        {
            "combine": "any",
            "conditions": [
                {
                    "left": {"kind": "indicator", "spec": "sma:2"},
                    "op": "above",
                    "right": {"kind": "value", "value": 2.5},
                },
                price_condition("above", 10, field="volume"),
            ],
        }
    ]
    grouped_result = backtest(
        Rule.model_validate(indicator_group_data),
        [
            candle(0, 1),
            candle(1, 3),
            candle(2, 4, volume=20),
            candle(3, 5),
        ],
        fee_bps=0,
    )

    assert [trade.entry_time for trade in volume_result.trades] == [60_000]
    assert [trade.entry_time for trade in grouped_result.trades] == [120_000]


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


def test_slippage_reduces_rule_and_baseline_returns_and_reports_costs() -> None:
    data = [candle(index, close) for index, close in enumerate([100, 102, 100, 102])]
    fees_only = backtest(rule(), data, fee_bps=10)
    fees_and_slippage = backtest(rule(), data, fee_bps=10, slippage_bps=5)

    assert len(fees_and_slippage.trades) == len(fees_only.trades) == 2
    for trade, fee_trade in zip(fees_and_slippage.trades, fees_only.trades, strict=True):
        assert fee_trade.return_pct - trade.return_pct == pytest.approx(0.1)
    assert (
        fees_only.baseline.avg_return_pct - fees_and_slippage.baseline.avg_return_pct
        == pytest.approx(0.1)
    )
    assert (
        fees_only.buy_and_hold.return_pct - fees_and_slippage.buy_and_hold.return_pct
        == pytest.approx(0.1)
    )
    assert fees_and_slippage.costs.model_dump() == {
        "fee_bps": 10,
        "slippage_bps": 5,
        "round_trip_pct": pytest.approx(0.3),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"slippage_bps": 101}, "slippage_bps"),
        ({"holdout_pct": 51}, "holdout_pct"),
    ],
)
def test_backtest_validates_slippage_and_holdout_ranges(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        backtest(rule(), [candle(0, 10)], fee_bps=0, **kwargs)


def test_equity_curve_marks_open_trade_to_market_and_records_drawdown() -> None:
    result = backtest(
        rule(exit_data={"after_bars": 3}),
        [candle(index, close) for index, close in enumerate([10, 10, 8, 12])],
        fee_bps=0,
    )

    assert [point.equity for point in result.equity.curve] == pytest.approx([1.0, 1.0, 0.8, 1.2])
    assert [point.time for point in result.equity.curve] == [
        0,
        60_000,
        120_000,
        180_000,
    ]
    assert result.equity.total_return_pct == pytest.approx(20)
    assert result.equity.max_drawdown_pct == pytest.approx(20)


def test_equity_curve_compounds_sequential_trades_and_buy_and_hold() -> None:
    result = backtest(
        rule(),
        [candle(index, close) for index, close in enumerate([10, 11, 11, 12.1])],
        fee_bps=0,
    )

    assert [point.equity for point in result.equity.curve] == pytest.approx([1.0, 1.1, 1.1, 1.21])
    assert result.equity.total_return_pct == pytest.approx(21)
    assert result.by_year[0].compounded_return_pct == pytest.approx(21)

    drawdown = backtest(
        rule(expect="down", exit_data={"after_bars": 3}),
        [candle(index, close) for index, close in enumerate([10, 10, 8, 12])],
        fee_bps=0,
    ).buy_and_hold
    assert drawdown.return_pct == pytest.approx(20)
    assert drawdown.max_drawdown_pct == pytest.approx(20)


def test_by_year_groups_completed_trades_and_keeps_years_without_trades() -> None:
    times = (
        [datetime(2024, 12, 31, 23, minute, tzinfo=UTC) for minute in (57, 58, 59)]
        + [datetime(2025, 1, 1, 0, minute, tzinfo=UTC) for minute in (0, 1, 2)]
        + [datetime(2026, 1, 1, tzinfo=UTC)]
    )
    data = [
        candle(
            index,
            close,
            open_time=int(timestamp.timestamp() * 1000),
        )
        for index, (timestamp, close) in enumerate(
            zip(times, [10, 11, 12, 13, 14, 15, 20], strict=True)
        )
    ]

    result = backtest(rule(), data, fee_bps=0)

    assert [(row.year, row.trades) for row in result.by_year] == [
        (2024, 2),
        (2025, 1),
        (2026, 0),
    ]
    assert result.by_year[0].buy_and_hold_return_pct == pytest.approx(20)
    assert result.by_year[1].buy_and_hold_return_pct == pytest.approx((15 / 13 - 1) * 100)


def test_holdout_split_assigns_completed_trades_and_can_be_disabled() -> None:
    data = [candle(index, close) for index, close in enumerate([10, 11, 12, 13, 14, 15])]

    with_holdout = backtest(rule(), data, fee_bps=0, holdout_pct=50)
    without_holdout = backtest(rule(), data, fee_bps=0, holdout_pct=0)

    assert with_holdout.holdout is not None
    assert with_holdout.holdout.holdout_pct == 50
    assert with_holdout.holdout.split_time == 3 * 60_000
    assert with_holdout.holdout.in_sample.trades == 2
    assert with_holdout.holdout.recent.trades == 1
    assert without_holdout.holdout is None
    assert backtest(rule(), [], fee_bps=0, holdout_pct=20).holdout is None


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
    assert result.equity.curve[-1].equity == pytest.approx(1.01)
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


def test_batch_summary_counts_breadth_and_pools_full_stats_not_capped_trades() -> None:
    up_closes = [close for _ in range(251) for close in (100, 110)]
    up = backtest(rule(), [candle(index, close) for index, close in enumerate(up_closes)], 0)
    down = backtest(
        rule(expect="down"),
        [candle(index, close) for index, close in enumerate([100, 110, 100, 120])],
        0,
    )
    no_trades = backtest(
        rule(threshold=1000),
        [candle(index, close) for index, close in enumerate([100, 110, 100, 120])],
        0,
    )

    summary = summarize_batch([up, down, no_trades])

    assert len(up.trades) == 200
    assert up.stats.trades == 251
    assert summary.instruments_tested == 3
    assert summary.instruments_failed == 0
    assert summary.with_trades == 2
    assert summary.beat_baseline == 1
    assert summary.beat_buy_and_hold == 1
    assert summary.median_avg_return_pct == pytest.approx(
        (up.stats.avg_return_pct + down.stats.avg_return_pct) / 2
    )
    assert summary.pooled_trades == 253
    assert summary.pooled_wins == 251
    assert summary.pooled_win_rate == pytest.approx(251 / 253)
    assert summary.pooled_win_rate_ci95 == _wilson_interval(251, 253)


def test_tp_sl_only_rule_time_exits_after_max_hold_bars() -> None:
    data = [candle(index, 100) for index in range(MAX_HOLD_BARS + 2)]

    result = backtest(
        rule(exit_data={"take_profit_pct": 5, "stop_loss_pct": 5}),
        data,
        fee_bps=0,
    )

    assert result.trades[0].exit_time == MAX_HOLD_BARS * 60_000
    assert result.trades[0].exit == 100
    assert result.trades[0].exit_reason == "time"


def test_tp_sl_only_baseline_completes_trades_over_5000_bars() -> None:
    result = backtest(
        rule(exit_data={"take_profit_pct": 5, "stop_loss_pct": 5}),
        [candle(index, 100) for index in range(5000)],
        fee_bps=0,
    )

    assert result.baseline.trades == 5000 - MAX_HOLD_BARS
