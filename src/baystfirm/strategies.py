from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from baystfirm.indicators import (
    IndicatorSpec,
    compute_indicator,
    indicator_outputs,
    parse_indicator,
)
from baystfirm.models import Candle
from baystfirm.track_record import _wilson_interval

Interval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
]
Expectation = Literal["up", "down"]
ExitReason = Literal["take_profit", "stop_loss", "time"]

BACKTEST_NOTE = (
    "Backtest on historical candles with simulated fees. Past results overstate live results. "
    "Your rule, not investment advice."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriceOperand(StrictModel):
    kind: Literal["price"]
    field: Literal["open", "high", "low", "close"] = "close"


class IndicatorOperand(StrictModel):
    kind: Literal["indicator"]
    spec: str
    output: str = "value"

    @field_validator("spec")
    @classmethod
    def validate_spec(cls, value: str) -> str:
        return parse_indicator(value).key

    @model_validator(mode="after")
    def validate_output(self) -> IndicatorOperand:
        spec = parse_indicator(self.spec)
        if self.output not in indicator_outputs(spec):
            allowed = ", ".join(sorted(indicator_outputs(spec)))
            raise ValueError(f"{spec.kind} output must be one of: {allowed}.")
        return self


class ValueOperand(StrictModel):
    kind: Literal["value"]
    value: float


Operand = Annotated[
    PriceOperand | IndicatorOperand | ValueOperand,
    Field(discriminator="kind"),
]


class Condition(StrictModel):
    left: Operand
    op: Literal["above", "below", "crosses_above", "crosses_below"]
    right: Operand


class ExitRule(StrictModel):
    after_bars: int | None = Field(default=None, ge=1, le=500)
    take_profit_pct: float | None = Field(default=None, gt=0, le=100)
    stop_loss_pct: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def require_exit_condition(self) -> ExitRule:
        if self.after_bars is None and self.take_profit_pct is None and self.stop_loss_pct is None:
            raise ValueError("at least one exit condition must be set.")
        return self


class Rule(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    venue: str
    symbol: str
    interval: Interval
    conditions: list[Condition] = Field(min_length=1, max_length=5)
    combine: Literal["all", "any"] = "all"
    expect: Expectation
    exit: ExitRule


class BacktestRequest(StrictModel):
    rule: Rule
    bars: int = Field(default=1000, ge=100, le=5000)
    fee_bps: float = Field(default=10, ge=0, le=100)


class CompletedTrade(StrictModel):
    entry_time: int
    exit_time: int
    entry: float
    exit: float
    return_pct: float
    exit_reason: ExitReason


class OpenTrade(StrictModel):
    entry_time: int
    entry: float
    last_time: int
    last_close: float
    bars_held: int


class BacktestStats(StrictModel):
    trades: int
    wins: int
    win_rate: float | None
    win_rate_ci95: list[float] | None
    avg_return_pct: float | None
    median_return_pct: float | None
    worst_losing_streak: int


class BaselineStats(StrictModel):
    trades: int
    win_rate: float | None
    avg_return_pct: float | None


class BacktestResult(StrictModel):
    bars_tested: int
    first_bar_time: int | None
    last_bar_time: int | None
    stats: BacktestStats
    baseline: BaselineStats
    trades: list[CompletedTrade]
    open_trades: list[OpenTrade]
    note: str = BACKTEST_NOTE


def backtest(rule: Rule, candles: Sequence[Candle], fee_bps: float) -> BacktestResult:
    """Evaluate a user rule over closed candles; no trade execution is performed."""
    if not 0 <= fee_bps <= 100:
        raise ValueError("fee_bps must be between 0 and 100.")
    indicators = _compute_rule_indicators(rule, candles)
    completed: list[CompletedTrade] = []
    open_trades: list[OpenTrade] = []
    index = 0
    while index < len(candles):
        if not _signal_at(rule, candles, indicators, index):
            index += 1
            continue
        trade, exit_index = _simulate_exit(rule, candles, index, fee_bps)
        if trade is None or exit_index is None:
            last_candle = candles[-1]
            open_trades.append(
                OpenTrade(
                    entry_time=candles[index].open_time,
                    entry=candles[index].close,
                    last_time=last_candle.open_time,
                    last_close=last_candle.close,
                    bars_held=len(candles) - index - 1,
                )
            )
            break
        completed.append(trade)
        index = exit_index + 1

    returns = [trade.return_pct for trade in completed]
    wins = sum(value > 0 for value in returns)
    losing_streak = 0
    worst_losing_streak = 0
    for value in returns:
        if value <= 0:
            losing_streak += 1
            worst_losing_streak = max(worst_losing_streak, losing_streak)
        else:
            losing_streak = 0

    baseline_returns = [
        trade.return_pct
        for entry_index in range(len(candles))
        if (trade := _simulate_exit(rule, candles, entry_index, fee_bps)[0]) is not None
    ]
    baseline_wins = sum(value > 0 for value in baseline_returns)
    stats = BacktestStats(
        trades=len(completed),
        wins=wins,
        win_rate=wins / len(completed) if completed else None,
        win_rate_ci95=_wilson_interval(wins, len(completed)),
        avg_return_pct=sum(returns) / len(returns) if returns else None,
        median_return_pct=median(returns) if returns else None,
        worst_losing_streak=worst_losing_streak,
    )
    baseline = BaselineStats(
        trades=len(baseline_returns),
        win_rate=baseline_wins / len(baseline_returns) if baseline_returns else None,
        avg_return_pct=(
            sum(baseline_returns) / len(baseline_returns) if baseline_returns else None
        ),
    )
    return BacktestResult(
        bars_tested=len(candles),
        first_bar_time=candles[0].open_time if candles else None,
        last_bar_time=candles[-1].open_time if candles else None,
        stats=stats,
        baseline=baseline,
        trades=completed[-200:],
        open_trades=open_trades,
    )


def _compute_rule_indicators(
    rule: Rule, candles: Sequence[Candle]
) -> dict[str, dict[str, list[float | None]]]:
    specs: dict[str, IndicatorSpec] = {}
    for condition in rule.conditions:
        for operand in (condition.left, condition.right):
            if isinstance(operand, IndicatorOperand):
                parsed = parse_indicator(operand.spec)
                specs[parsed.key] = parsed
    return {key: compute_indicator(spec, candles) for key, spec in specs.items()}


def _signal_at(
    rule: Rule,
    candles: Sequence[Candle],
    indicators: dict[str, dict[str, list[float | None]]],
    index: int,
) -> bool:
    results = [
        _condition_at(condition, candles, indicators, index) for condition in rule.conditions
    ]
    return all(results) if rule.combine == "all" else any(results)


def _condition_at(
    condition: Condition,
    candles: Sequence[Candle],
    indicators: dict[str, dict[str, list[float | None]]],
    index: int,
) -> bool:
    left = _operand_at(condition.left, candles, indicators, index)
    right = _operand_at(condition.right, candles, indicators, index)
    if left is None or right is None:
        return False
    if condition.op == "above":
        return left > right
    if condition.op == "below":
        return left < right
    if index == 0:
        return False
    previous_left = _operand_at(condition.left, candles, indicators, index - 1)
    previous_right = _operand_at(condition.right, candles, indicators, index - 1)
    if previous_left is None or previous_right is None:
        return False
    if condition.op == "crosses_above":
        return previous_left <= previous_right and left > right
    return previous_left >= previous_right and left < right


def _operand_at(
    operand: Operand,
    candles: Sequence[Candle],
    indicators: dict[str, dict[str, list[float | None]]],
    index: int,
) -> float | None:
    if isinstance(operand, PriceOperand):
        return float(getattr(candles[index], operand.field))
    if isinstance(operand, ValueOperand):
        return operand.value
    parsed = parse_indicator(operand.spec)
    return indicators[parsed.key][operand.output][index]


def _simulate_exit(
    rule: Rule,
    candles: Sequence[Candle],
    entry_index: int,
    fee_bps: float,
) -> tuple[CompletedTrade | None, int | None]:
    entry_candle = candles[entry_index]
    entry_price = entry_candle.close
    if entry_price <= 0:
        raise ValueError("candle close prices must be positive.")
    exit_rule = rule.exit
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    if rule.expect == "up":
        if exit_rule.take_profit_pct is not None:
            take_profit_price = entry_price * (1 + exit_rule.take_profit_pct / 100)
        if exit_rule.stop_loss_pct is not None:
            stop_loss_price = entry_price * (1 - exit_rule.stop_loss_pct / 100)
    else:
        if exit_rule.take_profit_pct is not None:
            take_profit_price = entry_price * (1 - exit_rule.take_profit_pct / 100)
        if exit_rule.stop_loss_pct is not None:
            stop_loss_price = entry_price * (1 + exit_rule.stop_loss_pct / 100)

    for index in range(entry_index + 1, len(candles)):
        candle = candles[index]
        if rule.expect == "up":
            stop_hit = stop_loss_price is not None and candle.low <= stop_loss_price
            profit_hit = take_profit_price is not None and candle.high >= take_profit_price
        else:
            stop_hit = stop_loss_price is not None and candle.high >= stop_loss_price
            profit_hit = take_profit_price is not None and candle.low <= take_profit_price

        if stop_hit:
            exit_price = stop_loss_price
            exit_reason: ExitReason = "stop_loss"
        elif profit_hit:
            exit_price = take_profit_price
            exit_reason = "take_profit"
        elif exit_rule.after_bars is not None and index - entry_index >= exit_rule.after_bars:
            exit_price = candle.close
            exit_reason = "time"
        else:
            continue

        assert exit_price is not None
        if exit_price <= 0:
            raise ValueError("candle exit prices must be positive.")
        directional_return = (
            (exit_price / entry_price - 1) * 100
            if rule.expect == "up"
            else (entry_price / exit_price - 1) * 100
        )
        return_pct = directional_return - 2 * fee_bps / 100
        return (
            CompletedTrade(
                entry_time=entry_candle.open_time,
                exit_time=candle.open_time,
                entry=entry_price,
                exit=exit_price,
                return_pct=return_pct,
                exit_reason=exit_reason,
            ),
            index,
        )
    return None, None
