from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import floor, prod
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
    "Backtest on historical candles with simulated fees and slippage. Past results overstate "
    "live results. Your rule, not investment advice."
)
MAX_HOLD_BARS = 500


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriceOperand(StrictModel):
    kind: Literal["price"]
    field: Literal["open", "high", "low", "close", "volume"] = "close"


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


class ConditionGroup(StrictModel):
    combine: Literal["all", "any"]
    conditions: list[Condition] = Field(min_length=2, max_length=5)


class ExitRule(StrictModel):
    """When after_bars is unset, time-exit at the close after MAX_HOLD_BARS."""

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
    conditions: list[Condition | ConditionGroup] = Field(min_length=1, max_length=5)
    combine: Literal["all", "any"] = "all"
    expect: Expectation
    exit: ExitRule


class BacktestRequest(StrictModel):
    rule: Rule
    bars: int = Field(default=1000, ge=100, le=5000)
    fee_bps: float = Field(default=10, ge=0, le=100)
    slippage_bps: float = Field(default=5, ge=0, le=100)
    holdout_pct: float = Field(default=20, ge=0, le=50)


class Instrument(StrictModel):
    venue: str
    symbol: str


class BatchBacktestRequest(BacktestRequest):
    also: list[Instrument] = Field(min_length=1, max_length=19)


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


class Costs(StrictModel):
    fee_bps: float
    slippage_bps: float
    round_trip_pct: float


class EquityPoint(StrictModel):
    time: int
    equity: float


class EquityStats(StrictModel):
    total_return_pct: float
    max_drawdown_pct: float
    curve: list[EquityPoint]


class BuyAndHoldStats(StrictModel):
    """Always long from the first close to the last close, regardless of rule expectation."""

    return_pct: float
    max_drawdown_pct: float


class YearStats(StrictModel):
    year: int
    trades: int
    wins: int
    win_rate: float | None
    win_rate_ci95: list[float] | None
    avg_return_pct: float | None
    compounded_return_pct: float
    buy_and_hold_return_pct: float


class HoldoutStats(StrictModel):
    holdout_pct: float
    split_time: int
    in_sample: BacktestStats
    recent: BacktestStats


class BacktestResult(StrictModel):
    bars_tested: int
    first_bar_time: int | None
    last_bar_time: int | None
    stats: BacktestStats
    baseline: BaselineStats
    costs: Costs
    equity: EquityStats
    buy_and_hold: BuyAndHoldStats
    by_year: list[YearStats]
    holdout: HoldoutStats | None
    trades: list[CompletedTrade]
    open_trades: list[OpenTrade]
    note: str = BACKTEST_NOTE


BATCH_NOTE = (
    "Coins tend to move together, so results across instruments are not independent tests. "
    "Judge your rule by how many instruments it holds up on, not the best one. "
    "Your rule, not investment advice."
)


class BatchInstrumentResult(StrictModel):
    venue: str
    symbol: str
    error: str | None
    bars_tested: int | None
    truncated: bool | None
    stats: BacktestStats | None
    baseline: BaselineStats | None
    buy_and_hold: BuyAndHoldStats | None
    total_return_pct: float | None
    max_drawdown_pct: float | None
    recent: BacktestStats | None


class BatchSummary(StrictModel):
    instruments_tested: int
    instruments_failed: int = 0
    with_trades: int
    beat_baseline: int
    beat_buy_and_hold: int
    median_avg_return_pct: float | None
    pooled_trades: int
    pooled_wins: int
    pooled_win_rate: float | None
    pooled_win_rate_ci95: list[float] | None


def summarize_batch(results: Sequence[BacktestResult]) -> BatchSummary:
    with_trades = [result for result in results if result.stats.trades > 0]
    pooled_trades = sum(result.stats.trades for result in results)
    pooled_wins = sum(result.stats.wins for result in results)
    avg_returns = [
        result.stats.avg_return_pct
        for result in with_trades
        if result.stats.avg_return_pct is not None
    ]
    return BatchSummary(
        instruments_tested=len(results),
        with_trades=len(with_trades),
        beat_baseline=sum(
            result.stats.avg_return_pct is not None
            and result.baseline.avg_return_pct is not None
            and result.stats.avg_return_pct > result.baseline.avg_return_pct
            for result in results
        ),
        beat_buy_and_hold=sum(
            result.equity.total_return_pct > result.buy_and_hold.return_pct
            for result in with_trades
        ),
        median_avg_return_pct=median(avg_returns) if avg_returns else None,
        pooled_trades=pooled_trades,
        pooled_wins=pooled_wins,
        pooled_win_rate=pooled_wins / pooled_trades if pooled_trades else None,
        pooled_win_rate_ci95=_wilson_interval(pooled_wins, pooled_trades),
    )


def backtest(
    rule: Rule,
    candles: Sequence[Candle],
    fee_bps: float,
    *,
    slippage_bps: float = 0.0,
    holdout_pct: float = 0.0,
) -> BacktestResult:
    """Evaluate a user rule over closed candles; no trade execution is performed."""
    if not 0 <= fee_bps <= 100:
        raise ValueError("fee_bps must be between 0 and 100.")
    if not 0 <= slippage_bps <= 100:
        raise ValueError("slippage_bps must be between 0 and 100.")
    if not 0 <= holdout_pct <= 50:
        raise ValueError("holdout_pct must be between 0 and 50.")

    per_side_cost_bps = fee_bps + slippage_bps
    round_trip_pct = 2 * per_side_cost_bps / 100
    indicators = _compute_rule_indicators(rule, candles)
    completed: list[CompletedTrade] = []
    open_trades: list[OpenTrade] = []
    equity_values = [1.0] * len(candles)
    equity = 1.0
    index = 0
    while index < len(candles):
        if not _signal_at(rule, candles, indicators, index):
            equity_values[index] = equity
            index += 1
            continue
        entry_equity = equity
        trade, exit_index = _simulate_exit(rule, candles, index, per_side_cost_bps)
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
            for mark_index in range(index, len(candles)):
                equity_values[mark_index] = _mark_to_market(
                    rule.expect,
                    candles[index].close,
                    candles[mark_index].close,
                    entry_equity,
                    per_side_cost_bps,
                )
            break
        completed.append(trade)
        for mark_index in range(index, exit_index):
            equity_values[mark_index] = _mark_to_market(
                rule.expect,
                candles[index].close,
                candles[mark_index].close,
                entry_equity,
                per_side_cost_bps,
            )
        equity_values[exit_index] = entry_equity * (1 + trade.return_pct / 100)
        equity = equity_values[exit_index]
        index = exit_index + 1

    returns = [trade.return_pct for trade in completed]
    baseline_returns = [
        trade.return_pct
        for entry_index in range(len(candles))
        if (trade := _simulate_exit(rule, candles, entry_index, per_side_cost_bps)[0]) is not None
    ]
    stats = _stats(returns)
    baseline_full_stats = _stats(baseline_returns)
    baseline = BaselineStats(
        trades=baseline_full_stats.trades,
        win_rate=baseline_full_stats.win_rate,
        avg_return_pct=baseline_full_stats.avg_return_pct,
    )
    curve = [
        EquityPoint(time=candle.open_time, equity=value)
        for candle, value in zip(candles, equity_values, strict=True)
    ]
    equity_stats = EquityStats(
        total_return_pct=(equity_values[-1] - 1) * 100 if equity_values else 0,
        max_drawdown_pct=_max_drawdown(equity_values),
        curve=curve,
    )
    return BacktestResult(
        bars_tested=len(candles),
        first_bar_time=candles[0].open_time if candles else None,
        last_bar_time=candles[-1].open_time if candles else None,
        stats=stats,
        baseline=baseline,
        costs=Costs(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            round_trip_pct=round_trip_pct,
        ),
        equity=equity_stats,
        buy_and_hold=_buy_and_hold(candles, round_trip_pct),
        by_year=_by_year(candles, completed),
        holdout=_holdout(candles, completed, holdout_pct),
        trades=completed[-200:],
        open_trades=open_trades,
    )


def _stats(returns: Sequence[float]) -> BacktestStats:
    wins = sum(value > 0 for value in returns)
    losing_streak = 0
    worst_losing_streak = 0
    for value in returns:
        if value <= 0:
            losing_streak += 1
            worst_losing_streak = max(worst_losing_streak, losing_streak)
        else:
            losing_streak = 0
    count = len(returns)
    return BacktestStats(
        trades=count,
        wins=wins,
        win_rate=wins / count if count else None,
        win_rate_ci95=_wilson_interval(wins, count),
        avg_return_pct=sum(returns) / count if count else None,
        median_return_pct=median(returns) if returns else None,
        worst_losing_streak=worst_losing_streak,
    )


def _mark_to_market(
    expect: Expectation,
    entry: float,
    close: float,
    entry_equity: float,
    per_side_cost_bps: float,
) -> float:
    if close <= 0:
        raise ValueError("candle close prices must be positive.")
    directional_pct = (close / entry - 1) * 100 if expect == "up" else (entry / close - 1) * 100
    return entry_equity * (1 + (directional_pct - per_side_cost_bps / 100) / 100)


def _max_drawdown(values: Sequence[float]) -> float:
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (1 - value / peak) * 100)
    return max_drawdown


def _buy_and_hold(candles: Sequence[Candle], round_trip_pct: float) -> BuyAndHoldStats:
    if not candles:
        return BuyAndHoldStats(return_pct=0, max_drawdown_pct=0)
    first_close = candles[0].close
    if first_close <= 0:
        raise ValueError("candle close prices must be positive.")
    normalized_closes = []
    for candle in candles:
        if candle.close <= 0:
            raise ValueError("candle close prices must be positive.")
        normalized_closes.append(candle.close / first_close)
    return BuyAndHoldStats(
        return_pct=(normalized_closes[-1] - 1) * 100 - round_trip_pct,
        max_drawdown_pct=_max_drawdown(normalized_closes),
    )


def _by_year(candles: Sequence[Candle], completed: Sequence[CompletedTrade]) -> list[YearStats]:
    candles_by_year: dict[int, list[Candle]] = {}
    returns_by_year: dict[int, list[float]] = {}
    for candle in candles:
        year = datetime.fromtimestamp(candle.open_time / 1000, UTC).year
        candles_by_year.setdefault(year, []).append(candle)
    for trade in completed:
        year = datetime.fromtimestamp(trade.entry_time / 1000, UTC).year
        returns_by_year.setdefault(year, []).append(trade.return_pct)

    result: list[YearStats] = []
    for year, year_candles in sorted(candles_by_year.items()):
        chronological = sorted(year_candles, key=lambda candle: candle.open_time)
        first_close = chronological[0].close
        last_close = chronological[-1].close
        if first_close <= 0:
            raise ValueError("candle close prices must be positive.")
        year_returns = returns_by_year.get(year, [])
        year_stats = _stats(year_returns)
        result.append(
            YearStats(
                year=year,
                trades=year_stats.trades,
                wins=year_stats.wins,
                win_rate=year_stats.win_rate,
                win_rate_ci95=year_stats.win_rate_ci95,
                avg_return_pct=year_stats.avg_return_pct,
                compounded_return_pct=(prod(1 + value / 100 for value in year_returns) - 1) * 100,
                buy_and_hold_return_pct=(last_close / first_close - 1) * 100,
            )
        )
    return result


def _holdout(
    candles: Sequence[Candle], completed: Sequence[CompletedTrade], holdout_pct: float
) -> HoldoutStats | None:
    if holdout_pct == 0 or not candles:
        return None
    split_index = floor(len(candles) * (1 - holdout_pct / 100))
    split_time = candles[split_index].open_time
    in_sample = [trade.return_pct for trade in completed if trade.entry_time < split_time]
    recent = [trade.return_pct for trade in completed if trade.entry_time >= split_time]
    return HoldoutStats(
        holdout_pct=holdout_pct,
        split_time=split_time,
        in_sample=_stats(in_sample),
        recent=_stats(recent),
    )


def _compute_rule_indicators(
    rule: Rule, candles: Sequence[Candle]
) -> dict[str, dict[str, list[float | None]]]:
    specs: dict[str, IndicatorSpec] = {}
    for item in rule.conditions:
        conditions = item.conditions if isinstance(item, ConditionGroup) else [item]
        for condition in conditions:
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
    results = [_condition_item_at(item, candles, indicators, index) for item in rule.conditions]
    return all(results) if rule.combine == "all" else any(results)


def _condition_item_at(
    item: Condition | ConditionGroup,
    candles: Sequence[Candle],
    indicators: dict[str, dict[str, list[float | None]]],
    index: int,
) -> bool:
    if isinstance(item, Condition):
        return _condition_at(item, candles, indicators, index)
    results = [
        _condition_at(condition, candles, indicators, index) for condition in item.conditions
    ]
    return all(results) if item.combine == "all" else any(results)


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
    per_side_cost_bps: float,
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

    time_exit_bars = exit_rule.after_bars if exit_rule.after_bars is not None else MAX_HOLD_BARS
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
        elif index - entry_index >= time_exit_bars:
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
        return_pct = directional_return - 2 * per_side_cost_bps / 100
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
