from __future__ import annotations

import io
import math

import pytest

from baystfirm.vision.chart import (
    HORIZON_SECONDS,
    MIN_TRADES,
    ChartSample,
    Tick,
    build_samples,
    momentum_label,
    render_chart,
)
from baystfirm.vision.chart_train import (
    _balance_train_samples,
    _compute_logit_adjust,
    _fit_adjusted_temperature,
)


def _window_ticks(
    start_ms: int = 0,
    *,
    count: int = 31,
    duration_ms: int = 30_000,
    first_price: float = 100.0,
    last_price: float = 100.25,
) -> list[Tick]:
    return [
        Tick(
            start_ms + index * duration_ms // (count - 1),
            first_price + (last_price - first_price) * index / (count - 1),
            1.0,
        )
        for index in range(count)
    ]


def test_momentum_label_uses_exact_thresholds() -> None:
    upward = momentum_label(_window_ticks(), 30_000)
    assert upward is not None
    assert upward[0] == "upward_momentum"
    assert upward[1] == pytest.approx(25.0)
    downward = _window_ticks(first_price=100.0, last_price=99.75)
    result = momentum_label(downward, 30_000)
    assert result is not None
    assert result[0] == "downward_momentum"
    assert result[1] == pytest.approx(-25.0)


def test_momentum_label_abstains_for_insufficient_or_invalid_windows() -> None:
    assert momentum_label(_window_ticks(count=MIN_TRADES - 1), 30_000) is None
    assert momentum_label(_window_ticks(duration_ms=9_000), 30_000) is None
    assert momentum_label(_window_ticks(first_price=0.0, last_price=0.0), 30_000) is None


def test_future_ticks_do_not_change_rendered_chart() -> None:
    observed = _window_ticks(start_ms=1_000_000)
    anchor_ms = observed[-1].ts_ms
    future = observed + [Tick(anchor_ms + 1, 1_000_000.0, 1_000_000.0)]
    original_bytes = io.BytesIO()
    changed_bytes = io.BytesIO()
    render_chart(observed, anchor_ms).save(original_bytes, format="PNG")
    render_chart(future, anchor_ms).save(changed_bytes, format="PNG")
    assert original_bytes.getvalue() == changed_bytes.getvalue()


def test_build_samples_stride_skip_and_deterministic_subsample() -> None:
    ticks = [Tick(ts_ms, 100.0, 1.0) for ts_ms in range(0, 700_000, 1000)]
    samples = build_samples(ticks, "BTC-USDT", "train", stride_seconds=60, max_samples=3)
    assert [sample.anchor_ms for sample in samples] == [300_000, 480_000, 660_000]
    assert samples == build_samples(ticks, "BTC-USDT", "train", stride_seconds=60, max_samples=3)
    sparse = [tick for tick in ticks if not 360_000 <= tick.ts_ms < 400_000]
    skipped = build_samples(sparse, "BTC-USDT", "train", stride_seconds=60, max_samples=100)
    assert 360_000 not in {sample.anchor_ms for sample in skipped}
    assert all(sample.label == "range_bound" for sample in samples)


def test_build_samples_can_keep_more_than_the_old_default_cap() -> None:
    ticks = [Tick(ts_ms, 100.0, 1.0) for ts_ms in range(0, 1_400_000, 1000)]
    samples = build_samples(ticks, "BTC-USDT", "train", stride_seconds=1, max_samples=None)
    assert len(samples) > 1000


def _sample(anchor_ms: int, label: str) -> ChartSample:
    return ChartSample(
        symbol="BTC-USDT",
        anchor_ms=anchor_ms,
        split="train",
        label=label,
        outcome_bps=0.0,
        persistence_label=None,
        persistence_bps=None,
        evidence={},
    )


def test_train_balancing_keeps_movement_and_caps_range_candidates() -> None:
    candidates = [
        _sample(index, label)
        for index, label in enumerate(
            [
                "range_bound",
                "upward_momentum",
                "range_bound",
                "downward_momentum",
                "range_bound",
                "upward_momentum",
                "range_bound",
                "downward_momentum",
                "range_bound",
                "range_bound",
                "range_bound",
                "range_bound",
                "range_bound",
                "range_bound",
                "range_bound",
                "range_bound",
            ]
        )
    ]
    selected = _balance_train_samples(candidates)
    movement = [sample for sample in selected if sample.label != "range_bound"]
    range_bound = [sample for sample in selected if sample.label == "range_bound"]
    assert {sample.anchor_ms for sample in movement} == {1, 3, 5, 7}
    assert len(range_bound) == 2 * len(movement)


def test_logit_adjust_uses_natural_and_selected_train_priors() -> None:
    adjustment = _compute_logit_adjust(
        {
            "natural_train_counts": {
                "upward_momentum": 10,
                "downward_momentum": 5,
                "range_bound": 85,
            },
            "selected_counts": {
                "train": {
                    "upward_momentum": 10,
                    "downward_momentum": 5,
                    "range_bound": 30,
                }
            },
        }
    )
    assert adjustment == pytest.approx(
        [
            math.log(0.1) - math.log(10 / 45),
            math.log(0.05) - math.log(5 / 45),
            math.log(0.85) - math.log(30 / 45),
        ]
    )


def test_render_chart_is_deterministic_448_rgb_and_clips_price_bps() -> None:
    ticks = [
        Tick(1_000_000, 200.0, 1.0),
        Tick(1_299_000, 100.0, 1.0),
    ]
    image = render_chart(ticks, 1_300_000)
    first = io.BytesIO()
    second = io.BytesIO()
    image.save(first, format="PNG")
    render_chart(ticks, 1_300_000).save(second, format="PNG")
    assert image.size == (448, 448)
    assert image.mode == "RGB"
    assert first.getvalue() == second.getvalue()
    assert any(image.getpixel((x, y)) == (0, 0, 0) for x in range(448) for y in range(3))


def test_outcome_and_persistence_use_distinct_30_second_windows() -> None:
    anchor_ms = 300_000
    ticks = [
        Tick(
            ts_ms,
            100 + (ts_ms / 1_000_000)
            if ts_ms <= anchor_ms
            else 100.3 + (ts_ms - anchor_ms) / 100_000,
            1.0,
        )
        for ts_ms in range(anchor_ms - 30_000, anchor_ms + 30_001, 1000)
    ]
    persistence = momentum_label(ticks, anchor_ms)
    outcome = momentum_label(ticks, anchor_ms + HORIZON_SECONDS * 1000)
    assert persistence is not None
    assert outcome is not None
    assert persistence[1] < outcome[1]


def test_chart_head_forward_mask_temperature_and_abstention() -> None:
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    from baystfirm.vision.chart_head import ChartStateHead

    torch.manual_seed(7)
    head = ChartStateHead(hidden_size=16, proj=8)
    embeds = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    output = head(embeds, mask)
    assert output.shape == (2, 3)
    changed = embeds.clone()
    changed[0, 2:] = 1e5
    assert torch.allclose(output[0], head(changed, mask)[0])
    bias = torch.tensor([0.25, -0.5, 1.0])
    head.logit_bias.copy_(bias)
    assert torch.allclose(head(embeds, mask), output + bias)
    head.logit_bias.zero_()

    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1, 2, 1])
    before = F.cross_entropy(logits, targets).item()
    head.fit_temperature(logits, targets)
    after = F.cross_entropy(logits / head.temperature, targets).item()
    assert after < before
    assert head.decide(torch.tensor([0.49, 0.31, 0.20])) == (
        "upward_momentum",
        pytest.approx(0.49),
        True,
    )


def test_temperature_fit_uses_bias_adjusted_logits() -> None:
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    from baystfirm.vision.chart_head import ChartStateHead

    head = ChartStateHead(hidden_size=8, proj=4)
    raw_logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]
    )
    targets = torch.tensor([0, 1, 2, 1])
    logit_adjust = torch.tensor([0.6, -0.4, 0.2])
    adjusted_logits = _fit_adjusted_temperature(head, raw_logits, targets, logit_adjust)
    assert torch.allclose(adjusted_logits, raw_logits + logit_adjust)
    assert torch.equal(head.logit_bias, logit_adjust)
    before = F.cross_entropy(adjusted_logits, targets)
    after = F.cross_entropy(adjusted_logits / head.temperature, targets)
    assert after < before
