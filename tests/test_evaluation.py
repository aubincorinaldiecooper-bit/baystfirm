from datetime import UTC, datetime, timedelta

import pytest

from baystfirm.evaluation import (
    EvaluationRecord,
    PromotionGate,
    evaluate,
    evaluate_by_classifier,
    label_classifications,
)
from baystfirm.models import Classification


def test_evaluation_metrics_and_promotion_gate() -> None:
    records = [EvaluationRecord("pegged", "pegged", 0.95, False, 10, "pegged") for _ in range(500)]
    metrics = evaluate(records)
    assert metrics.accuracy == 1
    assert metrics.coverage == 1
    assert metrics.p95_latency_ms == 10
    assert PromotionGate().failures(metrics) == []


def test_abstentions_reduce_coverage() -> None:
    records = [
        EvaluationRecord("unknown", "pegged", 0.5, True, 10, "pegged"),
        EvaluationRecord("pegged", "pegged", 0.9, False, 20, "pegged"),
    ]
    metrics = evaluate(records)
    assert metrics.coverage == 0.5
    assert metrics.accuracy == 1


def test_always_normal_fails_macro_recall_on_imbalanced_data() -> None:
    records = [
        EvaluationRecord("range_bound", label, 0.98, False, 10, "range_bound")
        for label, count in (
            ("range_bound", 980),
            ("downward_momentum", 10),
            ("upward_momentum", 10),
        )
        for _ in range(count)
    ]
    metrics = evaluate(records)
    assert metrics.accuracy == 0.98
    assert metrics.macro_recall == pytest.approx(1 / 3)
    assert PromotionGate().failures(metrics) == ["macro_recall_below_threshold"]


def test_abstentions_count_as_recall_misses() -> None:
    metrics = evaluate(
        [
            EvaluationRecord("upward_momentum", "upward_momentum", 0.9, True, 10, "range_bound"),
            EvaluationRecord("range_bound", "range_bound", 0.9, False, 10, "range_bound"),
        ]
    )
    assert metrics.label_recall == {"range_bound": 1.0, "upward_momentum": 0.0}
    assert metrics.macro_recall == 0.5


def test_recall_uses_only_labels_present_in_expected_records() -> None:
    metrics = evaluate(
        [EvaluationRecord("upward_momentum", "upward_momentum", 0.9, False, 10, "range_bound")]
    )
    assert metrics.label_recall == {"upward_momentum": 1.0}
    assert metrics.macro_recall == 1.0


def prediction(label: str, seconds: int, abstained: bool = False) -> Classification:
    return Classification(
        classifier="stablecoin_peg",
        classifier_version="test",
        symbol="USDC",
        label=label,
        probability=0.9,
        abstained=abstained,
        horizon_seconds=30,
        observed_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        evidence=[],
        freshness_ms=12,
    )


def test_labels_come_from_state_one_horizon_later() -> None:
    records = label_classifications(
        [
            prediction("pegged", 0),
            prediction("insufficient_cross_venue_data", 1, abstained=True),
            prediction("peg_watch", 30),
            prediction("pegged", 31),
            prediction("pegged", 90),
        ]
    )
    assert [(r.predicted_label, r.expected_label) for r in records] == [
        ("pegged", "peg_watch"),
        ("insufficient_cross_venue_data", "pegged"),
    ]
    metrics = evaluate_by_classifier(records)["stablecoin_peg"]
    assert metrics.coverage == 0.5
    assert metrics.accuracy == 0


def test_chart_classifier_uses_rules_classifier_as_realized_outcome() -> None:
    chart = Classification(
        classifier="baystfirm_chart_momentum",
        classifier_version="shadow-v1",
        symbol="BTC-USDT",
        label="upward_momentum",
        probability=0.72,
        abstained=False,
        horizon_seconds=30,
        observed_at=datetime(2025, 1, 1, tzinfo=UTC),
        evidence=[],
        freshness_ms=0,
    )
    realized = Classification(
        classifier="short_horizon_momentum",
        classifier_version="rules-0.2.1",
        symbol="BTC-USDT",
        label="range_bound",
        probability=0.8,
        abstained=False,
        horizon_seconds=30,
        observed_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
        evidence=[],
        freshness_ms=0,
    )
    later_chart_prediction = Classification(
        classifier="baystfirm_chart_momentum",
        classifier_version="shadow-v1",
        symbol="BTC-USDT",
        label="downward_momentum",
        probability=0.81,
        abstained=False,
        horizon_seconds=30,
        observed_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
        evidence=[],
        freshness_ms=0,
    )
    record = label_classifications([chart, later_chart_prediction, realized])[0]
    assert record.predicted_label == "upward_momentum"
    assert record.expected_label == "range_bound"
    assert record.normal_label == "range_bound"
    assert record.classifier == "baystfirm_chart_momentum"


def test_multi_horizon_predictions_only_match_same_horizon_outcomes() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)

    def regime(label: str, seconds: int, horizon: int) -> Classification:
        return Classification(
            classifier="momentum_regime",
            classifier_version="rules-0.1.0",
            symbol="BTC-USDT",
            label=label,
            probability=0.8,
            abstained=False,
            horizon_seconds=horizon,
            observed_at=start + timedelta(seconds=seconds),
            evidence=[],
            freshness_ms=10,
        )

    records = label_classifications(
        [
            regime("upward_momentum", 0, 60),
            regime("upward_momentum", 0, 300),
            regime("downward_momentum", 60, 300),
            regime("range_bound", 300, 300),
        ]
    )

    assert [(record.horizon_seconds, record.expected_label) for record in records] == [
        (300, "range_bound")
    ]


def test_multi_horizon_evaluation_key_includes_horizon() -> None:
    record = EvaluationRecord(
        "upward_momentum",
        "upward_momentum",
        0.8,
        False,
        10,
        "range_bound",
        classifier="momentum_regime",
        horizon_seconds=3600,
    )
    assert set(evaluate_by_classifier([record])) == {"momentum_regime:3600s"}
