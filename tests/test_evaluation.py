from datetime import UTC, datetime, timedelta

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
