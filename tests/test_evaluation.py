from baystfirm.evaluation import EvaluationRecord, PromotionGate, evaluate


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
