from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EvaluationRecord:
    predicted_label: str
    expected_label: str
    probability: float
    abstained: bool
    latency_ms: float
    normal_label: str

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.expected_label


@dataclass(frozen=True)
class EvaluationMetrics:
    sample_count: int
    coverage: float
    accuracy: float
    false_alert_rate: float
    brier_score: float
    expected_calibration_error: float
    p95_latency_ms: float


@dataclass(frozen=True)
class PromotionGate:
    minimum_samples: int = 500
    minimum_coverage: float = 0.8
    minimum_accuracy: float = 0.9
    maximum_false_alert_rate: float = 0.05
    maximum_brier_score: float = 0.15
    maximum_ece: float = 0.1
    maximum_p95_latency_ms: float = 250

    def failures(self, metrics: EvaluationMetrics) -> list[str]:
        checks = (
            (metrics.sample_count < self.minimum_samples, "insufficient_samples"),
            (metrics.coverage < self.minimum_coverage, "coverage_below_threshold"),
            (metrics.accuracy < self.minimum_accuracy, "accuracy_below_threshold"),
            (
                metrics.false_alert_rate > self.maximum_false_alert_rate,
                "false_alert_rate_above_threshold",
            ),
            (metrics.brier_score > self.maximum_brier_score, "brier_score_above_threshold"),
            (
                metrics.expected_calibration_error > self.maximum_ece,
                "calibration_error_above_threshold",
            ),
            (metrics.p95_latency_ms > self.maximum_p95_latency_ms, "latency_above_threshold"),
        )
        return [name for failed, name in checks if failed]


def evaluate(records: list[EvaluationRecord], bins: int = 10) -> EvaluationMetrics:
    if not records:
        raise ValueError("at least one evaluation record is required")
    if bins <= 0:
        raise ValueError("bins must be positive")
    covered = [record for record in records if not record.abstained]
    coverage = len(covered) / len(records)
    accuracy = mean(record.correct for record in covered) if covered else 0.0
    normal = [record for record in covered if record.expected_label == record.normal_label]
    false_alert_rate = (
        mean(record.predicted_label != record.normal_label for record in normal) if normal else 0.0
    )
    brier = (
        mean((record.probability - float(record.correct)) ** 2 for record in covered)
        if covered
        else 1.0
    )
    ece = _expected_calibration_error(covered, bins)
    latencies = sorted(record.latency_ms for record in records)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    return EvaluationMetrics(
        sample_count=len(records),
        coverage=coverage,
        accuracy=accuracy,
        false_alert_rate=false_alert_rate,
        brier_score=brier,
        expected_calibration_error=ece,
        p95_latency_ms=latencies[p95_index],
    )


def _expected_calibration_error(records: list[EvaluationRecord], bins: int) -> float:
    if not records:
        return 1.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            record
            for record in records
            if lower <= record.probability < upper
            or (index == bins - 1 and record.probability == 1)
        ]
        if not bucket:
            continue
        confidence = mean(record.probability for record in bucket)
        accuracy = mean(record.correct for record in bucket)
        error += len(bucket) / len(records) * abs(confidence - accuracy)
    return error
