from datetime import UTC, datetime, timedelta

import pytest

from baystfirm.evaluation import OUTCOME_CLASSIFIER
from baystfirm.storage import ClassificationRow
from baystfirm.track_record import NOTE, TrackRecordService, build_track_record


def row(
    classifier: str,
    observed_at: datetime,
    *,
    label: str = "range_bound",
    horizon_seconds: int = 60,
    symbol: str = "BTC-USD",
    abstained: bool = False,
    version: str = "rules-0.1.0",
    shadow: bool = True,
    calibration_status: str = "uncalibrated",
) -> ClassificationRow:
    return (
        classifier,
        symbol,
        horizon_seconds,
        observed_at.isoformat(),
        label,
        0.8,
        int(abstained),
        15.0,
        version,
        int(shadow),
        calibration_status,
    )


def test_track_record_scores_window_pending_unmatched_hits_and_recall(monkeypatch) -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    window_start = now - timedelta(hours=1)
    monkeypatch.setitem(OUTCOME_CLASSIFIER, "momentum_regime", "momentum_outcome")
    rows = [
        row("momentum_regime", window_start - timedelta(seconds=61)),
        row("momentum_regime", window_start - timedelta(seconds=60), label="upward_momentum"),
        row(
            "momentum_regime",
            now - timedelta(seconds=180),
            label="downward_momentum",
        ),
        row(
            "momentum_regime",
            now - timedelta(seconds=150),
            abstained=True,
        ),
        row(
            "momentum_regime",
            now - timedelta(seconds=65),
            label="range_bound",
            version="rules-0.1.0",
        ),
        row(
            "momentum_regime",
            now - timedelta(seconds=60),
            label="upward_momentum",
            version="latest-version",
            shadow=False,
            calibration_status="validated",
        ),
        row(
            "momentum_outcome",
            now - timedelta(seconds=5),
            label="range_bound",
            version="outcome-v1",
        ),
    ]

    response = build_track_record(rows, 1, now=now)

    [group] = [item for item in response["groups"] if item["classifier"] == "momentum_regime"]
    assert group == {
        "classifier": "momentum_regime",
        "horizon_seconds": 60,
        "classifier_version": "latest-version",
        "shadow": False,
        "calibration_status": "validated",
        "predictions": 4,
        "abstained": 1,
        "scored": 1,
        "unmatched": 2,
        "pending": 1,
        "hits": 1,
        "hit_rate": 1,
        "hit_rate_ci95": pytest.approx([0.2065432915, 1.0]),
        "baseline_hit_rate": 1,
        "label_recall": {"range_bound": 1},
    }
    assert response["computed_at"] == now.isoformat()
    assert response["window_hours"] == 1
    assert response["window_start"] == window_start.isoformat()
    assert response["note"] == NOTE
    assert {item["classifier"] for item in response["groups"]} == {
        "momentum_outcome",
        "momentum_regime",
    }


def test_track_record_window_includes_lower_boundary_and_ignores_earlier_maturity() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    window_start = now - timedelta(hours=1)
    rows = [
        row("momentum_regime", window_start - timedelta(seconds=61)),
        row("momentum_regime", window_start - timedelta(seconds=60)),
    ]

    response = build_track_record(rows, 1, now=now)

    [group] = response["groups"]
    assert group["predictions"] == 1
    assert group["unmatched"] == 1


def test_track_record_accepts_five_second_outcome_tolerance_only() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    rows = [
        row("momentum_regime", now - timedelta(seconds=125), label="upward_momentum"),
        row("momentum_regime", now - timedelta(seconds=115), label="downward_momentum"),
        row("momentum_regime", now - timedelta(seconds=60), label="upward_momentum"),
        row("momentum_regime", now - timedelta(seconds=49), label="downward_momentum"),
    ]

    response = build_track_record(rows, 1, now=now)

    [group] = response["groups"]
    assert group["predictions"] == 2
    assert group["scored"] == 1
    assert group["unmatched"] == 1
    assert group["hits"] == 1


@pytest.mark.asyncio
async def test_track_record_cache_is_per_window_hours() -> None:
    class StoreStub:
        def __init__(self) -> None:
            self.calls = 0

        async def classification_rows_since(self, since: datetime) -> list[ClassificationRow]:
            self.calls += 1
            return []

    store = StoreStub()
    service = TrackRecordService(store)  # type: ignore[arg-type]

    await service.get(24)
    await service.get(24)
    await service.get(1)

    assert store.calls == 2
