from datetime import UTC, datetime, timedelta
from pathlib import Path

from baystfirm.models import Classification
from baystfirm.storage import EventStore
from tests.test_models import make_trade


async def test_event_round_trip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    await store.open()
    event = make_trade()
    await store.append_event(event)

    loaded = [item async for item in store.iter_events(symbol="BTC-USD")]
    assert loaded == [event]
    assert await store.event_count() == 1
    await store.close()


async def test_classification_rows_since_uses_indexed_columns_and_payload_fields(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "classifications.db")
    await store.open()
    observed_at = datetime(2025, 1, 1, tzinfo=UTC)
    classification = Classification(
        classifier="momentum_regime",
        classifier_version="rules-0.1.0",
        symbol="BTC-USD",
        label="range_bound",
        probability=0.8,
        abstained=False,
        horizon_seconds=60,
        observed_at=observed_at,
        evidence=[],
        shadow=True,
        freshness_ms=12.5,
    )
    await store.append_classification(classification)

    rows = await store.classification_rows_since(observed_at - timedelta(seconds=1))

    assert rows == [
        (
            "momentum_regime",
            "BTC-USD",
            60,
            observed_at.isoformat(),
            "range_bound",
            0.8,
            0,
            12.5,
            "rules-0.1.0",
            1,
            "uncalibrated",
        )
    ]
    await store.close()
