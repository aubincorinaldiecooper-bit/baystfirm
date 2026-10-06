from datetime import UTC, datetime, timedelta
from pathlib import Path

import baystfirm.storage as storage_module
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


async def test_prune_events_deletes_old_rows_across_batches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "PRUNE_BATCH_SIZE", 2)
    store = EventStore(tmp_path / "prune-events.db")
    await store.open()
    cutoff = datetime(2025, 1, 2, tzinfo=UTC)
    old_events = [
        make_trade().model_copy(
            update={"received_timestamp": cutoff - timedelta(minutes=index + 1)}
        )
        for index in range(5)
    ]
    retained = make_trade().model_copy(update={"received_timestamp": cutoff})
    await store.append_events([*old_events, retained])

    assert await store.prune_events(cutoff) == 5
    remaining = [event async for event in store.iter_events()]

    assert remaining == [retained]
    await store.close()


async def test_prune_classifications_deletes_old_rows_across_batches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(storage_module, "PRUNE_BATCH_SIZE", 2)
    store = EventStore(tmp_path / "prune-classifications.db")
    await store.open()
    cutoff = datetime(2025, 1, 2, tzinfo=UTC)
    old_classifications = [
        Classification(
            classifier="momentum_regime",
            classifier_version="rules-0.1.0",
            symbol=f"OLD{index}-USD",
            label="range_bound",
            probability=0.8,
            abstained=False,
            horizon_seconds=60,
            observed_at=cutoff - timedelta(minutes=index + 1),
            evidence=[],
            shadow=True,
            freshness_ms=12.5,
        )
        for index in range(5)
    ]
    retained = Classification(
        classifier="momentum_regime",
        classifier_version="rules-0.1.0",
        symbol="NEW-USD",
        label="range_bound",
        probability=0.8,
        abstained=False,
        horizon_seconds=60,
        observed_at=cutoff,
        evidence=[],
        shadow=True,
        freshness_ms=12.5,
    )
    for classification in [*old_classifications, retained]:
        await store.append_classification(classification)

    assert await store.prune_classifications(cutoff) == 5
    remaining = [item async for item in store.iter_classifications()]

    assert remaining == [retained]
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
