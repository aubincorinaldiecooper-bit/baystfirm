from pathlib import Path

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
