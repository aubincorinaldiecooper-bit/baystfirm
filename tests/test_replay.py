from pathlib import Path

from baystfirm.replay import ReplayRunner
from baystfirm.storage import EventStore
from tests.test_models import make_trade


async def test_replay_emits_stored_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    await store.open()
    event = make_trade()
    await store.append_event(event)
    seen = []

    async def capture(item) -> None:
        seen.append(item)

    count = await ReplayRunner(store).run(capture)
    assert count == 1
    assert seen == [event]
    await store.close()
