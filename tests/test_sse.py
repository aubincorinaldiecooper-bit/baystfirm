import asyncio
import json

from baystfirm.hub import EventHub
from baystfirm.sse import sse_frames
from tests.test_classifiers import stablecoin_trade


async def test_sse_frames_filter_symbols_and_keep_alive() -> None:
    hub = EventHub()
    frames = sse_frames(hub, symbols=["usdc-usd"], keepalive_seconds=0.05)
    assert await anext(frames) == "retry: 2000\n\n"
    pending = asyncio.ensure_future(anext(frames))
    await asyncio.sleep(0)
    await hub.publish(stablecoin_trade("kraken", 1.0, symbol="USDT-USD"))
    await hub.publish(stablecoin_trade("coinbase", 1.0001))
    frame = await pending
    name, data = frame.strip().split("\n")
    assert name == "event: market_event"
    assert json.loads(data.removeprefix("data: "))["symbol"] == "USDC-USD"
    assert await anext(frames) == ": keepalive\n\n"
    await frames.aclose()
    assert hub.subscriber_count == 0
