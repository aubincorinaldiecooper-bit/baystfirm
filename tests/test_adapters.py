import json

from baystfirm.adapters.bybit import BybitLinearAdapter
from baystfirm.adapters.coinbase import CoinbaseAdapter
from baystfirm.adapters.kraken import KrakenAdapter
from baystfirm.models import InstrumentKind, Side


def test_coinbase_trade_normalization() -> None:
    adapter = CoinbaseAdapter(("BTC-USD",))
    events = adapter.parse_message(
        json.dumps(
            {
                "type": "match",
                "trade_id": 1,
                "sequence": 2,
                "maker_order_id": "maker",
                "taker_order_id": "taker",
                "time": "2025-01-01T00:00:00.000000Z",
                "product_id": "BTC-USD",
                "size": "0.2",
                "price": "50000",
                "side": "sell",
            }
        )
    )
    assert events[0].symbol == "BTC-USD"
    assert events[0].side is Side.SELL


def test_kraken_trade_normalization() -> None:
    adapter = KrakenAdapter(("USDC-USD",))
    events = adapter.parse_message(
        json.dumps(
            {
                "channel": "trade",
                "type": "update",
                "data": [
                    {
                        "symbol": "USDC/USD",
                        "side": "buy",
                        "qty": 100,
                        "price": 0.9998,
                        "timestamp": "2025-01-01T00:00:00.000000Z",
                        "trade_id": 123,
                    }
                ],
            }
        )
    )
    assert events[0].symbol == "USDC-USD"
    assert events[0].price == 0.9998


def test_bybit_perpetual_trade_normalization() -> None:
    adapter = BybitLinearAdapter(("BTC-USDT-PERP",))
    events = adapter.parse_message(
        json.dumps(
            {
                "topic": "publicTrade.BTCUSDT",
                "data": [
                    {
                        "T": 1_735_689_600_000,
                        "s": "BTCUSDT",
                        "S": "Buy",
                        "v": "0.01",
                        "p": "50000",
                        "i": "trade-id",
                    }
                ],
            }
        )
    )
    assert events[0].instrument_kind is InstrumentKind.PERPETUAL
    assert events[0].symbol == "BTC-USDT-PERP"
