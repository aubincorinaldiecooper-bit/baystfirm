import json

from baystfirm.adapters.bybit import BybitLinearAdapter, BybitSpotAdapter
from baystfirm.adapters.coinbase import CoinbaseAdapter
from baystfirm.adapters.kraken import KrakenAdapter
from baystfirm.adapters.okx import OkxSpotAdapter
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


def test_bybit_spot_stablecoin_normalization() -> None:
    adapter = BybitSpotAdapter(("USDC-USDT", "BTC-USD"))
    assert adapter.symbols == ("USDCUSDT",)
    events = adapter.parse_message(
        json.dumps(
            {
                "topic": "publicTrade.USDCUSDT",
                "data": [
                    {"T": 1_735_689_600_000, "s": "USDCUSDT", "S": "Sell", "v": "55", "p": "1.0001"}
                ],
            }
        )
    )
    assert events[0].symbol == "USDC-USDT"
    assert events[0].instrument_kind is InstrumentKind.SPOT


def test_okx_spot_normalization() -> None:
    adapter = OkxSpotAdapter(("USDC-USDT", "BTC-USD", "BTC-USDT-PERP"))
    assert adapter.symbols == ("USDC-USDT",)
    assert adapter.parse_message("pong") == []
    events = adapter.parse_message(
        json.dumps(
            {
                "arg": {"channel": "trades", "instId": "USDC-USDT"},
                "data": [
                    {
                        "instId": "USDC-USDT",
                        "tradeId": "1",
                        "px": "1.00017",
                        "sz": "500",
                        "side": "sell",
                        "ts": "1790960686678",
                        "seqId": 9,
                    }
                ],
            }
        )
    )
    assert events[0].quote_asset == "USDT"
    assert events[0].side is Side.SELL
