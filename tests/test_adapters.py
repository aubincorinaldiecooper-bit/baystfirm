import json
from pathlib import Path

from baystfirm.adapters.binanceus import BinanceUSAdapter
from baystfirm.adapters.bybit import BybitLinearAdapter, BybitSpotAdapter
from baystfirm.adapters.coinbase import CoinbaseAdapter
from baystfirm.adapters.kraken import KrakenAdapter
from baystfirm.adapters.okx import OkxSpotAdapter, OkxSwapAdapter
from baystfirm.config import DEFAULT_VENUES, Settings
from baystfirm.ingestion import build_adapters
from baystfirm.models import EventType, InstrumentKind, Side

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_messages(name: str) -> list[str]:
    frames = json.loads((FIXTURES / name).read_text())
    return [json.dumps(frame) for frame in frames]


def test_okx_swap_adapter_is_registered() -> None:
    settings = Settings(
        database_path=Path(":memory:"),
        enabled_venues=("okx",),
        shadow_mode=True,
        symbols=("BTC-USDT", "BTC-USDT-PERP"),
    )
    adapters = build_adapters(settings)
    assert [type(adapter) for adapter in adapters] == [OkxSpotAdapter, OkxSwapAdapter]


def test_binanceus_adapter_uses_only_verified_spot_symbols() -> None:
    settings = Settings(
        database_path=Path(":memory:"),
        enabled_venues=("binanceus",),
        shadow_mode=True,
        symbols=("BTC-USDT", "BTC-USDT-PERP", "PYUSD-USD", "USDC-USDT"),
    )
    adapters = build_adapters(settings)
    assert "binanceus" in DEFAULT_VENUES
    assert len(adapters) == 1
    assert isinstance(adapters[0], BinanceUSAdapter)
    assert adapters[0].symbols == ("BTCUSDT", "USDCUSDT")


def test_coinbase_trade_fixture_normalization() -> None:
    adapter = CoinbaseAdapter(("BTC-USD",))
    assert adapter.subscription_messages() == [
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]},
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]},
    ]
    event = adapter.parse_message(fixture_messages("coinbase_matches.json")[0])[0]
    assert event.event_type is EventType.TRADE
    assert event.symbol == "BTC-USD"
    assert event.sequence == 137234410134
    assert event.metadata["trade_id"] == 1102238919
    assert event.price == 85324.99
    assert event.size == 0.00000005
    assert event.side is Side.BUY


def test_coinbase_ticker_fixture_normalization() -> None:
    adapter = CoinbaseAdapter(("BTC-USD",))
    event = adapter.parse_message(fixture_messages("coinbase_ticker.json")[0])[0]
    assert event.event_type is EventType.QUOTE
    assert event.symbol == "BTC-USD"
    assert event.sequence == 137234409912
    assert event.bid == 85324.99
    assert event.ask == 85325.0
    assert event.bid_size == 1.08296808
    assert event.ask_size == 0.01692372


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


def test_bybit_orderbook_subscriptions_use_documented_maximum() -> None:
    spot = BybitSpotAdapter(("BTC-USDT",))
    linear = BybitLinearAdapter(("BTC-USDT-PERP",))
    assert "orderbook.1000.BTCUSDT" in spot.subscription_messages()[0]["args"]
    assert "orderbook.1000.BTCUSDT" in linear.subscription_messages()[0]["args"]


def test_bybit_ticker_delta_merges_cached_snapshot() -> None:
    adapter = BybitLinearAdapter(("BTC-USDT-PERP",))
    snapshot, delta = fixture_messages("bybit_linear_tickers.json")
    initial = adapter.parse_message(snapshot)
    updated = adapter.parse_message(delta)

    initial_funding = next(event for event in initial if event.event_type is EventType.FUNDING)
    funding = next(event for event in updated if event.event_type is EventType.FUNDING)
    open_interest = next(event for event in initial if event.event_type is EventType.OPEN_INTEREST)
    assert funding.index_price == 85370.46
    assert funding.funding_rate == initial_funding.funding_rate
    assert funding.mark_price == initial_funding.mark_price
    assert funding.next_funding_at == initial_funding.next_funding_at
    assert open_interest.open_interest == 55812.808
    assert open_interest.open_interest_value == 4762545975.61
    assert not any(event.event_type is EventType.OPEN_INTEREST for event in updated)


def test_bybit_orderbook_snapshot_delta_delete_and_reset() -> None:
    linear = BybitLinearAdapter(("BTC-USDT-PERP",))
    linear_snapshot, linear_delta = fixture_messages("bybit_linear_book.json")
    initial = linear.parse_message(linear_snapshot)[0]
    updated = linear.parse_message(linear_delta)[0]
    assert initial.event_type is EventType.BOOK
    assert initial.depth_levels == 50
    assert initial.bid_depth_10bps is None
    assert initial.ask_depth_10bps is None
    assert initial.bid_depth_50bps is None
    assert initial.ask_depth_50bps is None
    assert 85322.1 in linear._books["BTCUSDT"].bids
    assert updated.bid == initial.bid

    spot = BybitSpotAdapter(("BTC-USDT",))
    snapshot, delta = fixture_messages("bybit_spot_book.json")
    spot.parse_message(snapshot)
    spot.parse_message(delta)
    book = spot._books["BTCUSDT"]
    assert 85396.6 not in book.asks
    spot.parse_message(
        json.dumps(
            {
                "topic": "orderbook.50.BTCUSDT",
                "type": "delta",
                "ts": 1791135349300,
                "data": {"s": "BTCUSDT", "b": [], "a": [["85390", "0.2"]], "u": 1},
            }
        )
    )
    assert 85390.0 in book.asks
    reset = json.loads(snapshot)
    reset["data"]["b"] = reset["data"]["b"][:1]
    reset["data"]["a"] = reset["data"]["a"][:1]
    reset["ts"] += 100
    reset["type"] = "snapshot"
    spot.parse_message(json.dumps(reset))
    assert 85390.0 not in book.asks
    assert len(book.bids) == len(book.asks) == 1


def test_kraken_book_fixture_snapshot_and_update() -> None:
    adapter = KrakenAdapter(("BTC-USD",))
    assert adapter.subscription_messages()[1]["params"]["depth"] == 1000
    snapshot, update = fixture_messages("kraken_book.json")
    initial = adapter.parse_message(snapshot)[0]
    changed = adapter.parse_message(update)[0]
    book = adapter._books["BTC/USD"]
    assert initial.depth_levels == 25
    assert initial.bid_depth_10bps is None
    assert initial.ask_depth_10bps is None
    assert initial.bid_depth_50bps is None
    assert initial.ask_depth_50bps is None
    assert changed.event_type is EventType.BOOK
    assert 85344.4 not in book.bids
    assert 85339.1 in book.bids


def test_binanceus_partial_depth_fixture_normalization() -> None:
    adapter = BinanceUSAdapter(
        (
            "BTC-USDT",
            "BTC-USD",
            "SOL-USDT",
            "BTC-USDT-PERP",
            "PYUSD-USD",
            "USDC-USDT",
        )
    )
    assert adapter.symbols == ("BTCUSDT", "BTCUSD", "SOLUSDT", "USDCUSDT")
    assert "btcusdt@trade" in adapter.connection_url()
    assert "btcusdt@depth20@100ms" in adapter.connection_url()
    assert "btcusdtperp" not in adapter.connection_url()
    trade_raw = json.dumps(json.loads((FIXTURES / "binanceus_trade.json").read_text()))
    trade = adapter.parse_message(trade_raw)[0]
    assert trade.venue == "binanceus"
    assert trade.symbol == "SOL-USDT"
    assert trade.event_type is EventType.TRADE
    assert trade.side is Side.BUY
    assert trade.sequence == 11909392
    assert trade.price == 121.47
    assert trade.size == 0.272
    assert trade.metadata["buyer_is_maker"] is False
    raw = json.dumps(json.loads((FIXTURES / "binanceus_depth20.json").read_text()))
    event = adapter.parse_message(raw)[0]
    assert event.venue == "binanceus"
    assert event.symbol == "BTC-USDT"
    assert event.event_type is EventType.BOOK
    assert event.depth_levels == 20
    assert event.bid == 85330.76
    assert event.ask == 85330.77
    assert event.bid_depth_10bps is None
    assert event.ask_depth_50bps is None
    assert not hasattr(adapter, "_books")


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


def test_okx_books_spot_snapshot_update_delete_and_reset() -> None:
    adapter = OkxSpotAdapter(("BTC-USDT",))
    subscription = adapter.subscription_messages()[0]["args"]
    assert {"channel": "books", "instId": "BTC-USDT"} in subscription
    assert not any(argument["channel"] == "books5" for argument in subscription)
    snapshot, update = fixture_messages("okx_books_spot.json")
    initial = adapter.parse_message(snapshot)[0]
    changed = adapter.parse_message(update)[0]
    book = adapter._books["BTC-USDT"]
    assert initial.event_type is EventType.BOOK
    assert initial.depth_levels == 400
    assert initial.bid == 85325.2
    assert initial.ask is not None
    assert initial.bid_depth_10bps is not None
    assert initial.ask_depth_10bps is not None
    assert initial.bid_depth_50bps is None
    assert initial.ask_depth_50bps is None
    assert changed.depth_levels == 400
    assert 85313.1 not in book.bids
    assert 85497.6 not in book.asks

    reset = json.loads(snapshot)
    reset["data"][0]["bids"] = reset["data"][0]["bids"][:1]
    reset["data"][0]["asks"] = reset["data"][0]["asks"][:1]
    reset["data"][0]["ts"] = str(int(reset["data"][0]["ts"]) + 100)
    adapter.parse_message(json.dumps(reset))
    assert len(book.bids) == len(book.asks) == 1


def test_okx_swap_fixtures_normalize_derivatives_and_books() -> None:
    adapter = OkxSwapAdapter(("BTC-USDT-PERP",))
    subscription = adapter.subscription_messages()[0]["args"]
    assert {"channel": "books", "instId": "BTC-USDT-SWAP"} in subscription
    assert not any(argument["channel"] == "books5" for argument in subscription)
    funding = adapter.parse_message(fixture_messages("okx_funding_rate.json")[0])[0]
    mark = adapter.parse_message(fixture_messages("okx_mark_price.json")[0])[0]
    index = adapter.parse_message(fixture_messages("okx_index_tickers.json")[0])[0]
    assert funding.event_type is EventType.FUNDING
    assert funding.funding_rate == 0.0000614463111508
    assert funding.next_funding_at is not None
    assert mark.mark_price == 85324.4
    assert index.index_price == 85370.7
    assert index.funding_rate == funding.funding_rate
    assert index.mark_price == mark.mark_price

    book_snapshot, book_update = fixture_messages("okx_books_swap.json")
    assert adapter.parse_message(book_snapshot) == []
    assert adapter.parse_message(book_update) == []
    oi_events = adapter.parse_message(fixture_messages("okx_open_interest.json")[0])
    open_interest = next(
        event for event in oi_events if event.event_type is EventType.OPEN_INTEREST
    )
    book = next(event for event in oi_events if event.event_type is EventType.BOOK)
    assert open_interest.open_interest == 2854317.50000000943
    assert open_interest.open_interest_value == 2435429280.97000804609092
    assert book.depth_levels == 400
    assert book.bid_size == 5.39
    assert book.metadata["contract_multiplier"] == 0.01
    assert book.bid_depth_10bps is None
    assert book.ask_depth_10bps is None
    assert book.bid_depth_50bps is None
    assert book.ask_depth_50bps is None


def test_okx_liquidation_fixture_filters_global_channel_by_subscription() -> None:
    raw = fixture_messages("okx_liquidation_orders.json")[0]
    btc = OkxSwapAdapter(("BTC-USDT-PERP",))
    assert btc.parse_message(raw) == []

    grass = OkxSwapAdapter(("GRASS-USDT-PERP",))
    event = grass.parse_message(raw)[0]
    assert event.event_type is EventType.LIQUIDATION
    assert event.symbol == "GRASS-USDT-PERP"
    assert event.price == 0.6806
    assert event.size == 338
    assert event.side is Side.SELL
    assert event.metadata["position_side"] == "long"
