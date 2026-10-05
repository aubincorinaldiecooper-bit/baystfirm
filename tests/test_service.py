from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from solders.pubkey import Pubkey

import baystfirm.candles as candle_module
import baystfirm.service as service_module
from baystfirm.config import Settings
from baystfirm.models import Candle
from baystfirm.service import create_app
from baystfirm.solana_tokens import (
    NotTokenMint,
    SolanaTokenClient,
    SolanaTokenEngine,
    SourceError,
)
from tests.test_classifiers import stablecoin_trade


def test_health_and_evaluation_gate(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            database_path=tmp_path / "service.db",
            enabled_venues=(),
            shadow_mode=True,
            symbols=(),
            solana_tokens_enabled=False,
        )
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["shadow_mode"] is True

        gate = client.get("/v1/evaluation/gate")
        assert gate.status_code == 200
        assert gate.json()["status"] == "shadow"
        assert gate.json()["runs"] == []
        assert gate.json()["thresholds"]["minimum_macro_recall"] == 0.5


def _settings(tmp_path: Path, api_key: str | None = None) -> Settings:
    return Settings(
        database_path=tmp_path / "service.db",
        enabled_venues=(),
        shadow_mode=True,
        symbols=("USDC-USD",),
        api_key=api_key,
        solana_tokens_enabled=False,
    )


def test_api_key_guards_v1_routes(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, api_key="secret"))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/snapshot").status_code == 401
        assert client.get("/v1/track-record").status_code == 401
        assert client.get("/v1/track-record/backtest").status_code == 401
        assert client.get("/v1/solana/search?q=BONK").status_code == 401
        assert client.get("/v1/solana/tokens/new").status_code == 401
        assert (
            client.get(
                "/v1/solana/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263/candles?interval=1h"
            ).status_code
            == 401
        )
        assert (
            client.get("/v1/solana/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263").status_code
            == 401
        )
        assert client.post("/v1/backtest", json=_backtest_body()).status_code == 401
        batch_body = _backtest_body()
        batch_body["also"] = [{"venue": "coinbase", "symbol": "ETH-USD"}]
        assert client.post("/v1/backtest/batch", json=batch_body).status_code == 401
        wrong = client.get("/v1/snapshot", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = client.get("/v1/snapshot", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        track_record = client.get(
            "/v1/track-record",
            headers={"Authorization": "Bearer secret"},
        )
        assert track_record.status_code == 200
        assert track_record.json()["window_hours"] == 24
        assert (
            client.get(
                "/v1/track-record/backtest",
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        solana_feed = client.get(
            "/v1/solana/tokens/new",
            headers={"Authorization": "Bearer secret"},
        )
        assert solana_feed.status_code == 200
        assert solana_feed.json()["status"] == "warming"
        assert (
            client.get(
                "/v1/solana/tokens/not-a-mint",
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 400
        )


def test_solana_new_tokens_is_warming_before_first_cycle(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v1/solana/tokens/new")

    assert response.status_code == 200
    body = response.json()
    assert app.state.runtime.solana_tokens_task is None
    assert body["status"] == "warming"
    assert body["updated_at"] is None
    assert body["tokens"] == []
    assert [source["name"] for source in body["sources"]] == [
        "solana_rpc",
        "dexscreener",
        "geckoterminal",
        "raydium",
        "rugcheck",
    ]
    assert body["note"] == service_module.SOLANA_TOKENS_NOTE


def test_solana_detail_returns_404_for_non_token_mint(tmp_path: Path) -> None:
    class NonMintEngine:
        async def get_card(self, mint: str):
            raise NotTokenMint("Address is not a token mint.")

    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        app.state.runtime.solana_tokens = NonMintEngine()
        response = client.get("/v1/solana/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")

    assert response.status_code == 404
    assert response.json()["detail"] == "Address is not a token mint."


@pytest.mark.asyncio
async def test_solana_discovery_cycle_bounds_feed_and_orders_newest_first() -> None:
    mints = [str(Pubkey.new_unique()) for _ in range(205)]
    included = [
        {
            "id": f"solana_{mint}",
            "type": "token",
            "attributes": {
                "name": f"Token {index}",
                "symbol": ("SOL", "USDC", "USDT")[index] if index < 3 else f"T{index}",
            },
        }
        for index, mint in enumerate(mints)
    ]
    pools = [
        {
            "relationships": {
                "base_token": {"data": {"id": f"solana_{mint}"}},
            },
        }
        for mint in mints
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.geckoterminal.com":
            return httpx.Response(
                200,
                json={"data": pools, "included": included},
            )
        if request.url.host == "api.dexscreener.com":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = SolanaTokenClient(
            _solana_settings(),
            client,
            rate_limits={
                "solana_rpc": 0.0,
                "solana_largest": 0.0,
                "dexscreener": 0.0,
                "geckoterminal": 0.0,
                "raydium": 0.0,
                "rugcheck": 0.0,
            },
        )
        engine = SolanaTokenEngine(checker)
        await service_module._solana_discovery_cycle(engine)

    response = engine.response(200)
    assert engine.ready is True
    assert len(response["tokens"]) == 200
    assert response["tokens"][0]["mint"] == mints[-1]
    assert response["tokens"][-1]["mint"] == mints[5]
    assert all(token["symbol"] not in {"SOL", "USDC", "USDT"} for token in response["tokens"])
    assert len([request for request in requests if request.url.host == "api.dexscreener.com"]) == 7


def _solana_settings() -> Settings:
    return Settings(
        database_path=Path("unused-solana.db"),
        enabled_venues=(),
        shadow_mode=True,
        symbols=(),
        solana_rpc_url="https://solana.example.invalid",
        solana_tokens_enabled=False,
    )


def test_track_record_window_hours_validation(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/v1/track-record?window_hours=0").status_code == 422
        assert client.get("/v1/track-record?window_hours=169").status_code == 422


def test_snapshot_reports_latest_event_and_classification(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        pipeline = app.state.runtime.pipeline
        client.portal.call(pipeline.ingest, stablecoin_trade("coinbase", 0.999))
        client.portal.call(pipeline.ingest, stablecoin_trade("coinbase", 1.0, seconds=2))
        body = client.get("/v1/snapshot").json()
        assert body["shadow_mode"] is True
        assert [event["price"] for event in body["latest_events"]] == [1.0]
        [classification] = body["latest_classifications"]
        assert classification["classifier"] == "stablecoin_peg"
        assert classification["shadow"] is True


def _strategy_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "strategies.db",
        enabled_venues=("coinbase",),
        shadow_mode=True,
        symbols=("BTC-USD",),
        solana_tokens_enabled=False,
    )


async def _no_background_work(*args: object, **kwargs: object) -> None:
    return None


def _strategy_app(tmp_path: Path, monkeypatch) -> object:
    app = create_app(_strategy_settings(tmp_path))
    monkeypatch.setattr(service_module, "_seed_minute_bars", _no_background_work)
    monkeypatch.setattr(service_module, "_signal_backtest_loop", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "start", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "stop", _no_background_work)
    return app


def _batch_strategy_app(tmp_path: Path, monkeypatch) -> object:
    app = create_app(
        Settings(
            database_path=tmp_path / "batch-strategies.db",
            enabled_venues=("coinbase",),
            shadow_mode=True,
            symbols=("BTC-USD", "ETH-USD"),
            solana_tokens_enabled=False,
        )
    )
    monkeypatch.setattr(service_module, "_seed_minute_bars", _no_background_work)
    monkeypatch.setattr(service_module, "_signal_backtest_loop", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "start", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "stop", _no_background_work)
    return app


def _signal_backtest_app(
    tmp_path: Path,
    monkeypatch,
    *,
    symbols: tuple[str, ...],
    enabled_venues: tuple[str, ...] = ("coinbase",),
) -> object:
    app = create_app(
        Settings(
            database_path=tmp_path / "signal-backtest.db",
            enabled_venues=enabled_venues,
            shadow_mode=True,
            symbols=symbols,
            solana_tokens_enabled=False,
        )
    )
    monkeypatch.setattr(service_module, "_seed_minute_bars", _no_background_work)
    monkeypatch.setattr(service_module, "_signal_backtest_loop", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "start", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "stop", _no_background_work)
    return app


def _signal_candles() -> list[Candle]:
    current_minute = (int(datetime.now(UTC).timestamp() * 1000) // 60_000) * 60_000
    start_ms = current_minute - 360 * 60_000
    rising = [100 * 1.001**index for index in range(120)]
    closes = rising + [rising[-1]] * 240
    return [
        Candle(
            open_time=start_ms + index * 60_000,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1,
        )
        for index, close in enumerate(closes)
    ]


def _backtest_body(*, symbol: str = "BTC-USD", bars: int = 100) -> dict[str, object]:
    return {
        "rule": {
            "name": "test rule",
            "venue": "coinbase",
            "symbol": symbol,
            "interval": "1m",
            "conditions": [
                {
                    "left": {"kind": "price"},
                    "op": "above",
                    "right": {"kind": "value", "value": 0},
                }
            ],
            "expect": "up",
            "exit": {"after_bars": 1},
        },
        "bars": bars,
        "fee_bps": 10,
    }


def test_candles_route_computes_requested_indicators_only_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    source_candles = [
        Candle(
            open_time=index * 60_000,
            open=float(index + 1),
            high=float(index + 1),
            low=float(index + 1),
            close=float(index + 1),
            volume=1,
        )
        for index in range(3)
    ]

    async def fetcher(*args: object) -> list[Candle]:
        return source_candles

    monkeypatch.setattr(candle_module, "fetch_source_candles", fetcher)
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get(
            "/v1/candles",
            params=[
                ("venue", "coinbase"),
                ("symbol", "BTC-USD"),
                ("interval", "1m"),
                ("indicator", "sma:2"),
                ("indicator", "rsi:2"),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["indicators"]["sma:2"]["value"] == [None, 1.5, 2.5]
        assert body["indicators"]["rsi:2"]["value"] == [None, None, 100.0]
        assert len(body["indicators"]["sma:2"]["value"]) == len(body["candles"])

        no_indicators = client.get("/v1/candles?venue=coinbase&symbol=BTC-USD&interval=1m")
        assert no_indicators.status_code == 200
        assert "indicators" not in no_indicators.json()


def test_candles_route_rejects_invalid_or_excessive_indicator_specs(
    tmp_path: Path, monkeypatch
) -> None:
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        base = "/v1/candles?venue=coinbase&symbol=BTC-USD&interval=1m"
        invalid = client.get(f"{base}&indicator=sma%3A1")
        assert invalid.status_code == 422
        assert "between 2 and 200" in invalid.json()["detail"]

        params = [("venue", "coinbase"), ("symbol", "BTC-USD"), ("interval", "1m")]
        params.extend(("indicator", "sma:2") for _ in range(7))
        excessive = client.get("/v1/candles", params=params)
        assert excessive.status_code == 422


def test_solana_search_groups_sorts_and_caches_casefolded_queries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bonk_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    other_mint = str(Pubkey.new_unique())
    pairs = [
        {
            "chainId": "solana",
            "baseToken": {"address": bonk_mint, "symbol": "BONK", "name": "Bonk"},
            "pairAddress": "bonk-low",
            "priceUsd": "0.00001",
            "liquidity": {"usd": 100},
            "volume": {"h24": 11},
            "info": {"imageUrl": "https://example.invalid/bonk-low.png"},
        },
        {
            "chainId": "solana",
            "baseToken": {"address": bonk_mint, "symbol": "BONK", "name": "Bonk"},
            "pairAddress": "bonk-high",
            "priceUsd": "0.00003",
            "liquidity": {"usd": 300},
            "volume": {"h24": 22},
            "info": {"imageUrl": "https://example.invalid/bonk.png"},
        },
        {
            "chainId": "solana",
            "baseToken": {"address": other_mint, "symbol": "OTHER", "name": "Other"},
            "pairAddress": "other-pool",
            "priceUsd": "2.5",
            "liquidity": {"usd": 500},
            "volume": {"h24": 55},
        },
        {
            "chainId": "ethereum",
            "baseToken": {"address": str(Pubkey.new_unique()), "symbol": "BONK"},
            "pairAddress": "not-solana",
            "liquidity": {"usd": 1_000_000},
            "volume": {"h24": 1_000_000},
        },
    ]
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    async def get_json(source, url, *, params=None, headers=None, priority="background"):
        calls.append((source, url, params))
        return {"pairs": pairs}

    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        checker = app.state.runtime.solana_tokens.checker
        monkeypatch.setattr(checker, "get_json", get_json)
        first = client.get("/v1/solana/search", params={"q": " BONK "})
        second = client.get("/v1/solana/search", params={"q": "bonk"})
        invalid = client.get("/v1/solana/search", params={"q": "  "})
        overlong = client.get("/v1/solana/search", params={"q": "x" * 33})

    assert first.status_code == 200
    body = first.json()
    assert [token["mint"] for token in body["tokens"]] == [other_mint, bonk_mint]
    bonk = body["tokens"][1]
    assert bonk == {
        "mint": bonk_mint,
        "symbol": "BONK",
        "name": "Bonk",
        "image": "https://example.invalid/bonk.png",
        "pool_count": 2,
        "total_liquidity_usd": 400.0,
        "volume_24h_usd": 33.0,
        "main_pool_address": "bonk-high",
        "price_usd": 3e-05,
        "symbol_match": True,
    }
    assert body["source"] == "dexscreener"
    assert "not an endorsement" in body["note"]
    assert second.status_code == 200
    assert second.json()["query"] == "bonk"
    assert len(calls) == 1
    assert calls[0][0] == "dexscreener"
    assert calls[0][1] == "https://api.dexscreener.com/latest/dex/search"
    assert calls[0][2] == {"q": "BONK"}
    assert invalid.status_code == 422
    assert overlong.status_code == 422


def test_solana_search_upstream_error_returns_502(tmp_path: Path, monkeypatch) -> None:
    async def get_json(source, url, *, params=None, headers=None, priority="background"):
        raise SourceError("dexscreener", "Couldn't check right now")

    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        checker = app.state.runtime.solana_tokens.checker
        monkeypatch.setattr(checker, "get_json", get_json)
        response = client.get("/v1/solana/search?q=BONK")

    assert response.status_code == 502
    assert response.json()["detail"] == "Couldn't check right now"


def test_solana_token_candles_map_intervals_cache_and_add_indicators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    pairs = [
        {
            "baseToken": {"address": mint, "symbol": "BONK"},
            "pairAddress": "pool-low",
            "dexId": "raydium",
            "liquidity": {"usd": 100},
        },
        {
            "baseToken": {"address": mint, "symbol": "BONK"},
            "pairAddress": "pool-main",
            "dexId": "orca",
            "liquidity": {"usd": 500},
        },
    ]
    candle_body = {
        "data": {
            "attributes": {
                "ohlcv_list": [
                    [1_730_000_060, 2.0, 2.5, 1.5, 2.0, 20.0],
                    [1_730_000_000, 1.0, 2.0, 0.5, 1.5, 10.0],
                ]
            }
        }
    }
    calls: list[tuple[str, str, dict[str, object] | None, str]] = []

    async def get_json(source, url, *, params=None, headers=None, priority="background"):
        calls.append((source, url, params, priority))
        if source == "dexscreener":
            return pairs
        return candle_body

    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        checker = app.state.runtime.solana_tokens.checker
        monkeypatch.setattr(checker, "get_json", get_json)
        first = client.get(
            f"/v1/solana/tokens/{mint}/candles",
            params=[("interval", "1h"), ("limit", "2"), ("indicator", "sma:2")],
        )
        cached = client.get(
            f"/v1/solana/tokens/{mint}/candles",
            params=[("interval", "1h"), ("limit", "2")],
        )
        different_indicator = client.get(
            f"/v1/solana/tokens/{mint}/candles",
            params=[("interval", "1h"), ("limit", "2"), ("indicator", "sma:3")],
        )
        intervals = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "1h": ("hour", 1),
            "4h": ("hour", 4),
            "1d": ("day", 1),
        }
        interval_responses = {
            interval: client.get(
                f"/v1/solana/tokens/{mint}/candles",
                params=[("interval", interval), ("limit", "2")],
            )
            for interval in intervals
        }

    assert first.status_code == 200
    body = first.json()
    assert body["venue"] == "geckoterminal"
    assert body["symbol"] == "BONK"
    assert body["interval"] == "1h"
    assert body["source_url_template"] == "https://www.geckoterminal.com/solana/pools/{pool}"
    assert body["aggregated_from"] is None
    assert body["stale"] is False
    assert body["truncated"] is False
    assert body["mint"] == mint
    assert body["pool_address"] == "pool-main"
    assert body["dex_id"] == "orca"
    assert body["price_currency"] == "usd"
    assert set(body["indicators"]) == {"sma:2"}
    assert [candle["open_time"] for candle in body["candles"]] == [
        1_730_000_000_000,
        1_730_000_060_000,
    ]
    assert [candle["volume"] for candle in body["candles"]] == [10, 20]
    assert body["indicators"]["sma:2"]["value"] == [None, 1.75]
    assert cached.status_code == 200
    assert "indicators" not in cached.json()
    assert set(different_indicator.json()["indicators"]) == {"sma:3"}
    assert len(calls) == 7
    assert calls[0][1] == ("https://api.dexscreener.com/token-pairs/v1/solana/" + mint)
    gecko_calls = [call for call in calls if call[0] == "geckoterminal"]
    assert {(call[1].rsplit("/", 1)[-1], call[2]["aggregate"]) for call in gecko_calls} == set(
        intervals.values()
    )
    hourly = next(call for call in gecko_calls if call[1].endswith("/ohlcv/hour"))
    assert all(call[3] == "interactive" for call in gecko_calls)
    assert hourly[2] == {
        "aggregate": 1,
        "limit": 2,
        "currency": "usd",
        "token": mint,
    }
    assert all(response.status_code == 200 for response in interval_responses.values())


def test_solana_token_candles_no_pool_and_validation_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        checker = app.state.runtime.solana_tokens.checker

        async def empty_pairs(source, url, *, params=None, headers=None, priority="background"):
            return []

        monkeypatch.setattr(checker, "get_json", empty_pairs)
        no_pool = client.get(
            "/v1/solana/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263/candles",
            params={"interval": "1h"},
        )
        invalid_mint = client.get("/v1/solana/tokens/not-a-mint/candles?interval=1h")
        invalid_interval = client.get(
            "/v1/solana/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263/candles?interval=2m"
        )

    assert no_pool.status_code == 404
    assert no_pool.json()["detail"] == "No trading pool found for this token"
    assert invalid_mint.status_code == 400
    assert invalid_interval.status_code == 422


def test_solana_token_candles_stale_cache_and_unavailable_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    pairs = [
        {
            "baseToken": {"address": mint, "symbol": "BONK"},
            "pairAddress": "pool-main",
            "dexId": "orca",
            "liquidity": {"usd": 500},
        }
    ]
    body = {
        "data": {
            "attributes": {
                "ohlcv_list": [[1_730_000_000, 1, 2, 0.5, 1.5, 10]],
            }
        }
    }
    fail_gecko = False

    async def get_json(source, url, *, params=None, headers=None, priority="background"):
        if source == "dexscreener":
            return pairs
        if fail_gecko:
            raise SourceError("geckoterminal", "Couldn't check right now (rate limited)")
        return body

    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        checker = app.state.runtime.solana_tokens.checker
        monkeypatch.setattr(checker, "get_json", get_json)
        path = f"/v1/solana/tokens/{mint}/candles?interval=1h"
        fresh = client.get(path)
        cache_key = ("pool-main", "1h", 300)
        cache_entry = app.state.runtime.solana_candle_cache[cache_key]
        app.state.runtime.solana_candle_cache[cache_key] = (
            service_module.monotonic() - service_module.SOLANA_CANDLE_CACHE_TTL_SECONDS - 1,
            cache_entry[1],
        )
        fail_gecko = True
        stale = client.get(path)

    assert fresh.status_code == 200
    assert fresh.json()["truncated"] is True
    assert stale.status_code == 200
    assert stale.json()["stale"] is True

    other_app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(other_app) as client:
        checker = other_app.state.runtime.solana_tokens.checker

        async def unavailable(source, url, *, params=None, headers=None, priority="background"):
            if source == "dexscreener":
                return pairs
            raise SourceError("geckoterminal", "Couldn't check right now")

        monkeypatch.setattr(checker, "get_json", unavailable)
        failed = client.get(path)

    assert failed.status_code == 502
    assert failed.json()["detail"] == (
        "GeckoTerminal is rate-limited or unavailable; try again shortly"
    )


def test_backtest_endpoint_requires_valid_bounds_and_known_symbols(
    tmp_path: Path, monkeypatch
) -> None:
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        too_few_bars = client.post("/v1/backtest", json=_backtest_body(bars=99))
        assert too_few_bars.status_code == 422
        too_many_bars = client.post("/v1/backtest", json=_backtest_body(bars=5001))
        assert too_many_bars.status_code == 422
        too_much_fee = _backtest_body()
        too_much_fee["fee_bps"] = 101
        assert client.post("/v1/backtest", json=too_much_fee).status_code == 422
        too_much_slippage = _backtest_body()
        too_much_slippage["slippage_bps"] = 101
        assert client.post("/v1/backtest", json=too_much_slippage).status_code == 422

        unknown_symbol = client.post(
            "/v1/backtest",
            json=_backtest_body(symbol="ETH-USD"),
        )
        assert unknown_symbol.status_code == 404
        assert unknown_symbol.json()["detail"] == "Unknown symbol."


def test_backtest_endpoint_fetches_closed_candles_and_reports_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    current_minute = (now_ms // 60_000) * 60_000
    source_candles = [
        Candle(
            open_time=current_minute - offset * 60_000,
            open=100 + offset,
            high=102 + offset,
            low=99 + offset,
            close=101 + offset,
            volume=1,
        )
        for offset in (4, 3, 2, 1, -10)
    ]
    calls = 0

    async def fetcher(*args: object) -> list[Candle]:
        nonlocal calls
        calls += 1
        return source_candles

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        backtest_body = _backtest_body()
        backtest_body["slippage_bps"] = 5
        backtest_body["holdout_pct"] = 25
        response = client.post("/v1/backtest", json=backtest_body)
        assert response.status_code == 200
        body = response.json()
        assert body["bars_requested"] == 100
        assert body["bars_tested"] == 4
        assert body["truncated"] is True
        assert body["first_bar_time"] == current_minute - 4 * 60_000
        assert body["last_bar_time"] == current_minute - 60_000
        assert body["stats"]["trades"] == 2
        assert body["costs"] == {
            "fee_bps": 10,
            "slippage_bps": 5,
            "round_trip_pct": 0.3,
        }
        assert set(body) >= {"equity", "buy_and_hold", "by_year", "holdout"}
        assert body["holdout"]["holdout_pct"] == 25

        second = client.post("/v1/backtest", json=backtest_body)
        assert second.status_code == 200
        assert calls == 1


def test_backtest_cache_prunes_expired_entries_and_evicts_oldest(
    tmp_path: Path, monkeypatch
) -> None:
    current_time = 1000.0
    monkeypatch.setattr(service_module, "monotonic", lambda: current_time)

    async def fetcher(*args: object) -> list[Candle]:
        return []

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        runtime = app.state.runtime
        expired_key = ("coinbase", "EXPIRED-USD", "1m", 100)
        runtime.backtest_candles[expired_key] = (
            current_time - service_module.BACKTEST_CACHE_TTL_SECONDS - 1,
            [],
        )
        fresh_keys = [("coinbase", f"TEST-{index}-USD", "1m", 100) for index in range(33)]
        for index, cache_key in enumerate(fresh_keys):
            runtime.backtest_candles[cache_key] = (current_time - 50 + index, [])

        response = client.post("/v1/backtest", json=_backtest_body())

        assert response.status_code == 200
        assert expired_key not in runtime.backtest_candles
        assert fresh_keys[0] not in runtime.backtest_candles
        assert fresh_keys[1] not in runtime.backtest_candles
        assert fresh_keys[2] in runtime.backtest_candles
        assert ("coinbase", "BTC-USD", "1m", 100) in runtime.backtest_candles
        assert len(runtime.backtest_candles) == service_module.BACKTEST_CACHE_MAX_ENTRIES


def test_batch_backtest_keeps_request_order_reports_unknown_symbols_and_reuses_cache(
    tmp_path: Path, monkeypatch
) -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    current_minute = (now_ms // 60_000) * 60_000
    source_candles = [
        Candle(
            open_time=current_minute - offset * 60_000,
            open=100 + offset,
            high=102 + offset,
            low=99 + offset,
            close=101 + offset,
            volume=1,
        )
        for offset in (4, 3, 2, 1, -10)
    ]
    fetched_symbols: list[str] = []

    async def fetcher(*args: object) -> list[Candle]:
        fetched_symbols.append(str(args[2]))
        return source_candles

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _batch_strategy_app(tmp_path, monkeypatch)
    body = _backtest_body()
    body["also"] = [
        {"venue": "COINBASE", "symbol": "eth-usd"},
        {"venue": "coinbase", "symbol": "LTC-USD"},
    ]

    with TestClient(app) as client:
        response = client.post("/v1/backtest/batch", json=body)

        assert response.status_code == 200
        data = response.json()
        assert [(item["venue"], item["symbol"]) for item in data["results"]] == [
            ("coinbase", "BTC-USD"),
            ("coinbase", "ETH-USD"),
            ("coinbase", "LTC-USD"),
        ]
        assert data["results"][0]["error"] is None
        assert data["results"][0]["stats"]["trades"] == 2
        assert data["results"][2]["error"] == "Unknown symbol."
        assert data["results"][2]["stats"] is None
        assert data["results"][2]["bars_tested"] is None
        assert data["summary"]["instruments_tested"] == 2
        assert data["summary"]["instruments_failed"] == 1
        assert data["note"].startswith("Coins tend to move together")
        assert "curve" not in data["results"][0]
        assert "trades" not in data["results"][0]
        assert sorted(fetched_symbols) == ["BTC-USD", "ETH-USD"]

        repeated = client.post("/v1/backtest/batch", json=body)
        assert repeated.status_code == 200
        assert sorted(fetched_symbols) == ["BTC-USD", "ETH-USD"]


def test_batch_backtest_rejects_normalized_duplicates_and_too_many_instruments(
    tmp_path: Path, monkeypatch
) -> None:
    app = _batch_strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        duplicate = _backtest_body()
        duplicate["also"] = [{"venue": "COINBASE", "symbol": "btc-usd"}]
        response = client.post("/v1/backtest/batch", json=duplicate)
        assert response.status_code == 422
        assert response.json()["detail"] == "Duplicate instrument."

        excessive = _backtest_body()
        excessive["also"] = [
            {"venue": "coinbase", "symbol": f"SYMBOL-{index}-USD"} for index in range(20)
        ]
        assert client.post("/v1/backtest/batch", json=excessive).status_code == 422


def test_batch_backtest_keeps_candle_fetch_failures_per_instrument(
    tmp_path: Path, monkeypatch
) -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    current_minute = (now_ms // 60_000) * 60_000
    source_candles = [
        Candle(
            open_time=current_minute - offset * 60_000,
            open=100 + offset,
            high=102 + offset,
            low=99 + offset,
            close=101 + offset,
            volume=1,
        )
        for offset in (4, 3, 2, 1)
    ]

    async def fetcher(
        _client: object, _venue: str, symbol: str, _interval: str, _limit: int
    ) -> list[Candle]:
        if symbol == "ETH-USD":
            raise RuntimeError("upstream unavailable")
        return source_candles

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _batch_strategy_app(tmp_path, monkeypatch)
    body = _backtest_body()
    body["also"] = [{"venue": "coinbase", "symbol": "ETH-USD"}]

    with TestClient(app) as client:
        response = client.post("/v1/backtest/batch", json=body)

    assert response.status_code == 200
    assert response.json()["results"][1]["error"] == "upstream unavailable"
    assert response.json()["results"][1]["stats"] is None
    assert response.json()["summary"]["instruments_failed"] == 1


def test_track_record_backtest_returns_computing_before_first_result(
    tmp_path: Path, monkeypatch
) -> None:
    app = _signal_backtest_app(tmp_path, monkeypatch, symbols=("BTC-USD",))

    with TestClient(app) as client:
        response = client.get("/v1/track-record/backtest")

    assert response.status_code == 200
    assert response.json() == {
        "status": "computing",
        "computed_at": None,
        "span_start": None,
        "span_end": None,
        "sources": [],
        "groups": [],
        "note": service_module.BACKTEST_NOTE,
    }


def test_track_record_backtest_pools_ready_results_and_sources(tmp_path: Path, monkeypatch) -> None:
    candles = _signal_candles()
    calls: list[tuple[str, str, int]] = []

    async def fetcher(
        _client: object,
        venue: str,
        symbol: str,
        _interval: str,
        limit: int,
    ) -> list[Candle]:
        calls.append((venue, symbol, limit))
        return candles

    app = _signal_backtest_app(
        tmp_path,
        monkeypatch,
        symbols=("BTC-USD", "ETH-USD", "USDC-USD"),
    )
    with TestClient(app) as client:
        runtime = app.state.runtime

        async def compute() -> dict[str, object]:
            return await service_module._compute_signal_backtest(runtime, fetcher=fetcher)

        runtime.signal_backtest = client.portal.call(compute)
        response = client.get("/v1/track-record/backtest")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["computed_at"] is not None
    assert (
        body["span_start"] == datetime.fromtimestamp(candles[0].open_time / 1000, UTC).isoformat()
    )
    assert (
        body["span_end"]
        == datetime.fromtimestamp(
            (candles[-1].open_time + 60_000) / 1000,
            UTC,
        ).isoformat()
    )
    assert body["sources"] == [
        {"symbol": "BTC-USD", "venue": "coinbase", "bars": 360},
        {"symbol": "ETH-USD", "venue": "coinbase", "bars": 360},
    ]
    assert calls == [
        ("coinbase", "BTC-USD", service_module.BACKTEST_BAR_COUNT),
        ("coinbase", "ETH-USD", service_module.BACKTEST_BAR_COUNT),
    ]
    groups = {group["horizon_seconds"]: group for group in body["groups"]}
    assert set(groups) >= {60, 300}
    assert all(group["classifier"] == "momentum_regime" for group in body["groups"])
    assert groups[60]["scored"] == 714
    assert groups[60]["hits"] == 712
    assert groups[60]["pending"] == 4
    assert body["note"] == service_module.BACKTEST_NOTE


def test_track_record_backtest_falls_through_to_next_eligible_venue(
    tmp_path: Path, monkeypatch
) -> None:
    candles = _signal_candles()
    calls: list[str] = []

    async def fetcher(
        _client: object,
        venue: str,
        _symbol: str,
        _interval: str,
        _limit: int,
    ) -> list[Candle]:
        calls.append(venue)
        if venue == "coinbase":
            raise RuntimeError("coinbase unavailable")
        return candles

    app = _signal_backtest_app(
        tmp_path,
        monkeypatch,
        symbols=("BTC-USD",),
        enabled_venues=("coinbase", "okx"),
    )
    with TestClient(app) as client:

        async def compute() -> dict[str, object]:
            return await service_module._compute_signal_backtest(
                app.state.runtime,
                fetcher=fetcher,
            )

        result = client.portal.call(compute)

    assert calls == ["coinbase", "okx"]
    assert result["status"] == "ready"
    assert result["sources"] == [{"symbol": "BTC-USD", "venue": "okx", "bars": 360}]


def test_backtest_endpoint_maps_upstream_failure_to_502(tmp_path: Path, monkeypatch) -> None:
    async def fetcher(*args: object) -> list[Candle]:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/v1/backtest", json=_backtest_body())

    assert response.status_code == 502
    assert response.json()["detail"] == "upstream unavailable"
