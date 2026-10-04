from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

import baystfirm.candles as candle_module
import baystfirm.service as service_module
from baystfirm.config import Settings
from baystfirm.models import Candle
from baystfirm.service import create_app
from tests.test_classifiers import stablecoin_trade


def test_health_and_evaluation_gate(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            database_path=tmp_path / "service.db",
            enabled_venues=(),
            shadow_mode=True,
            symbols=(),
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
    )


def test_api_key_guards_v1_routes(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, api_key="secret"))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/snapshot").status_code == 401
        assert client.get("/v1/track-record").status_code == 401
        assert client.post("/v1/backtest", json=_backtest_body()).status_code == 401
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
    )


async def _no_background_work(*args: object, **kwargs: object) -> None:
    return None


def _strategy_app(tmp_path: Path, monkeypatch) -> object:
    app = create_app(_strategy_settings(tmp_path))
    monkeypatch.setattr(service_module, "_seed_minute_bars", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "start", _no_background_work)
    monkeypatch.setattr(app.state.runtime.ingestion, "stop", _no_background_work)
    return app


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
        response = client.post("/v1/backtest", json=_backtest_body())
        assert response.status_code == 200
        body = response.json()
        assert body["bars_requested"] == 100
        assert body["bars_tested"] == 4
        assert body["truncated"] is True
        assert body["first_bar_time"] == current_minute - 4 * 60_000
        assert body["last_bar_time"] == current_minute - 60_000
        assert body["stats"]["trades"] == 2

        second = client.post("/v1/backtest", json=_backtest_body())
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


def test_backtest_endpoint_maps_upstream_failure_to_502(tmp_path: Path, monkeypatch) -> None:
    async def fetcher(*args: object) -> list[Candle]:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(service_module, "fetch_source_candles", fetcher)
    app = _strategy_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/v1/backtest", json=_backtest_body())

    assert response.status_code == 502
    assert response.json()["detail"] == "upstream unavailable"
