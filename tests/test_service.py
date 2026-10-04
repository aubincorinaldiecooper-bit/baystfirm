from pathlib import Path

from fastapi.testclient import TestClient

from baystfirm.config import Settings
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
        wrong = client.get("/v1/snapshot", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = client.get("/v1/snapshot", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200


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
