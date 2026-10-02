from pathlib import Path

from fastapi.testclient import TestClient

from baystfirm.config import Settings
from baystfirm.service import create_app


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
