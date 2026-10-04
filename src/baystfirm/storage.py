from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from baystfirm.models import Classification, MarketEvent


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.path)
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_events (
                event_id TEXT PRIMARY KEY,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                received_timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        await self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_events_symbol_time
            ON market_events(symbol, exchange_timestamp)
            """
        )
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                classification_id TEXT PRIMARY KEY,
                classifier TEXT NOT NULL,
                symbol TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                label TEXT NOT NULL,
                probability REAL NOT NULL,
                abstained INTEGER NOT NULL,
                shadow INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                classifier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("event store is not open")
        return self._connection

    async def append_event(self, event: MarketEvent) -> None:
        await self.append_events([event])

    async def append_events(self, events: Iterable[MarketEvent]) -> None:
        values = [
            (
                str(event.event_id),
                event.venue,
                event.symbol,
                event.event_type.value,
                event.exchange_timestamp.isoformat(),
                event.received_timestamp.isoformat(),
                event.model_dump_json(),
            )
            for event in events
        ]
        if not values:
            return
        await self.connection.executemany(
            """
            INSERT OR IGNORE INTO market_events (
                event_id, venue, symbol, event_type, exchange_timestamp,
                received_timestamp, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        await self.connection.commit()

    async def append_classification(self, classification: Classification) -> None:
        await self.connection.execute(
            """
            INSERT OR REPLACE INTO classifications (
                classification_id, classifier, symbol, observed_at, label,
                probability, abstained, shadow, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(classification.classification_id),
                classification.classifier,
                classification.symbol,
                classification.observed_at.isoformat(),
                classification.label,
                classification.probability,
                int(classification.abstained),
                int(classification.shadow),
                classification.model_dump_json(),
            ),
        )
        await self.connection.commit()

    async def iter_events(
        self,
        *,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[MarketEvent]:
        clauses: list[str] = []
        parameters: list[str | int] = []
        if symbol:
            clauses.append("symbol = ?")
            parameters.append(symbol.upper())
        if start:
            clauses.append("exchange_timestamp >= ?")
            parameters.append(start.isoformat())
        if end:
            clauses.append("exchange_timestamp <= ?")
            parameters.append(end.isoformat())
        query = "SELECT payload FROM market_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY exchange_timestamp ASC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        cursor = await self.connection.execute(query, parameters)
        async for row in cursor:
            yield MarketEvent.model_validate_json(row[0])
        await cursor.close()

    async def event_count(self) -> int:
        cursor = await self.connection.execute("SELECT COUNT(*) FROM market_events")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0

    async def iter_classifications(
        self,
        *,
        symbol: str | None = None,
        classifier: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[Classification]:
        clauses: list[str] = []
        parameters: list[str | int] = []
        if symbol:
            clauses.append("symbol = ?")
            parameters.append(symbol.upper())
        if classifier:
            clauses.append("classifier = ?")
            parameters.append(classifier)
        query = "SELECT payload FROM classifications"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY observed_at DESC LIMIT ?"
        parameters.append(limit)
        cursor = await self.connection.execute(query, parameters)
        async for row in cursor:
            yield Classification.model_validate_json(row[0])
        await cursor.close()

    async def classification_count(self) -> int:
        cursor = await self.connection.execute("SELECT COUNT(*) FROM classifications")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0

    async def append_evaluation_run(self, classifier: str, payload: dict[str, Any]) -> str:
        run_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        record = {"run_id": run_id, "classifier": classifier, "created_at": created_at, **payload}
        await self.connection.execute(
            "INSERT INTO evaluation_runs (run_id, classifier, created_at, payload) "
            "VALUES (?, ?, ?, ?)",
            (run_id, classifier, created_at, json.dumps(record)),
        )
        await self.connection.commit()
        return run_id

    async def latest_evaluation_runs(self) -> list[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT payload FROM evaluation_runs AS run
            WHERE created_at = (
                SELECT MAX(created_at) FROM evaluation_runs WHERE classifier = run.classifier
            )
            ORDER BY classifier
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [json.loads(row[0]) for row in rows]
