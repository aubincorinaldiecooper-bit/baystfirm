from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import uvicorn

from baystfirm.archive import _iter_archive_records, fetch_archive, normalized_symbol
from baystfirm.evaluation import (
    EvaluationRecord,
    PromotionGate,
    evaluate,
    evaluate_by_classifier,
    label_classifications,
)
from baystfirm.models import (
    Classification,
    EventType,
    InstrumentKind,
    MarketEvent,
    payload_digest,
)
from baystfirm.pipeline import default_classifiers
from baystfirm.replay import ReplayRunner
from baystfirm.storage import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Baystfirm crypto intelligence tooling.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the API and live market streams.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")

    replay = subparsers.add_parser("replay", help="Replay stored events through classifiers.")
    replay.add_argument("--database", default="var/baystfirm.db", type=Path)
    replay.add_argument("--symbol")
    replay.add_argument("--speed", default=0.0, type=float)
    replay.add_argument("--limit", type=int)

    evaluation = subparsers.add_parser(
        "evaluate",
        help="Evaluate JSONL prediction records against the promotion gate.",
    )
    evaluation.add_argument("records", type=Path)

    replay_evaluation = subparsers.add_parser(
        "evaluate-replay",
        help="Replay stored events, label outcomes, evaluate, and record the run.",
    )
    replay_evaluation.add_argument("--database", default="var/baystfirm.db", type=Path)
    replay_evaluation.add_argument("--tolerance-seconds", default=5.0, type=float)

    archive = subparsers.add_parser("import-archive", help="Import Binance spot aggTrades.")
    archive.add_argument("--symbol", required=True)
    archive.add_argument("--date", action="append", required=True)
    archive.add_argument("--database", type=Path)
    archive.add_argument("--ticks-dir", type=Path)

    args = parser.parse_args()
    if args.command is None:
        args.host = "127.0.0.1"
        args.port = 8000
        args.reload = False
        _serve(args)
    elif args.command == "serve":
        _serve(args)
    elif args.command == "replay":
        asyncio.run(_replay(args))
    elif args.command == "evaluate":
        _evaluate(args.records)
    elif args.command == "evaluate-replay":
        print(json.dumps(asyncio.run(_evaluate_replay(args)), indent=2))
    elif args.command == "import-archive":
        if args.database is None and args.ticks_dir is None:
            parser.error("import-archive requires at least one of --database or --ticks-dir")
        asyncio.run(_import_archive(args))


def _serve(args: argparse.Namespace) -> None:
    uvicorn.run(
        "baystfirm.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


async def _replay(args: argparse.Namespace) -> None:
    store = EventStore(args.database)
    await store.open()
    classifiers = default_classifiers(shadow=True)

    async def classify(event: MarketEvent) -> None:
        for classifier in classifiers:
            for result in classifier.observe(event):
                print(result.model_dump_json())

    try:
        count = await ReplayRunner(store).run(
            classify,
            symbol=args.symbol,
            speed=args.speed,
            limit=args.limit,
        )
    finally:
        await store.close()
    print(json.dumps({"replayed_events": count}))


async def _evaluate_replay(args: argparse.Namespace) -> dict[str, object]:
    store = EventStore(args.database)
    await store.open()
    classifiers = default_classifiers(shadow=True)
    predictions: list[Classification] = []
    first: MarketEvent | None = None
    last: MarketEvent | None = None

    async def classify(event: MarketEvent) -> None:
        nonlocal first, last
        first = first or event
        last = event
        for classifier in classifiers:
            predictions.extend(classifier.observe(event))

    gate = PromotionGate()
    report: dict[str, object] = {}
    try:
        event_count = await ReplayRunner(store).run(classify)
        records = label_classifications(predictions, tolerance_seconds=args.tolerance_seconds)
        dataset = {
            "events": event_count,
            "predictions": len(predictions),
            "labeled_records": len(records),
            "start": first.exchange_timestamp.isoformat() if first else None,
            "end": last.exchange_timestamp.isoformat() if last else None,
            "labeling": "same-classifier observed state one horizon later",
        }
        report["dataset"] = dataset
        for name, metrics in evaluate_by_classifier(records).items():
            failures = gate.failures(metrics)
            run = {
                "metrics": asdict(metrics),
                "promotion_eligible": not failures,
                "failures": failures,
                "dataset": dataset,
            }
            run["run_id"] = await store.append_evaluation_run(name, run)
            report[name] = run
    finally:
        await store.close()
    return report


def _evaluate(path: Path) -> None:
    records = [
        EvaluationRecord(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    metrics = evaluate(records)
    failures = PromotionGate().failures(metrics)
    print(
        json.dumps(
            {
                "metrics": asdict(metrics),
                "promotion_eligible": not failures,
                "failures": failures,
            },
            indent=2,
        )
    )


async def _import_archive(args: argparse.Namespace) -> None:
    native_symbol = args.symbol.upper()
    symbol, base_asset, quote_asset = normalized_symbol(native_symbol)
    if args.ticks_dir is not None:
        args.ticks_dir.mkdir(parents=True, exist_ok=True)
    store = EventStore(args.database) if args.database is not None else None
    if store is not None:
        await store.open()
    imported = 0
    try:
        for date in args.date:
            archive = fetch_archive(native_symbol, date)
            csv_file: TextIO | None = None
            writer: Any = None
            if args.ticks_dir is not None:
                tick_path = args.ticks_dir / f"{native_symbol}-{date}.csv"
                csv_file = tick_path.open("w", newline="", encoding="utf-8")
                writer = csv.writer(csv_file)
                writer.writerow(("timestamp_ms", "price", "size"))
            batch: list[MarketEvent] = []
            previous_timestamp: int | None = None
            try:
                for (timestamp_ms, price, size, side, agg_id), raw_line in _iter_archive_records(
                    archive
                ):
                    if previous_timestamp is not None and timestamp_ms < previous_timestamp:
                        raise ValueError(f"archive rows are not time-sorted: {archive.name}")
                    previous_timestamp = timestamp_ms
                    if writer is not None:
                        writer.writerow((timestamp_ms, price, size))
                    if store is not None:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
                        batch.append(
                            MarketEvent(
                                venue="binance",
                                symbol=symbol,
                                native_symbol=native_symbol,
                                base_asset=base_asset,
                                quote_asset=quote_asset,
                                instrument_kind=InstrumentKind.SPOT,
                                event_type=EventType.TRADE,
                                exchange_timestamp=timestamp,
                                received_timestamp=timestamp,
                                sequence=agg_id,
                                price=price,
                                size=size,
                                side=side,
                                payload_hash=payload_digest(raw_line),
                                metadata={
                                    "source": "binance_archive",
                                    "archive_file": archive.name,
                                },
                            )
                        )
                        if len(batch) >= 1000:
                            await store.append_events(batch)
                            batch.clear()
                    imported += 1
                if store is not None and batch:
                    await store.append_events(batch)
            finally:
                if csv_file is not None:
                    csv_file.close()
    finally:
        if store is not None:
            await store.close()
    print(json.dumps({"symbol": native_symbol, "rows_imported": imported}))


if __name__ == "__main__":
    main()
