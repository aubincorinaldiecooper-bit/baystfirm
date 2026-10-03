from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import uvicorn

from baystfirm.classifiers import MarketStateClassifier
from baystfirm.evaluation import (
    EvaluationRecord,
    PromotionGate,
    evaluate,
    evaluate_by_classifier,
    label_classifications,
)
from baystfirm.models import Classification, MarketEvent
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
    classifier = MarketStateClassifier(shadow=True)

    async def classify(event: MarketEvent) -> None:
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
    classifier = MarketStateClassifier(shadow=True)
    predictions: list[Classification] = []
    first: MarketEvent | None = None
    last: MarketEvent | None = None

    async def classify(event: MarketEvent) -> None:
        nonlocal first, last
        first = first or event
        last = event
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


if __name__ == "__main__":
    main()
