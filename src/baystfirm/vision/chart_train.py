from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .chart import (
    CHART_LABELS,
    CLASSIFIER,
    NORMAL_LABEL,
    ChartSample,
    Tick,
    build_samples,
    render_chart,
    sample_dict,
)

BACKBONE_REPO = "openbmb/MiniCPM-V-4.6"
SEED = 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and train the shadow chart-state head."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build", help="Render labeled chart samples from trade ticks."
    )
    build.add_argument("--ticks-dir", type=Path, required=True)
    build.add_argument("--days", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)

    extract = commands.add_parser(
        "extract", help="Extract frozen visual backbone embeddings."
    )
    extract.add_argument("--dataset", type=Path, required=True)
    extract.add_argument("--model-dir", required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--limit-per-split", type=int)

    train = commands.add_parser(
        "train", help="Train and calibrate the chart-state pointer head."
    )
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)

    predict = commands.add_parser(
        "predict", help="Write test predictions and persistence baseline."
    )
    predict.add_argument("--dataset", type=Path, required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--head", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        _build(args.ticks_dir, args.days, args.out)
    elif args.command == "extract":
        _extract(
            args.dataset, args.model_dir, args.out, args.limit, args.limit_per_split
        )
    elif args.command == "train":
        _train(args.features, args.out)
    elif args.command == "predict":
        _predict(args.dataset, args.model_dir, args.head, args.out)


def _read_days(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    days = payload["days"] if isinstance(payload, dict) else payload
    if not isinstance(days, list):
        raise ValueError("days.json must contain a list of days")
    return days


def _build(ticks_dir: Path, days_path: Path, out_dir: Path) -> None:
    days = _read_days(days_path)
    candidate_groups: dict[tuple[str, str], list[ChartSample]] = defaultdict(list)
    natural_train_counts = dict.fromkeys(CHART_LABELS, 0)
    days_by_date = {str(day["date"]): day for day in days}
    symbols = ("BTC-USDT", "ETH-USDT", "SOL-USDT")
    for day in days:
        date = str(day["date"])
        split = str(day["split"])
        for symbol in symbols:
            native = symbol.replace("-", "")
            path = ticks_dir / f"{native}-{date}.csv"
            ticks = _load_ticks(path)
            stride_seconds = 10 if split == "train" else 15
            candidates = build_samples(
                ticks,
                symbol,
                split,
                stride_seconds=stride_seconds,
                max_samples=None,
            )
            if split == "train":
                for sample in candidates:
                    natural_train_counts[sample.label] += 1
                candidates = _balance_train_samples(candidates)
            candidate_groups[(split, symbol)].extend(candidates)

    out_dir.mkdir(parents=True, exist_ok=True)
    ticks_output = out_dir / "ticks"
    ticks_output.mkdir(parents=True, exist_ok=True)
    for date in days_by_date:
        for symbol in symbols:
            native = symbol.replace("-", "")
            shutil.copyfile(
                ticks_dir / f"{native}-{date}.csv",
                ticks_output / f"{native}-{date}.csv",
            )
    selected_groups: dict[tuple[str, str], list[ChartSample]] = {}
    selected_counts = {
        split: dict.fromkeys(CHART_LABELS, 0) for split in ("train", "val", "test")
    }
    for (split, symbol), candidates in sorted(candidate_groups.items()):
        selected = (
            _evenly_sample(candidates, 4000) if split in ("val", "test") else candidates
        )
        selected_groups[(split, symbol)] = selected
        for sample in selected:
            selected_counts[split][sample.label] += 1

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for (split, symbol), selected in sorted(selected_groups.items()):
            by_date: dict[str, list[ChartSample]] = defaultdict(list)
            for sample in selected:
                date = (
                    datetime.fromtimestamp(sample.anchor_ms / 1000, tz=UTC)
                    .date()
                    .isoformat()
                )
                if date not in days_by_date:
                    raise ValueError(
                        f"sample anchor date {date} is absent from days.json"
                    )
                by_date[date].append(sample)
            native = symbol.replace("-", "")
            for date, day_samples in sorted(by_date.items()):
                ticks = _load_ticks(ticks_dir / f"{native}-{date}.csv")
                for sample in day_samples:
                    relative_image = (
                        Path("images") / split / f"{symbol}-{sample.anchor_ms}.png"
                    )
                    image_path = out_dir / relative_image
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    render_chart(ticks, sample.anchor_ms).save(image_path, format="PNG")
                    record = sample_dict(sample)
                    record["sample_id"] = f"{symbol}-{sample.anchor_ms}"
                    record["image_path"] = relative_image.as_posix()
                    samples_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    (out_dir / "days.json").write_text(
        json.dumps({"days": days}, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "dataset.json").write_text(
        json.dumps(
            {
                "natural_train_counts": natural_train_counts,
                "selected_counts": selected_counts,
                "sampling_rules": {
                    "train": (
                        "For each symbol and day, use a 10-second stride, keep every "
                        "non-range_bound candidate, and evenly sample range_bound "
                        "candidates up to twice the number of movement candidates."
                    ),
                    "val_test": (
                        "For each symbol and day, use a 15-second stride and preserve "
                        "the natural label distribution; evenly sample at most 4,000 "
                        "candidates per split and symbol."
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _balance_train_samples(samples: list[ChartSample]) -> list[ChartSample]:
    movement = [sample for sample in samples if sample.label != NORMAL_LABEL]
    range_bound = [sample for sample in samples if sample.label == NORMAL_LABEL]
    selected_range = _evenly_sample(
        range_bound, min(len(range_bound), 2 * len(movement))
    )
    selected_anchors = {sample.anchor_ms for sample in selected_range}
    return [
        sample
        for sample in samples
        if sample.label != NORMAL_LABEL or sample.anchor_ms in selected_anchors
    ]


def _evenly_sample(samples: list[ChartSample], maximum: int) -> list[ChartSample]:
    if maximum <= 0:
        return []
    if len(samples) <= maximum:
        return samples
    return [samples[index * len(samples) // maximum] for index in range(maximum)]


def _load_ticks(path: Path) -> list[Tick]:
    from .chart import load_ticks

    return load_ticks(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract(
    dataset_dir: Path,
    model_dir: str,
    output_path: Path,
    limit: int | None,
    limit_per_split: int | None,
) -> None:
    import torch
    import numpy as np

    from .backbone import BackboneConfig, MiniCPMVisionBackbone

    records = _read_jsonl(dataset_dir / "samples.jsonl")
    dataset_metadata = json.loads(
        (dataset_dir / "dataset.json").read_text(encoding="utf-8")
    )
    if limit_per_split is not None:
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            split = str(record["split"])
            if counts[split] < limit_per_split:
                selected.append(record)
                counts[split] += 1
        records = selected
    if limit is not None:
        records = records[:limit]

    backbone = MiniCPMVisionBackbone(
        BackboneConfig(model_dir=model_dir, dtype="bfloat16", device="cuda")
    )
    if not records:
        raise ValueError("no dataset samples selected for feature extraction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_features = output_path.with_suffix(".embeds.f16")
    embedding_store: np.memmap | None = None
    embedding_shape: tuple[int, ...] | None = None
    for index, record in enumerate(records):
        with Image.open(dataset_dir / record["image_path"]) as image:
            visual = backbone.encode_visual(image).embeds
        embedding = visual.detach().to(device="cpu", dtype=torch.float16).numpy()
        if embedding_store is None:
            embedding_shape = tuple(embedding.shape)
            embedding_store = np.memmap(
                temporary_features,
                dtype=np.float16,
                mode="w+",
                shape=(len(records), *embedding_shape),
            )
        if tuple(embedding.shape) != embedding_shape:
            raise ValueError(
                "visual embedding shapes differ; expected fixed-size chart inputs"
            )
        embedding_store[index] = embedding
    if embedding_store is None:
        raise RuntimeError("feature extraction produced no visual embeddings")
    embedding_store.flush()
    payload = {
        "embeds": torch.from_numpy(embedding_store),
        "labels": torch.tensor(
            [CHART_LABELS.index(str(record["label"])) for record in records],
            dtype=torch.long,
        ),
        "splits": [str(record["split"]) for record in records],
        "sample_ids": [str(record.get("sample_id", "")) for record in records],
        "samples": records,
        "days": _read_days(dataset_dir / "days.json"),
        "dataset": dataset_metadata,
    }
    torch.save(payload, output_path)
    del payload
    del embedding_store
    temporary_features.unlink(missing_ok=True)


def _train(features_path: Path, output_path: Path) -> None:
    import torch
    from torch.nn import functional as F

    from .chart_head import ChartStateHead

    torch.manual_seed(SEED)
    payload = torch.load(
        features_path, map_location="cpu", weights_only=False, mmap=True
    )
    embeddings = payload["embeds"]
    labels = payload["labels"].long()
    splits = payload["splits"]
    logit_adjust_values = _compute_logit_adjust(payload["dataset"])
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    val_indices = [index for index, split in enumerate(splits) if split == "val"]
    if not train_indices or not val_indices:
        raise ValueError("features must include both train and val splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChartStateHead(hidden_size=embeddings.shape[-1]).to(device)
    logit_adjust = torch.tensor(logit_adjust_values, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(30):
        model.train()
        order = torch.tensor(train_indices)[
            torch.randperm(len(train_indices), generator=generator)
        ]
        for batch_start in range(0, len(order), 256):
            indices = order[batch_start : batch_start + 256]
            batch_x = embeddings[indices].to(device=device, dtype=torch.float32)
            batch_y = labels[indices].to(device)
            mask = torch.ones(batch_x.shape[:2], dtype=torch.bool, device=device)
            loss = F.cross_entropy(model(batch_x, mask), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_y = labels[val_indices].to(device)
        with torch.no_grad():
            val_logits = _logits_for_indices(
                model, embeddings, val_indices, device, torch
            )
            val_loss = float(F.cross_entropy(val_logits + logit_adjust, val_y).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    val_y = labels[val_indices].to(device)
    with torch.no_grad():
        raw_val_logits = _logits_for_indices(
            model, embeddings, val_indices, device, torch
        )
    _fit_adjusted_temperature(model, raw_val_logits, val_y, logit_adjust)
    with torch.no_grad():
        calibrated_logits = _logits_for_indices(
            model, embeddings, val_indices, device, torch
        )
        val_probabilities = torch.softmax(calibrated_logits, dim=-1)
        val_confidence, val_predictions = val_probabilities.max(dim=-1)
        label_recall: dict[str, float] = {}
        for label_index in sorted({int(index) for index in val_y.tolist()}):
            expected = val_y == label_index
            true_positives = (
                expected & (val_predictions == label_index) & (val_confidence >= 0.5)
            )
            label_recall[CHART_LABELS[label_index]] = float(
                true_positives.sum().item() / expected.sum().item()
            )
        val_metrics = {
            "nll": float(F.cross_entropy(calibrated_logits, val_y).item()),
            "accuracy": float(
                (calibrated_logits.argmax(-1) == val_y).float().mean().item()
            ),
            "samples": len(val_indices),
            "label_recall": label_recall,
            "macro_recall": sum(label_recall.values()) / len(label_recall),
        }

    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "hidden_size": int(embeddings.shape[-1]),
        "proj": 256,
        "label_order": list(CHART_LABELS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    sidecar = {
        "config": {
            "hidden_size": int(embeddings.shape[-1]),
            "proj": 256,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "batch_size": 256,
            "max_epochs": 30,
            "early_stopping_patience": 3,
            "seed": SEED,
            "temperature": float(model.temperature.item()),
            "best_epoch": best_epoch,
        },
        "label_order": list(CHART_LABELS),
        "backbone": BACKBONE_REPO,
        "data_days": payload.get("days", []),
        "logit_bias": model.logit_bias.detach().cpu().tolist(),
        "val_metrics": val_metrics,
        "classifier": CLASSIFIER,
        "shadow": True,
        "calibration_status": "uncalibrated",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


def _predict(dataset_dir: Path, model_dir: str, head_path: Path, out_dir: Path) -> None:
    import torch

    from .backbone import BackboneConfig, MiniCPMVBackbone
    from .chart_head import ChartStateHead
    from .chart import chart_evidence, load_ticks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    model = ChartStateHead(
        hidden_size=int(checkpoint["hidden_size"]),
        proj=int(checkpoint["proj"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    backbone = MiniCPMVisionBackbone(
        BackboneConfig(model_dir=model_dir, dtype="bfloat16", device="cuda")
    )

    samples = [
        item
        for item in _read_jsonl(dataset_dir / "samples.jsonl")
        if item["split"] == "test"
    ]
    tick_cache: dict[tuple[str, str], list[Tick]] = {}
    records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        symbol = str(sample["symbol"])
        anchor_ms = int(sample["anchor_ms"])
        date = datetime.fromtimestamp(anchor_ms / 1000, tz=UTC).date().isoformat()
        native = symbol.replace("-", "")
        key = native, date
        if key not in tick_cache:
            tick_cache[key] = load_ticks(dataset_dir / "ticks" / f"{native}-{date}.csv")
        ticks = tick_cache[key]

        _sync_device(device)
        started = time.perf_counter()
        image = render_chart(ticks, anchor_ms)
        visual = backbone.encode_visual(image).embeds.to(
            device=device, dtype=torch.float32
        )
        visual = visual.unsqueeze(0)
        mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = model(visual, mask)
            probs = torch.softmax(logits, dim=-1)[0]
        _sync_device(device)
        latency_ms = (time.perf_counter() - started) * 1000
        label, probability, abstained = model.decide(probs)
        expected = str(sample["label"])
        records.append(
            {
                "predicted_label": label,
                "expected_label": expected,
                "probability": probability,
                "abstained": abstained,
                "latency_ms": latency_ms,
                "normal_label": "range_bound",
                "classifier": CLASSIFIER,
            }
        )
        if sample.get("persistence_label") is not None:
            persistence_bps = float(sample["persistence_bps"])
            baseline_records.append(
                {
                    "predicted_label": str(sample["persistence_label"]),
                    "expected_label": expected,
                    "probability": min(0.95, 0.55 + abs(persistence_bps) / 250),
                    "abstained": False,
                    "latency_ms": 0.0,
                    "normal_label": "range_bound",
                    "classifier": "persistence_baseline",
                }
            )
        details.append(
            {
                "sample_id": sample.get("sample_id"),
                "symbol": symbol,
                "anchor_ms": anchor_ms,
                "split": "test",
                "expected_label": expected,
                "predicted_label": label,
                "probability": probability,
                "abstained": abstained,
                "latency_ms": latency_ms,
                "probabilities": {
                    class_label: float(probs[index].item())
                    for index, class_label in enumerate(CHART_LABELS)
                },
                "evidence": chart_evidence(ticks, anchor_ms),
            }
        )
    _write_jsonl(out_dir / "records.jsonl", records)
    _write_jsonl(out_dir / "baseline_records.jsonl", baseline_records)
    _write_jsonl(out_dir / "predictions.jsonl", details)


def _sync_device(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compute_logit_adjust(dataset: dict[str, Any]) -> list[float]:
    natural_counts = dataset["natural_train_counts"]
    selected_counts = dataset["selected_counts"]["train"]
    natural_total = sum(int(natural_counts[label]) for label in CHART_LABELS)
    selected_total = sum(int(selected_counts[label]) for label in CHART_LABELS)
    if natural_total <= 0 or selected_total <= 0:
        raise ValueError("logit adjustment requires non-empty train counts")
    if any(
        int(natural_counts[label]) <= 0 or int(selected_counts[label]) <= 0
        for label in CHART_LABELS
    ):
        raise ValueError("logit adjustment requires train examples for every label")
    return [
        math.log(int(natural_counts[label]) / natural_total)
        - math.log(int(selected_counts[label]) / selected_total)
        for label in CHART_LABELS
    ]


def _fit_adjusted_temperature(
    model: Any,
    raw_logits: Any,
    targets: Any,
    logit_adjust: Any,
) -> Any:
    import torch

    adjusted_logits = raw_logits + logit_adjust
    with torch.no_grad():
        model.logit_bias.copy_(logit_adjust)
    model.fit_temperature(adjusted_logits, targets)
    return adjusted_logits


def _logits_for_indices(
    model: Any, embeddings: Any, indices: list[int], device: Any, torch: Any
) -> Any:
    chunks = []
    for start in range(0, len(indices), 256):
        batch_indices = indices[start : start + 256]
        visual = embeddings[batch_indices].to(device=device, dtype=torch.float32)
        mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=device)
        chunks.append(model(visual, mask))
    return torch.cat(chunks, dim=0)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
