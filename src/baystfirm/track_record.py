from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any

from baystfirm.evaluation import NORMAL_LABELS, OUTCOME_CLASSIFIER
from baystfirm.storage import ClassificationRow, EventStore

NOTE = (
    "Overlapping predictions are scored individually, so samples are correlated and intervals "
    "are optimistic. Shadow signals have not passed the evaluation gate. Probabilities, not "
    "investment advice."
)
MAX_HORIZON_SECONDS = 86_400
CACHE_SECONDS = 60.0
OUTCOME_TOLERANCE_SECONDS = 5
WILSON_Z = 1.96


def build_track_record(
    rows: Sequence[ClassificationRow],
    window_hours: int,
    *,
    now: datetime,
) -> dict[str, Any]:
    if not 1 <= window_hours <= 168:
        raise ValueError("window_hours must be between 1 and 168")
    now = now.astimezone(UTC)
    window_start = now - timedelta(hours=window_hours)
    latest_matured_at = now - timedelta(seconds=OUTCOME_TOLERANCE_SECONDS)
    groups: dict[tuple[str, int], list[ClassificationRow]] = defaultdict(list)
    outcomes: dict[tuple[str, str, int], list[ClassificationRow]] = defaultdict(list)

    ordered_rows = sorted(rows, key=lambda row: row[3])
    for row in ordered_rows:
        classifier, symbol, horizon_seconds = row[0], row[1], row[2]
        groups[(classifier, horizon_seconds)].append(row)
        if not row[6]:
            outcomes[(classifier, symbol, horizon_seconds)].append(row)

    outcome_times = {
        key: [datetime.fromisoformat(row[3]) for row in predictions]
        for key, predictions in outcomes.items()
    }
    result_groups: list[dict[str, Any]] = []
    for (classifier, horizon_seconds), predictions in sorted(groups.items()):
        in_window: list[ClassificationRow] = []
        pending = 0
        for prediction in predictions:
            observed_at = datetime.fromisoformat(prediction[3])
            matured_at = observed_at + timedelta(seconds=horizon_seconds)
            if matured_at > latest_matured_at:
                pending += 1
            elif window_start <= matured_at <= latest_matured_at:
                in_window.append(prediction)

        scored: list[tuple[ClassificationRow, ClassificationRow]] = []
        unmatched = 0
        outcome_classifier = OUTCOME_CLASSIFIER.get(classifier, classifier)
        for prediction in in_window:
            if prediction[6]:
                continue
            target = datetime.fromisoformat(prediction[3]) + timedelta(seconds=horizon_seconds)
            outcome_key = (outcome_classifier, prediction[1], horizon_seconds)
            candidates = outcome_times.get(outcome_key, [])
            index = _first_at_or_after(candidates, target)
            if index is None:
                unmatched += 1
                continue
            scored.append((prediction, outcomes[outcome_key][index]))

        hits = sum(prediction[4] == outcome[4] for prediction, outcome in scored)
        scored_count = len(scored)
        normal_label = NORMAL_LABELS.get(classifier, "")
        baseline_count = sum(outcome[4] == normal_label for _, outcome in scored)
        by_label: dict[str, list[tuple[ClassificationRow, ClassificationRow]]] = defaultdict(list)
        for pair in scored:
            by_label[pair[1][4]].append(pair)
        label_recall = {
            label: sum(prediction[4] == label for prediction, _ in label_pairs) / len(label_pairs)
            for label, label_pairs in sorted(by_label.items())
        }
        latest = predictions[-1]
        result_groups.append(
            {
                "classifier": classifier,
                "horizon_seconds": horizon_seconds,
                "classifier_version": latest[8],
                "shadow": bool(latest[9]),
                "calibration_status": latest[10],
                "predictions": len(in_window),
                "abstained": sum(bool(prediction[6]) for prediction in in_window),
                "scored": scored_count,
                "unmatched": unmatched,
                "pending": pending,
                "hits": hits,
                "hit_rate": hits / scored_count if scored_count else None,
                "hit_rate_ci95": _wilson_interval(hits, scored_count),
                "baseline_hit_rate": baseline_count / scored_count if scored_count else None,
                "label_recall": label_recall,
            }
        )

    return {
        "computed_at": now.isoformat(),
        "window_hours": window_hours,
        "window_start": window_start.isoformat(),
        "groups": result_groups,
        "note": NOTE,
    }


def _first_at_or_after(times: list[datetime], target: datetime) -> int | None:
    index = bisect_left(times, target)
    if index < len(times) and times[index] <= target + timedelta(seconds=OUTCOME_TOLERANCE_SECONDS):
        return index
    return None


def _wilson_interval(hits: int, samples: int) -> list[float] | None:
    if samples == 0:
        return None
    proportion = hits / samples
    z_squared = WILSON_Z**2
    denominator = 1 + z_squared / samples
    center = (proportion + z_squared / (2 * samples)) / denominator
    margin = (
        WILSON_Z
        * ((proportion * (1 - proportion) / samples + z_squared / (4 * samples**2)) ** 0.5)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


class TrackRecordService:
    def __init__(self, store: EventStore) -> None:
        self.store = store
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}

    async def get(self, window_hours: int) -> dict[str, Any]:
        if not 1 <= window_hours <= 168:
            raise ValueError("window_hours must be between 1 and 168")
        cache_entry = self._cache.get(window_hours)
        if cache_entry is not None and monotonic() - cache_entry[0] < CACHE_SECONDS:
            return cache_entry[1]

        now = datetime.now(UTC)
        since = now - timedelta(hours=window_hours, seconds=MAX_HORIZON_SECONDS)
        rows = await self.store.classification_rows_since(since)
        response = await asyncio.to_thread(
            build_track_record,
            rows,
            window_hours,
            now=now,
        )
        self._cache[window_hours] = (monotonic(), response)
        return response
