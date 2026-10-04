from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

BookSide = Literal["bids", "asks"]
Level = tuple[float, float]


def depth_within(levels: Iterable[Level], mid: float, bps: float) -> float | None:
    levels = list(levels)
    if not levels or mid <= 0 or bps < 0:
        return None
    distances = [abs(price - mid) * 10_000 / mid for price, _ in levels]
    if max(distances) < bps - 1e-9:
        return None
    return sum(
        price * size
        for (price, size), distance in zip(levels, distances, strict=True)
        if distance <= bps + 1e-9
    )


class LocalOrderBook:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def update(
        self,
        bids: Iterable[Level],
        asks: Iterable[Level],
        *,
        snapshot: bool = False,
    ) -> None:
        if snapshot:
            self.bids.clear()
            self.asks.clear()
        self._apply(self.bids, bids)
        self._apply(self.asks, asks)
        self.bids = dict(self.top_n("bids", self.depth))
        self.asks = dict(self.top_n("asks", self.depth))

    def top_n(self, side: BookSide, count: int | None = None) -> list[Level]:
        levels = self.bids if side == "bids" else self.asks
        ordered = sorted(levels.items(), reverse=side == "bids")
        return ordered if count is None else ordered[:count]

    @staticmethod
    def _apply(book_side: dict[float, float], levels: Iterable[Level]) -> None:
        for price, size in levels:
            if size == 0:
                book_side.pop(price, None)
            else:
                book_side[price] = size
