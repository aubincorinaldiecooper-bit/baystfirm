from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class InstrumentKind(StrEnum):
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURE = "future"
    OPTION = "option"
    TOKENIZED_ASSET = "tokenized_asset"


class EventType(StrEnum):
    TRADE = "trade"
    QUOTE = "quote"
    BOOK = "book"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    CHAIN = "chain"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class CalibrationStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    VALIDATED = "validated"


class MarketEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    venue: str
    symbol: str
    native_symbol: str
    base_asset: str
    quote_asset: str
    instrument_kind: InstrumentKind
    event_type: EventType
    exchange_timestamp: datetime
    received_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sequence: int | str | None = None
    price: float | None = None
    size: float | None = None
    side: Side = Field(
        default=Side.UNKNOWN,
        description=(
            "For Bybit liquidations this is the position side (BUY means a long was liquidated; "
            "SELL means a short). For OKX it is the liquidation order side; the position side is "
            "preserved in metadata."
        ),
    )
    bid: float | None = None
    ask: float | None = None
    funding_rate: float | None = None
    next_funding_at: datetime | None = None
    open_interest: float | None = None
    open_interest_value: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    bid_depth_10bps: float | None = None
    ask_depth_10bps: float | None = None
    bid_depth_50bps: float | None = None
    ask_depth_50bps: float | None = None
    depth_levels: int | None = None
    payload_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_market_values(self) -> MarketEvent:
        if self.event_type is EventType.TRADE and (self.price is None or self.size is None):
            raise ValueError("trade events require price and size")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("bid cannot be greater than ask")
        if self.exchange_timestamp.tzinfo is None or self.received_timestamp.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return self

    @property
    def latency_ms(self) -> float:
        return max(
            0.0,
            (self.received_timestamp - self.exchange_timestamp).total_seconds() * 1000,
        )


class Evidence(BaseModel):
    metric: str
    value: float | str
    threshold: float | str | None = None
    source_event_ids: list[UUID] = Field(default_factory=list)


class Classification(BaseModel):
    classification_id: UUID = Field(default_factory=uuid4)
    classifier: str
    classifier_version: str
    symbol: str
    label: str
    probability: float = Field(ge=0, le=1)
    abstained: bool
    horizon_seconds: int = Field(gt=0)
    observed_at: datetime
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence: list[Evidence]
    shadow: bool = True
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    freshness_ms: float = Field(ge=0)


def payload_digest(payload: str | bytes) -> str:
    raw = payload.encode() if isinstance(payload, str) else payload
    return sha256(raw).hexdigest()
