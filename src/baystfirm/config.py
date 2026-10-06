from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VENUES = "coinbase,kraken,bybit,okx,binanceus"
DEFAULT_SYMBOLS = ",".join(
    (
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "USDT-USD",
        "USDC-USD",
        "PYUSD-USD",
        "DAI-USD",
        "USDC-USDT",
        "DAI-USDT",
        "USDE-USDT",
        "FDUSD-USDT",
        "PYUSD-USDT",
        "BTC-USDT-PERP",
        "ETH-USDT-PERP",
        "SOL-USDT-PERP",
    )
)
DEFAULT_SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"


@dataclass(frozen=True)
class Settings:
    database_path: Path
    enabled_venues: tuple[str, ...]
    shadow_mode: bool
    symbols: tuple[str, ...]
    api_key: str | None = None
    solana_rpc_url: str = DEFAULT_SOLANA_RPC_URL
    rugcheck_api_key: str | None = None
    solana_tokens_enabled: bool = False
    event_retention_hours: float = 3.0
    classification_retention_days: float = 3.0
    news_enabled: bool = True
    sec_user_agent: str = "Baystfirm/0.1 aubincorinaldiecooper@gmail.com"

    def __post_init__(self) -> None:
        if not math.isfinite(self.event_retention_hours) or self.event_retention_hours <= 0:
            raise ValueError("event_retention_hours must be greater than 0")
        if (
            not math.isfinite(self.classification_retention_days)
            or self.classification_retention_days <= 0
        ):
            raise ValueError("classification_retention_days must be greater than 0")

    @classmethod
    def from_env(cls) -> Settings:
        venues = tuple(
            value.strip().lower()
            for value in os.getenv("BAYST_VENUES", DEFAULT_VENUES).split(",")
            if value.strip()
        )
        symbols = tuple(
            value.strip().upper()
            for value in os.getenv("BAYST_SYMBOLS", DEFAULT_SYMBOLS).split(",")
            if value.strip()
        )
        return cls(
            database_path=Path(os.getenv("BAYST_DATABASE_PATH", "var/baystfirm.db")),
            enabled_venues=venues,
            shadow_mode=os.getenv("BAYST_SHADOW_MODE", "true").lower() not in {"0", "false", "no"},
            symbols=symbols,
            api_key=os.getenv("BAYST_API_KEY", "").strip() or None,
            solana_rpc_url=os.getenv("BAYST_SOLANA_RPC_URL", DEFAULT_SOLANA_RPC_URL),
            rugcheck_api_key=os.getenv("BAYST_RUGCHECK_API_KEY", "").strip() or None,
            solana_tokens_enabled=os.getenv("BAYST_SOLANA_TOKENS", "false").lower()
            not in {"0", "false", "no"},
            event_retention_hours=float(os.getenv("BAYST_EVENT_RETENTION_HOURS", "3.0")),
            classification_retention_days=float(
                os.getenv("BAYST_CLASSIFICATION_RETENTION_DAYS", "3.0")
            ),
            news_enabled=os.getenv("BAYST_NEWS", "true").lower() not in {"0", "false", "no"},
            sec_user_agent=os.getenv(
                "BAYST_SEC_USER_AGENT",
                "Baystfirm/0.1 aubincorinaldiecooper@gmail.com",
            ),
        )
