from __future__ import annotations

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


@dataclass(frozen=True)
class Settings:
    database_path: Path
    enabled_venues: tuple[str, ...]
    shadow_mode: bool
    symbols: tuple[str, ...]
    api_key: str | None = None

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
        )
