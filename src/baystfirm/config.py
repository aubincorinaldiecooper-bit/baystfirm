from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    enabled_venues: tuple[str, ...]
    shadow_mode: bool
    symbols: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        venues = tuple(
            value.strip().lower()
            for value in os.getenv("BAYST_VENUES", "coinbase,kraken,bybit").split(",")
            if value.strip()
        )
        symbols = tuple(
            value.strip().upper()
            for value in os.getenv(
                "BAYST_SYMBOLS",
                "BTC-USD,ETH-USD,USDC-USD,USDT-USD,BTC-USDT-PERP,ETH-USDT-PERP",
            ).split(",")
            if value.strip()
        )
        return cls(
            database_path=Path(os.getenv("BAYST_DATABASE_PATH", "var/baystfirm.db")),
            enabled_venues=venues,
            shadow_mode=os.getenv("BAYST_SHADOW_MODE", "true").lower() not in {"0", "false", "no"},
            symbols=symbols,
        )
