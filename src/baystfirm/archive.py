from __future__ import annotations

import csv
import hashlib
import os
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from urllib.request import urlopen

from baystfirm.models import Side

TradeRow = tuple[int, float, float, Side, int]
ArchiveRecord = tuple[TradeRow, str]
ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "BTC", "ETH", "BNB", "EUR", "USD")


def parse_trade_line(line: str) -> TradeRow | None:
    """Parse one Binance aggTrades CSV row; return None for headers and blanks."""
    text = line.rstrip("\r\n")
    if not text:
        return None
    fields = next(csv.reader([text]))
    try:
        agg_id = int(fields[0])
        raw_timestamp = int(fields[5])
        price = float(fields[1])
        size = float(fields[2])
    except (IndexError, ValueError):
        if fields and fields[0].strip().lower() in {"agg_trade_id", "aggregate_trade_id"}:
            return None
        raise ValueError(f"invalid Binance aggTrades row: {text!r}") from None
    timestamp_ms = raw_timestamp // 1000 if raw_timestamp > 1e14 else raw_timestamp
    maker = fields[6].strip().lower() in {"true", "1"}
    side = Side.SELL if maker else Side.BUY
    return timestamp_ms, price, size, side, agg_id


def _iter_archive_records(path: Path) -> Iterator[ArchiveRecord]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path.name}, found {len(members)}")
        with archive.open(members[0]) as binary:
            for raw in binary:
                line = raw.decode("utf-8").rstrip("\r\n")
                parsed = parse_trade_line(line)
                if parsed is not None:
                    yield parsed, line


def iter_archive_trades(path: Path) -> Iterator[TradeRow]:
    for row, _ in _iter_archive_records(path):
        yield row


def _download_to_path(url: str, path: Path) -> None:
    with urlopen(url, timeout=60) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _expected_checksum(url: str) -> str:
    with urlopen(f"{url}.CHECKSUM", timeout=30) as response:
        contents = cast(str, response.read().decode("ascii").strip())
    fields = contents.split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ValueError(f"invalid Binance checksum response for {url}")
    return fields[0].lower()


def fetch_archive(symbol: str, date: str, cache_dir: Path = Path("var/archives")) -> Path:
    native_symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]+", native_symbol):
        raise ValueError(f"invalid Binance symbol: {symbol!r}")
    archive_name = f"{native_symbol}-aggTrades-{date}.zip"
    url = f"{ARCHIVE_BASE_URL}/{native_symbol}/{archive_name}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / archive_name
    expected = _expected_checksum(url)
    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".zip.part")
        try:
            _download_to_path(url, temporary_path)
            actual = _sha256(temporary_path)
            if actual != expected:
                raise ValueError(
                    f"Binance archive checksum mismatch for {archive_name}: "
                    f"expected {expected}, got {actual}"
                )
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    actual = _sha256(archive_path)
    if actual != expected:
        raise ValueError(
            f"Binance archive checksum mismatch for {archive_name}: "
            f"expected {expected}, got {actual}"
        )
    return archive_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_symbol(native_symbol: str) -> tuple[str, str, str]:
    native = native_symbol.upper()
    for quote in QUOTE_ASSETS:
        if native.endswith(quote) and len(native) > len(quote):
            base = native[: -len(quote)]
            return f"{base}-{quote}", base, quote
    raise ValueError(f"cannot determine quote asset for Binance symbol {native!r}")
