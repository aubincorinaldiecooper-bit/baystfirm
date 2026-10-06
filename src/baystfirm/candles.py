from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from baystfirm.config import Settings
from baystfirm.models import Candle
from baystfirm.storage import EventStore

CANDLE_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
)
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14_400,
    "6h": 21_600,
    "12h": 43_200,
    "1d": 86_400,
    "1w": 604_800,
}
WEEK_ORIGIN_MS = 4 * 86_400_000
SPOT_ONLY_VENUES = frozenset({"coinbase", "kraken", "binanceus"})
PERPETUAL_VENUES = frozenset({"bybit", "okx"})
MAX_LIMIT = 500
MAX_NATIVE_PAGE = {"coinbase": 300, "kraken": 720, "bybit": 1000, "okx": 300, "binanceus": 1000}
MAX_OKX_HISTORY_PAGE = 300
OKX_HISTORY_PAGE_DELAY_SECONDS = 0.11

NATIVE_INTERVALS: dict[str, dict[str, str]] = {
    "coinbase": {
        "1m": "60",
        "5m": "300",
        "15m": "900",
        "1h": "3600",
        "6h": "21600",
        "1d": "86400",
    },
    "kraken": {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "1440",
    },
    "bybit": {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
    },
    "okx": {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6Hutc",
        "12h": "12Hutc",
        "1d": "1Dutc",
        "1w": "1Wutc",
    },
    "binanceus": {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
    },
}
SOURCE_URL_TEMPLATES = {
    "coinbase": (
        "https://api.exchange.coinbase.com/products/{product_id}/candles"
        "?granularity={seconds}&start={start}&end={end}"
    ),
    "kraken": "https://api.kraken.com/0/public/OHLC?pair={pair}&interval={minutes}&since={since}",
    "bybit": "https://api.bybit.com/v5/market/kline?category={category}&symbol={symbol}&interval={interval}&limit={limit}",
    "okx": "https://www.okx.com/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}",
    "binanceus": "https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
}
SOURCE_URLS = {
    "coinbase": "https://api.exchange.coinbase.com/products/{product_id}/candles",
    "kraken": "https://api.kraken.com/0/public/OHLC",
    "bybit": "https://api.bybit.com/v5/market/kline",
    "okx": "https://www.okx.com/api/v5/market/candles",
    "binanceus": "https://api.binance.us/api/v3/klines",
}
OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"


class CandleNotFound(Exception):
    pass


class CandleUnavailable(Exception):
    pass


def source_interval(venue: str, interval: str) -> str:
    native = NATIVE_INTERVALS[venue]
    if interval in native:
        return interval
    seconds = INTERVAL_SECONDS[interval]
    divisors = [
        candidate
        for candidate in native
        if INTERVAL_SECONDS[candidate] < seconds and seconds % INTERVAL_SECONDS[candidate] == 0
    ]
    if not divisors:
        raise CandleUnavailable(f"{venue} has no native interval that divides {interval}.")
    return max(divisors, key=INTERVAL_SECONDS.__getitem__)


def validate_candle_request(
    settings: Settings, venue: str, symbol: str, interval: str
) -> tuple[str, str]:
    venue = venue.lower()
    symbol = symbol.upper()
    if venue not in NATIVE_INTERVALS or venue not in settings.enabled_venues:
        raise CandleNotFound("Unknown venue.")
    if symbol not in settings.symbols:
        raise CandleNotFound("Unknown symbol.")
    if symbol.endswith("-PERP") and venue not in PERPETUAL_VENUES:
        raise CandleNotFound("This venue does not provide perpetual candles.")
    if interval not in CANDLE_INTERVALS:
        raise CandleNotFound("Unsupported candle interval.")
    return venue, symbol


def bucket_open_time(open_time: int, interval: str) -> int:
    if interval == "1w":
        return WEEK_ORIGIN_MS + ((open_time - WEEK_ORIGIN_MS) // 604_800_000) * 604_800_000
    milliseconds = INTERVAL_SECONDS[interval] * 1000
    return (open_time // milliseconds) * milliseconds


def aggregate_candles(
    candles: list[Candle],
    source_interval_name: str,
    target_interval: str,
    *,
    now_ms: int | None = None,
) -> list[Candle]:
    source_seconds = INTERVAL_SECONDS[source_interval_name]
    target_seconds = INTERVAL_SECONDS[target_interval]
    if target_seconds % source_seconds != 0:
        raise ValueError("source interval must divide target interval")
    expected_count = target_seconds // source_seconds
    current_bucket = bucket_open_time(
        int(datetime.now(UTC).timestamp() * 1000) if now_ms is None else now_ms,
        target_interval,
    )
    groups: dict[int, list[Candle]] = defaultdict(list)
    for candle in sorted(candles, key=lambda item: item.open_time):
        groups[bucket_open_time(candle.open_time, target_interval)].append(candle)
    output: list[Candle] = []
    for bucket_time, group in sorted(groups.items()):
        if len(group) < expected_count and bucket_time != current_bucket:
            continue
        output.append(
            Candle(
                open_time=bucket_time,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
            )
        )
    return output


def _native_symbol(venue: str, symbol: str) -> str:
    base = symbol.removesuffix("-PERP")
    if venue == "kraken":
        base, quote = base.split("-", maxsplit=1)
        base = "XBT" if base == "BTC" else base
        return f"{base}{quote}"
    if venue == "bybit":
        return base.replace("-", "")
    if venue == "okx":
        return f"{base}-SWAP" if symbol.endswith("-PERP") else base
    if venue == "binanceus":
        return base.replace("-", "")
    return base


def _timestamp_ms(value: Any) -> int:
    timestamp = int(float(value))
    return timestamp * 1000 if timestamp < 100_000_000_000 else timestamp


def _coinbase_rows(body: Any) -> list[Candle]:
    if not isinstance(body, list):
        raise CandleUnavailable("Coinbase returned an unexpected candle response.")
    return [
        Candle(
            open_time=_timestamp_ms(row[0]),
            low=float(row[1]),
            high=float(row[2]),
            open=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in body
        if isinstance(row, list) and len(row) >= 6
    ]


def _kraken_rows(body: Any) -> tuple[list[Candle], int]:
    if not isinstance(body, dict):
        raise CandleUnavailable("Kraken returned an unexpected candle response.")
    errors = body.get("error", [])
    if errors:
        message = " ".join(str(item) for item in errors)
        if "unknown asset pair" in message.lower():
            raise CandleNotFound(f"Kraken does not list this pair: {message}")
        raise CandleUnavailable(f"Kraken candle API error: {message}")
    result = body.get("result")
    if not isinstance(result, dict):
        raise CandleUnavailable("Kraken returned an unexpected candle response.")
    pair_key = next((key for key in result if key != "last"), None)
    if pair_key is None:
        return [], int(result.get("last", 0))
    rows = result.get(pair_key, [])
    if not isinstance(rows, list):
        raise CandleUnavailable("Kraken returned an unexpected candle response.")
    candles = [
        Candle(
            open_time=_timestamp_ms(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[6]),
        )
        for row in rows
        if isinstance(row, list) and len(row) >= 8
    ]
    return candles, int(result.get("last", 0))


def _bybit_rows(body: Any) -> list[Candle]:
    if not isinstance(body, dict):
        raise CandleUnavailable("Bybit returned an unexpected candle response.")
    code = int(body.get("retCode", -1))
    if code != 0:
        message = str(body.get("retMsg", "unknown upstream error"))
        if code == 10001 and any(
            token in message.lower() for token in ("symbol", "category", "instrument")
        ):
            raise CandleNotFound(f"Bybit does not list this pair: {message}")
        raise CandleUnavailable(f"Bybit candle API error {code}: {message}")
    result = body.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("list"), list):
        raise CandleUnavailable("Bybit returned an unexpected candle response.")
    return [
        Candle(
            open_time=_timestamp_ms(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in result["list"]
        if isinstance(row, list) and len(row) >= 7
    ]


def _okx_rows(body: Any) -> list[Candle]:
    if not isinstance(body, dict):
        raise CandleUnavailable("OKX returned an unexpected candle response.")
    code = str(body.get("code", ""))
    if code != "0":
        message = str(body.get("msg", "unknown upstream error"))
        if code == "51001":
            raise CandleNotFound(f"OKX does not list this instrument: {message}")
        raise CandleUnavailable(f"OKX candle API error {code}: {message}")
    rows = body.get("data")
    if not isinstance(rows, list):
        raise CandleUnavailable("OKX returned an unexpected candle response.")
    return [
        Candle(
            open_time=_timestamp_ms(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
        if isinstance(row, list) and len(row) >= 9
    ]


def _binanceus_rows(body: Any) -> list[Candle]:
    if isinstance(body, dict):
        code = body.get("code")
        message = str(body.get("msg", ""))
        if code == -1121:
            raise CandleNotFound(f"Binance.US does not list this symbol: {message}")
        raise CandleUnavailable(f"Binance.US candle API error: {message or code}")
    if not isinstance(body, list):
        raise CandleUnavailable("Binance.US returned an unexpected candle response.")
    return [
        Candle(
            open_time=_timestamp_ms(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in body
        if isinstance(row, list) and len(row) >= 12
    ]


PARSERS: dict[str, Callable[[Any], list[Candle]]] = {
    "coinbase": _coinbase_rows,
    "bybit": _bybit_rows,
    "okx": _okx_rows,
    "binanceus": _binanceus_rows,
}


def _iso_timestamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat().replace("+00:00", "Z")


def _body_json(response: httpx.Response, venue: str) -> Any:
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    if response.status_code == 404:
        raise CandleNotFound(f"{venue} does not list this symbol.")
    if response.status_code in {400, 422}:
        text = response.text.lower()
        if any(
            word in text
            for word in ("invalid symbol", "unknown symbol", "not found", "does not exist")
        ):
            raise CandleNotFound(f"{venue} does not list this symbol.")
    if response.status_code >= 400:
        raise CandleUnavailable(f"{venue} candle API returned HTTP {response.status_code}.")
    if body is None:
        raise CandleUnavailable(f"{venue} returned a non-JSON candle response.")
    return body


async def _fetch_coinbase(
    client: httpx.AsyncClient, product_id: str, granularity: str, source_limit: int
) -> list[Candle]:
    step = int(granularity)
    now_seconds = int(datetime.now(UTC).timestamp())
    latest_open = (now_seconds // step) * step
    cursor = latest_open - (source_limit - 1) * step
    output: list[Candle] = []
    url = SOURCE_URLS["coinbase"].format(product_id=product_id)
    while cursor <= now_seconds:
        end = min(now_seconds, cursor + (MAX_NATIVE_PAGE["coinbase"] - 1) * step)
        response = await client.get(
            url,
            params={
                "granularity": granularity,
                "start": _iso_timestamp(cursor),
                "end": _iso_timestamp(end),
            },
        )
        output.extend(_coinbase_rows(_body_json(response, "Coinbase")))
        if end >= now_seconds:
            break
        cursor = end + step
    return output


async def _fetch_kraken(
    client: httpx.AsyncClient, pair: str, interval_minutes: str, source_limit: int
) -> list[Candle]:
    step = int(interval_minutes) * 60
    now_seconds = int(datetime.now(UTC).timestamp())
    latest_open = (now_seconds // step) * step
    cursor = max(0, latest_open - (source_limit - 1) * step)
    output: list[Candle] = []
    url = SOURCE_URLS["kraken"]
    pages = math.ceil(source_limit / MAX_NATIVE_PAGE["kraken"]) + 2
    for _ in range(pages):
        response = await client.get(
            url,
            params={"pair": pair, "interval": interval_minutes, "since": cursor},
        )
        rows, next_cursor = _kraken_rows(_body_json(response, "Kraken"))
        output.extend(rows)
        if len({item.open_time for item in output}) >= source_limit or next_cursor <= cursor:
            break
        cursor = next_cursor
    return output


async def _fetch_bybit(
    client: httpx.AsyncClient,
    category: str,
    symbol: str,
    interval: str,
    source_limit: int,
) -> list[Candle]:
    response = await client.get(
        SOURCE_URLS["bybit"],
        params={
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(source_limit, MAX_NATIVE_PAGE["bybit"]),
        },
    )
    body = _body_json(response, "Bybit")
    return _parse_and_raise(PARSERS["bybit"], body)


async def _fetch_okx(
    client: httpx.AsyncClient, inst_id: str, bar: str, source_limit: int
) -> list[Candle]:
    cursor: str | None = None
    output: list[Candle] = []
    pages = math.ceil(source_limit / MAX_OKX_HISTORY_PAGE) + 2
    for page in range(pages):
        unique_count = len({item.open_time for item in output})
        remaining = source_limit - unique_count
        if remaining <= 0:
            break
        page_limit = MAX_NATIVE_PAGE["okx"] if page == 0 else MAX_OKX_HISTORY_PAGE
        params: dict[str, str] = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(min(page_limit, remaining)),
        }
        if cursor is not None:
            params["after"] = cursor
        if page > 1:
            await asyncio.sleep(OKX_HISTORY_PAGE_DELAY_SECONDS)
        url = SOURCE_URLS["okx"] if page == 0 else OKX_HISTORY_URL
        response = await client.get(url, params=params)
        rows = _okx_rows(_body_json(response, "OKX"))
        if not rows:
            break
        output.extend(rows)
        oldest = min(item.open_time for item in rows)
        next_cursor = str(oldest)
        if len({item.open_time for item in output}) >= source_limit or next_cursor == cursor:
            break
        cursor = next_cursor
    unique = {item.open_time: item for item in output}
    return sorted(unique.values(), key=lambda candle: candle.open_time)


async def _fetch_binanceus(
    client: httpx.AsyncClient, symbol: str, interval: str, source_limit: int
) -> list[Candle]:
    response = await client.get(
        SOURCE_URLS["binanceus"],
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": min(source_limit, MAX_NATIVE_PAGE["binanceus"]),
        },
    )
    return _binanceus_rows(_body_json(response, "Binance.US"))


def _parse_and_raise(parser: Callable[[Any], list[Candle]], body: Any) -> list[Candle]:
    return parser(body)


async def fetch_source_candles(
    client: httpx.AsyncClient,
    venue: str,
    symbol: str,
    source_interval_name: str,
    source_limit: int,
) -> list[Candle]:
    native_interval = NATIVE_INTERVALS[venue][source_interval_name]
    native_symbol = _native_symbol(venue, symbol)
    if venue == "coinbase":
        return await _fetch_coinbase(client, native_symbol, native_interval, source_limit)
    if venue == "kraken":
        return await _fetch_kraken(client, native_symbol, native_interval, source_limit)
    if venue == "bybit":
        category = "linear" if symbol.endswith("-PERP") else "spot"
        return await _fetch_bybit(client, category, native_symbol, native_interval, source_limit)
    if venue == "okx":
        return await _fetch_okx(client, native_symbol, native_interval, source_limit)
    return await _fetch_binanceus(client, native_symbol, native_interval, source_limit)


class CandleService:
    def __init__(
        self,
        settings: Settings,
        store: EventStore,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self.settings = settings
        self.store = store
        self.client_factory = client_factory

    async def get(self, venue: str, symbol: str, interval: str, limit: int) -> dict[str, Any]:
        venue, symbol = validate_candle_request(self.settings, venue, symbol, interval)
        if not 1 <= limit <= MAX_LIMIT:
            raise CandleNotFound(f"Limit must be between 1 and {MAX_LIMIT}.")

        source_name = source_interval(venue, interval)
        aggregated_from = source_name if source_name != interval else None
        source_seconds = INTERVAL_SECONDS[source_name]
        target_seconds = INTERVAL_SECONDS[interval]
        source_limit = math.ceil(limit * target_seconds / source_seconds)
        source_url_template = SOURCE_URL_TEMPLATES[venue]
        cached, cached_at = await self.store.candles(venue, symbol, interval, limit)
        now = datetime.now(UTC)
        cache_ttl = min(INTERVAL_SECONDS[interval], 60)
        if cached and cached_at and now - cached_at <= timedelta(seconds=cache_ttl):
            return self._response(
                venue,
                symbol,
                interval,
                source_url_template,
                cached_at,
                cached,
                aggregated_from,
                limit,
                stale=False,
            )

        try:
            async with self.client_factory(timeout=10.0) as client:
                source_candles = await fetch_source_candles(
                    client,
                    venue,
                    symbol,
                    source_name,
                    source_limit,
                )
            unique = {candle.open_time: candle for candle in source_candles}
            normalized = sorted(unique.values(), key=lambda candle: candle.open_time)
            candles = (
                aggregate_candles(normalized, source_name, interval)
                if aggregated_from
                else normalized
            )
            candles = candles[-limit:]
            fetched_at = datetime.now(UTC)
            await self.store.store_candles(venue, symbol, interval, candles, fetched_at)
            return self._response(
                venue,
                symbol,
                interval,
                source_url_template,
                fetched_at,
                candles,
                aggregated_from,
                limit,
                stale=False,
            )
        except CandleNotFound:
            raise
        except Exception as error:
            if not cached:
                raise CandleUnavailable(str(error)) from error
            return self._response(
                venue,
                symbol,
                interval,
                source_url_template,
                cached_at or now,
                cached,
                aggregated_from,
                limit,
                stale=True,
                error=str(error),
            )

    @staticmethod
    def _response(
        venue: str,
        symbol: str,
        interval: str,
        source_url_template: str,
        fetched_at: datetime,
        candles: list[Candle],
        aggregated_from: str | None,
        limit: int,
        *,
        stale: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "venue": venue,
            "symbol": symbol,
            "interval": interval,
            "source_url_template": source_url_template,
            "fetched_at": fetched_at.isoformat(),
            "aggregated_from": aggregated_from,
            "candles": [candle.model_dump(mode="json") for candle in candles],
            "truncated": len(candles) < limit,
        }
        if stale:
            result["stale"] = True
            result["error"] = error or "Upstream request failed."
        else:
            result["stale"] = False
        return result
