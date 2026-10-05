from __future__ import annotations

import asyncio
import base64
import logging
import math
import struct
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from time import monotonic
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel
from solders.pubkey import Pubkey

from baystfirm.config import DEFAULT_SOLANA_RPC_URL, Settings

logger = logging.getLogger(__name__)
FactStatus = Literal["ok", "unavailable", "not_applicable"]
FactSource = Literal["solana_rpc", "dexscreener", "geckoterminal", "raydium", "rugcheck"]
PoolType = Literal["launch_curve", "lp_token", "position_based", "unknown"]

SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
METADATA_PROGRAM = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
RAYDIUM_AMM_V4_AUTHORITY = "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1"
RISKY_TOKEN_EXTENSIONS = (
    "transferFeeConfig",
    "permanentDelegate",
    "transferHook",
    "mintCloseAuthority",
    "defaultAccountState",
    "nonTransferable",
    "confidentialTransferMint",
)
FACT_NAMES = (
    "mint_authority",
    "freeze_authority",
    "token_extensions",
    "metadata_mutable",
    "top10_share",
    "liquidity_lock",
    "market",
)
SOURCE_NAMES: tuple[FactSource, ...] = (
    "solana_rpc",
    "dexscreener",
    "geckoterminal",
    "raydium",
    "rugcheck",
)
SOURCE_RATE_LIMITS = {
    "solana_rpc": 0.25,
    "solana_largest": 0.5,
    "dexscreener": 0.25,
    "raydium": 0.5,
    "rugcheck": 1.0,
}
TOP_HOLDER_RETRY_SECONDS = 600
TOP_HOLDER_RATE_LIMIT_RETRY_SECONDS = 60
MAX_TOP_HOLDER_ATTEMPTS = 6
GECKOTERMINAL_BUCKET_CAPACITY = 10.0
GECKOTERMINAL_REFILL_SECONDS = 6.0
GECKOTERMINAL_BACKGROUND_RESERVE = 3.0
GECKOTERMINAL_INTERACTIVE_TIMEOUT_SECONDS = 5.0
GECKOTERMINAL_INITIAL_BACKOFF_SECONDS = 60.0
GECKOTERMINAL_MAX_BACKOFF_SECONDS = 600.0
DEXSCREENER_TOKENS_URL = "https://api.dexscreener.com/tokens/v1/solana"
DEXSCREENER_TOKEN_PAIRS_URL = "https://api.dexscreener.com/token-pairs/v1/solana"
DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
GECKOTERMINAL_NEW_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
GECKOTERMINAL_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools"
GECKOTERMINAL_TOKEN_INFO_URL = "https://api.geckoterminal.com/api/v2/networks/solana/tokens"
RAYDIUM_POOL_INFO_URL = "https://api-v3.raydium.io/pools/info/ids"
RUGCHECK_SUMMARY_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary"
POOL_TYPE_RULES: tuple[tuple[str, frozenset[str], PoolType], ...] = (
    ("pumpfun", frozenset(), "launch_curve"),
    ("launchlab", frozenset(), "launch_curve"),
    ("launchlab", frozenset({"launchlab"}), "launch_curve"),
    ("raydium", frozenset({"clmm"}), "position_based"),
    ("raydium", frozenset({"cpmm"}), "lp_token"),
    ("raydium", frozenset({"standard"}), "lp_token"),
    ("raydium", frozenset(), "lp_token"),
    ("orca", frozenset({"wp"}), "position_based"),
    ("orca", frozenset({"whirlpool"}), "position_based"),
    ("meteora", frozenset({"dlmm"}), "position_based"),
    ("meteora", frozenset({"dyn2"}), "position_based"),
    ("meteora", frozenset({"damm v2"}), "position_based"),
)


class Fact(BaseModel):
    status: FactStatus
    value: Any = None
    source: FactSource
    fetched_at: str | None = None
    detail: str | None = None


class TokenFacts(BaseModel):
    mint_authority: Fact
    freeze_authority: Fact
    token_extensions: Fact
    metadata_mutable: Fact
    top10_share: Fact
    liquidity_lock: Fact
    market: Fact


class Holder(BaseModel):
    owner: str
    pct: float
    is_pool: bool


class SecondOpinion(BaseModel):
    provider: Literal["RugCheck"] = "RugCheck"
    status: Literal["ok", "unavailable"]
    fetched_at: str | None = None
    score_normalised: int | None = None
    lp_locked_pct: float | None = None
    risks: list[dict[str, str]]


class TokenCard(BaseModel):
    mint: str
    name: str | None = None
    symbol: str | None = None
    image_url: str | None = None
    token_program: Literal["spl-token", "token-2022"] | None = None
    first_seen_at: str
    checked_at: str
    facts: TokenFacts
    second_opinion: SecondOpinion


class NotTokenMint(ValueError):
    pass


class SourceError(RuntimeError):
    def __init__(self, source: FactSource, detail: str) -> None:
        super().__init__(detail)
        self.source = source
        self.detail = detail


class AsyncRateLimiter:
    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Any = monotonic,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.sleep = sleep
        self._lock = asyncio.Lock()
        self._last_request: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            now = self.clock()
            if self._last_request is not None:
                delay = self.interval_seconds - (now - self._last_request)
                if delay > 0:
                    await self.sleep(delay)
            self._last_request = self.clock()


class GeckoTerminalTokenBucket:
    def __init__(
        self,
        *,
        capacity: float = GECKOTERMINAL_BUCKET_CAPACITY,
        refill_seconds: float = GECKOTERMINAL_REFILL_SECONDS,
        background_reserve: float = GECKOTERMINAL_BACKGROUND_RESERVE,
        interactive_timeout_seconds: float = GECKOTERMINAL_INTERACTIVE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self.background_reserve = background_reserve
        self.interactive_timeout_seconds = interactive_timeout_seconds
        self.clock = clock
        self.sleep = sleep
        self._tokens = capacity
        self._last_refill = clock()
        self._paused_until = 0.0
        self._backoff_seconds = GECKOTERMINAL_INITIAL_BACKOFF_SECONDS
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        self._refill(self.clock())
        return self._tokens

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_seconds

    @property
    def pause_remaining(self) -> float:
        return max(0.0, self._paused_until - self.clock())

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed:
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed / self.refill_seconds,
            )
            self._last_refill = now

    async def wait(self, *, priority: Literal["background", "interactive"] = "background") -> None:
        started_at = self.clock()
        deadline = (
            started_at + self.interactive_timeout_seconds if priority == "interactive" else None
        )
        while True:
            async with self._lock:
                now = self.clock()
                self._refill(now)
                if deadline is not None and now > deadline:
                    raise SourceError(
                        "geckoterminal",
                        "Couldn't check right now (rate limited)",
                    )
                pause_delay = self._paused_until - now
                if pause_delay > 0:
                    if priority == "interactive":
                        raise SourceError(
                            "geckoterminal",
                            "Couldn't check right now (rate limited)",
                        )
                    delay = pause_delay
                else:
                    reserve = self.background_reserve if priority == "background" else 0.0
                    if self._tokens >= 1.0 and (
                        priority == "interactive" or self._tokens > reserve
                    ):
                        self._tokens -= 1.0
                        return
                    if priority == "background":
                        delay = (
                            max(
                                (reserve - self._tokens) * self.refill_seconds,
                                (1.0 - self._tokens) * self.refill_seconds,
                            )
                            + 1e-6
                        )
                    else:
                        remaining = (deadline or now) - now
                        if remaining <= 0:
                            raise SourceError(
                                "geckoterminal",
                                "Couldn't check right now (rate limited)",
                            )
                        delay = min((1.0 - self._tokens) * self.refill_seconds, remaining)
            await self.sleep(delay)
            if deadline is not None and self.clock() >= deadline:
                async with self._lock:
                    now = self.clock()
                    self._refill(now)
                    if now < self._paused_until or self._tokens < 1.0:
                        raise SourceError(
                            "geckoterminal",
                            "Couldn't check right now (rate limited)",
                        )

    def record_rate_limit(self, retry_after: str | None = None) -> None:
        delay = self._retry_after_seconds(retry_after)
        if delay is None:
            delay = self._backoff_seconds
        delay = min(delay, GECKOTERMINAL_MAX_BACKOFF_SECONDS)
        self._paused_until = max(self._paused_until, self.clock() + delay)
        self._backoff_seconds = min(
            self._backoff_seconds * 2,
            GECKOTERMINAL_MAX_BACKOFF_SECONDS,
        )

    def record_success(self) -> None:
        self._backoff_seconds = GECKOTERMINAL_INITIAL_BACKOFF_SECONDS

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            return None
        return seconds if math.isfinite(seconds) and seconds >= 0 else None


def unavailable_fact(
    source: FactSource,
    detail: str,
    *,
    fetched_at: str | None = None,
) -> Fact:
    return Fact(status="unavailable", source=source, fetched_at=fetched_at, detail=detail)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_status(source: FactSource) -> dict[str, Any]:
    return {"name": source, "fetched_at": None, "ok": False, "error": None}


def _is_geckoterminal_rate_limit(fact: Fact) -> bool:
    return (
        fact.status == "unavailable"
        and fact.source == "geckoterminal"
        and fact.detail is not None
        and "rate limited" in fact.detail.casefold()
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _pair_liquidity(pair: Mapping[str, Any]) -> float:
    liquidity = pair.get("liquidity")
    value = liquidity.get("usd") if isinstance(liquidity, Mapping) else None
    return _optional_float(value) or 0.0


def _pair_volume_24h(pair: Mapping[str, Any]) -> float:
    volume = pair.get("volume")
    value = volume.get("h24") if isinstance(volume, Mapping) else None
    return _optional_float(value) or 0.0


def parse_dex_pairs(body: Any, mint: str) -> list[dict[str, Any]]:
    if not isinstance(body, list):
        return []
    pairs: list[dict[str, Any]] = []
    for pair in body:
        if not isinstance(pair, dict):
            continue
        base_token = pair.get("baseToken")
        if not isinstance(base_token, dict) or base_token.get("address") != mint:
            continue
        pairs.append(pair)
    return pairs


def parse_dex_search(body: Any, query: str) -> list[dict[str, Any]]:
    raw_pairs = body.get("pairs") if isinstance(body, Mapping) else None
    if not isinstance(raw_pairs, list):
        return []
    pairs_by_mint: dict[str, list[dict[str, Any]]] = {}
    for pair in raw_pairs:
        if not isinstance(pair, dict) or pair.get("chainId") != "solana":
            continue
        base_token = pair.get("baseToken")
        mint = base_token.get("address") if isinstance(base_token, Mapping) else None
        if isinstance(mint, str):
            pairs_by_mint.setdefault(mint, []).append(pair)

    query_key = query.casefold()
    results: list[dict[str, Any]] = []
    for mint, pairs in pairs_by_mint.items():
        main = select_main_pool(pairs)
        if main is None:
            continue
        base_token = main.get("baseToken")
        base_token = base_token if isinstance(base_token, Mapping) else {}
        info = main.get("info")
        info = info if isinstance(info, Mapping) else {}
        symbol = _optional_string(base_token.get("symbol"))
        results.append(
            {
                "mint": mint,
                "symbol": symbol,
                "name": _optional_string(base_token.get("name")),
                "image": _optional_string(info.get("imageUrl")),
                "pool_count": len(pairs),
                "total_liquidity_usd": sum(_pair_liquidity(pair) for pair in pairs),
                "volume_24h_usd": sum(_pair_volume_24h(pair) for pair in pairs),
                "main_pool_address": _optional_string(main.get("pairAddress")),
                "price_usd": _optional_float(main.get("priceUsd")),
                "symbol_match": symbol is not None and symbol.casefold() == query_key,
            }
        )
    return sorted(
        results,
        key=lambda item: float(item["total_liquidity_usd"]),
        reverse=True,
    )[:10]


def parse_geckoterminal_holders(body: Any) -> dict[str, Any] | None:
    data = body.get("data") if isinstance(body, Mapping) else None
    attributes = data.get("attributes") if isinstance(data, Mapping) else None
    holders = attributes.get("holders") if isinstance(attributes, Mapping) else None
    if not isinstance(holders, Mapping):
        return None
    distribution = holders.get("distribution_percentage")
    top_10 = (
        _optional_float(distribution.get("top_10")) if isinstance(distribution, Mapping) else None
    )
    if top_10 is None:
        return None
    holder_count = holders.get("count")
    if isinstance(holder_count, bool) or not isinstance(holder_count, int):
        holder_count = None
    as_of = holders.get("last_updated")
    return {
        "pct": top_10,
        "holder_count": holder_count,
        "as_of": as_of if isinstance(as_of, str) else None,
        "pool_accounts_excluded": False,
        "holders": [],
    }


def select_main_pool(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pairs:
        return None
    return max(pairs, key=_pair_liquidity)


def classify_pool_type(dex_id: str, labels: Any) -> PoolType:
    normalized_dex = dex_id.casefold()
    normalized_labels = (
        {str(label).strip().casefold() for label in labels if str(label).strip()}
        if isinstance(labels, list)
        else set()
    )
    for known_dex, required_labels, pool_type in POOL_TYPE_RULES:
        labels_match = required_labels.issubset(normalized_labels)
        if not required_labels and normalized_labels:
            labels_match = False
        if known_dex == normalized_dex and labels_match:
            return pool_type
    return "unknown"


def _iso_from_milliseconds(value: Any) -> str | None:
    milliseconds = _optional_float(value)
    if milliseconds is None:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def market_value(
    pairs: list[dict[str, Any]],
    *,
    geckoterminal_liquidity_usd: float | None = None,
    pools_checked_at: str | None = None,
) -> dict[str, Any] | None:
    main = select_main_pool(pairs)
    if main is None:
        return None
    liquidity = main.get("liquidity")
    volume = main.get("volume")
    price_change = main.get("priceChange")
    return {
        "price_usd": _optional_float(main.get("priceUsd")),
        "liquidity_usd": _optional_float(
            liquidity.get("usd") if isinstance(liquidity, Mapping) else None
        ),
        "volume_24h_usd": _optional_float(
            volume.get("h24") if isinstance(volume, Mapping) else None
        ),
        "price_change_24h_pct": _optional_float(
            price_change.get("h24") if isinstance(price_change, Mapping) else None
        ),
        "pool_created_at": _iso_from_milliseconds(main.get("pairCreatedAt")),
        "pool_count": len(pairs),
        "total_liquidity_usd": sum(_pair_liquidity(pair) for pair in pairs),
        "total_volume_24h_usd": sum(_pair_volume_24h(pair) for pair in pairs),
        "pools_checked_at": pools_checked_at,
        "main_pool": {
            "dex": str(main.get("dexId") or "unknown"),
            "address": str(main.get("pairAddress") or ""),
            "labels": main.get("labels") if isinstance(main.get("labels"), list) else [],
        },
        "geckoterminal_liquidity_usd": geckoterminal_liquidity_usd,
    }


class SolanaTokenClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        rate_limits: Mapping[str, float] | None = None,
        gecko_bucket: GeckoTerminalTokenBucket | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        limits = dict(SOURCE_RATE_LIMITS)
        if rate_limits is not None:
            limits.update(rate_limits)
        self._limiters = {
            source: AsyncRateLimiter(interval)
            for source, interval in limits.items()
            if source != "geckoterminal"
        }
        self._gecko_bucket = gecko_bucket or GeckoTerminalTokenBucket()
        self.sources = {source: _source_status(source) for source in SOURCE_NAMES}
        self._request_id = 0

    async def rpc(self, method: str, params: list[Any], *, largest: bool = False) -> Any:
        limiter_name = "solana_largest" if largest else "solana_rpc"
        await self._limiters[limiter_name].wait()
        self._request_id += 1
        request_id = self._request_id
        try:
            response = await self.client.post(
                self.settings.solana_rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
            )
        except httpx.HTTPError as error:
            raise self._request_error(
                "solana_rpc",
                self.settings.solana_rpc_url,
                None,
            ) from error
        if response.status_code >= 400:
            raise self._request_error(
                "solana_rpc",
                self.settings.solana_rpc_url,
                response.status_code,
                rate_limited=largest and response.status_code == 429,
            )
        try:
            body = response.json()
        except ValueError as error:
            raise self._request_error(
                "solana_rpc", self.settings.solana_rpc_url, response.status_code
            ) from error
        if not isinstance(body, dict) or body.get("error") is not None:
            raise self._request_error(
                "solana_rpc", self.settings.solana_rpc_url, response.status_code
            )
        self._mark_success("solana_rpc")
        return body.get("result")

    async def get_mint_account(self, mint: str) -> Mapping[str, Any] | None:
        result = await self.rpc(
            "getAccountInfo",
            [mint, {"encoding": "jsonParsed"}],
        )
        value = result.get("value") if isinstance(result, dict) else None
        return value if isinstance(value, Mapping) else None

    async def get_metadata_account(self, mint: str) -> Mapping[str, Any] | None:
        address = str(metadata_pda(mint))
        result = await self.rpc(
            "getAccountInfo",
            [address, {"encoding": "base64"}],
        )
        value = result.get("value") if isinstance(result, dict) else None
        return value if isinstance(value, Mapping) else None

    async def get_json(
        self,
        source: FactSource,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        priority: Literal["background", "interactive"] = "background",
    ) -> Any:
        try:
            if source == "geckoterminal":
                await self._gecko_bucket.wait(priority=priority)
            else:
                await self._limiters[source].wait()
        except SourceError as error:
            self._mark_failure(source, error.detail)
            raise
        try:
            response = await self.client.get(url, params=params, headers=headers)
        except httpx.HTTPError as error:
            raise self._request_error(source, url, None) from error
        if response.status_code >= 400:
            if source == "geckoterminal" and response.status_code == 429:
                self._gecko_bucket.record_rate_limit(response.headers.get("Retry-After"))
            raise self._request_error(source, url, response.status_code)
        try:
            body = response.json()
        except ValueError as error:
            raise self._request_error(source, url, response.status_code) from error
        if source == "geckoterminal":
            self._gecko_bucket.record_success()
        self._mark_success(source)
        return body

    async def get_dex_pairs(self, mints: list[str]) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(mints), 30):
            batch = mints[start : start + 30]
            if not batch:
                continue
            body = await self.get_json(
                "dexscreener",
                f"{DEXSCREENER_TOKENS_URL}/" + ",".join(batch),
            )
            for mint in batch:
                results[mint] = parse_dex_pairs(body, mint)
        return results

    async def get_token_pairs(self, mint: str) -> list[dict[str, Any]]:
        body = await self.get_json(
            "dexscreener",
            f"{DEXSCREENER_TOKEN_PAIRS_URL}/{mint}",
        )
        return parse_dex_pairs(body, mint)

    async def search_tokens(self, query: str) -> list[dict[str, Any]]:
        body = await self.get_json(
            "dexscreener",
            DEXSCREENER_SEARCH_URL,
            params={"q": query},
        )
        return parse_dex_search(body, query)

    async def get_geckoterminal_ohlcv(
        self,
        pool_address: str,
        mint: str,
        timeframe: str,
        aggregate: int,
        limit: int,
        *,
        priority: Literal["background", "interactive"] = "background",
    ) -> Any:
        return await self.get_json(
            "geckoterminal",
            f"{GECKOTERMINAL_POOLS_URL}/{pool_address}/ohlcv/{timeframe}",
            params={
                "aggregate": aggregate,
                "limit": limit,
                "currency": "usd",
                "token": mint,
            },
            priority=priority,
        )

    async def gecko_liquidity(
        self,
        pool_address: str,
        *,
        priority: Literal["background", "interactive"] = "background",
    ) -> float | None:
        body = await self.get_json(
            "geckoterminal",
            f"{GECKOTERMINAL_POOLS_URL}/{pool_address}",
            priority=priority,
        )
        data = body.get("data") if isinstance(body, dict) else None
        attributes = data.get("attributes") if isinstance(data, dict) else None
        return _optional_float(
            attributes.get("reserve_in_usd") if isinstance(attributes, dict) else None
        )

    async def market_and_lock(
        self,
        mint: str,
        *,
        gecko_cross_check: bool,
        priority: Literal["background", "interactive"] = "background",
    ) -> tuple[Fact, Fact, list[dict[str, Any]], str | None, str | None]:
        try:
            pairs = await self.get_token_pairs(mint)
        except SourceError as error:
            unavailable = unavailable_fact("dexscreener", error.detail, fetched_at=_iso_now())
            return unavailable, unavailable, [], None, None
        pools_checked_at = _iso_now()
        main = select_main_pool(pairs)
        if main is None:
            unavailable = unavailable_fact(
                "dexscreener",
                "No trading pool found yet",
                fetched_at=_iso_now(),
            )
            return unavailable, unavailable, [], None, None
        pool_address = str(main.get("pairAddress") or "")
        gecko_liquidity: float | None = None
        if gecko_cross_check and pool_address:
            try:
                gecko_liquidity = await self.gecko_liquidity(
                    pool_address,
                    priority=priority,
                )
            except SourceError:
                gecko_liquidity = None
        value = market_value(
            pairs,
            geckoterminal_liquidity_usd=gecko_liquidity,
            pools_checked_at=pools_checked_at,
        )
        base = main.get("baseToken")
        name = base.get("name") if isinstance(base, Mapping) else None
        symbol = base.get("symbol") if isinstance(base, Mapping) else None
        market_fact = Fact(
            status="ok",
            value=value,
            source="dexscreener",
            fetched_at=_iso_now(),
        )
        lock_fact = await self.liquidity_lock(main)
        return (
            market_fact,
            lock_fact,
            pairs,
            name if isinstance(name, str) else None,
            symbol if isinstance(symbol, str) else None,
        )

    async def liquidity_lock(self, main_pool: dict[str, Any] | None) -> Fact:
        fetched_at = _iso_now()
        if main_pool is None:
            return unavailable_fact(
                "dexscreener",
                "No trading pool found yet",
                fetched_at=fetched_at,
            )
        dex = str(main_pool.get("dexId") or "unknown")
        pool = str(main_pool.get("pairAddress") or "")
        labels = main_pool.get("labels")
        pool_type = classify_pool_type(dex, labels)
        value: dict[str, Any] = {
            "pool_type": pool_type,
            "dex": dex,
            "pool": pool,
            "burned_pct": None,
        }
        if pool_type == "launch_curve":
            return Fact(
                status="not_applicable",
                value=value,
                source="dexscreener",
                fetched_at=fetched_at,
                detail=(
                    "Still on the launch curve; liquidity is held by the launch program, not a pool"
                ),
            )
        if pool_type == "position_based":
            return Fact(
                status="not_applicable",
                value=value,
                source="dexscreener",
                fetched_at=fetched_at,
                detail="Position-based pool; there's no pool token to lock or burn",
            )
        if pool_type == "unknown":
            return Fact(
                status="unavailable",
                value=value,
                source="dexscreener",
                fetched_at=fetched_at,
                detail="Not checked by us yet for this pool type",
            )
        try:
            body = await self.get_json(
                "raydium",
                RAYDIUM_POOL_INFO_URL,
                params={"ids": pool},
            )
        except SourceError as error:
            return unavailable_fact("raydium", error.detail, fetched_at=_iso_now())
        data = body.get("data") if isinstance(body, dict) else None
        entry = data[0] if isinstance(data, list) and data else None
        burned_pct = _optional_float(entry.get("burnPercent")) if isinstance(entry, dict) else None
        if burned_pct is None:
            return unavailable_fact(
                "raydium",
                "Raydium did not return a burn percentage for this pool",
                fetched_at=_iso_now(),
            )
        value["burned_pct"] = burned_pct
        return Fact(
            status="ok",
            value=value,
            source="raydium",
            fetched_at=_iso_now(),
        )

    async def top10_share(
        self,
        mint: str,
        pool_addresses: set[str],
        *,
        priority: Literal["background", "interactive"] = "background",
    ) -> Fact:
        if self.settings.solana_rpc_url == DEFAULT_SOLANA_RPC_URL:
            return await self.geckoterminal_top10_share(mint, priority=priority)
        try:
            largest_result = await self.rpc(
                "getTokenLargestAccounts",
                [mint],
                largest=True,
            )
            supply_result = await self.rpc("getTokenSupply", [mint])
            largest = largest_result.get("value") if isinstance(largest_result, dict) else None
            supply_info = supply_result.get("value") if isinstance(supply_result, dict) else None
            if not isinstance(largest, list) or not isinstance(supply_info, dict):
                raise SourceError("solana_rpc", "Couldn't check right now")
            supply = _decimal(
                supply_info.get("uiAmountString")
                if supply_info.get("uiAmountString") is not None
                else supply_info.get("uiAmount")
            )
            if supply is None or supply <= 0:
                raise SourceError("solana_rpc", "Token supply is unavailable")
            account_addresses = [
                str(item["address"])
                for item in largest
                if isinstance(item, dict) and item.get("address") is not None
            ][:20]
            account_values = await self.rpc(
                "getMultipleAccounts",
                [account_addresses, {"encoding": "jsonParsed"}],
            )
            account_list = account_values.get("value") if isinstance(account_values, dict) else None
            if not isinstance(account_list, list) or len(account_list) != len(account_addresses):
                raise SourceError("solana_rpc", "Couldn't resolve token account owners")
            owner_by_account: dict[str, str] = {}
            for address, account in zip(account_addresses, account_list, strict=True):
                if not isinstance(account, dict):
                    raise SourceError("solana_rpc", "Couldn't resolve token account owners")
                data = account.get("data")
                parsed = data.get("parsed") if isinstance(data, dict) else None
                info = parsed.get("info") if isinstance(parsed, dict) else None
                owner = info.get("owner") if isinstance(info, dict) else None
                if not isinstance(owner, str):
                    raise SourceError("solana_rpc", "Couldn't resolve token account owners")
                owner_by_account[address] = owner

            holder_rows: list[dict[str, Any]] = []
            for item in largest:
                if not isinstance(item, dict) or item.get("address") not in owner_by_account:
                    continue
                amount = _decimal(
                    item.get("uiAmountString")
                    if item.get("uiAmountString") is not None
                    else item.get("uiAmount")
                )
                if amount is None:
                    raise SourceError("solana_rpc", "Couldn't read token account balances")
                owner = owner_by_account[str(item["address"])]
                holder_rows.append(
                    {
                        "owner": owner,
                        "pct": float(amount / supply * Decimal(100)),
                        "is_pool": owner in pool_addresses or owner == RAYDIUM_AMM_V4_AUTHORITY,
                    }
                )
            non_pool = [holder for holder in holder_rows if not holder["is_pool"]][:10]
            value = {
                "pct": sum(float(holder["pct"]) for holder in non_pool),
                "pool_accounts_excluded": True,
                "holder_count": None,
                "as_of": None,
                "holders": holder_rows,
            }
            return Fact(status="ok", value=value, source="solana_rpc", fetched_at=_iso_now())
        except SourceError:
            return await self.geckoterminal_top10_share(mint, priority=priority)

    async def geckoterminal_top10_share(
        self,
        mint: str,
        *,
        priority: Literal["background", "interactive"] = "background",
    ) -> Fact:
        try:
            body = await self.get_json(
                "geckoterminal",
                f"{GECKOTERMINAL_TOKEN_INFO_URL}/{mint}/info",
                priority=priority,
            )
        except SourceError as error:
            return unavailable_fact(
                "geckoterminal",
                error.detail,
                fetched_at=_iso_now(),
            )
        fetched_at = _iso_now()
        value = parse_geckoterminal_holders(body)
        if value is None:
            return unavailable_fact(
                "geckoterminal",
                "No holder data from GeckoTerminal yet",
                fetched_at=fetched_at,
            )
        return Fact(
            status="ok",
            value=value,
            source="geckoterminal",
            fetched_at=fetched_at,
            detail="Top 10 holders per GeckoTerminal; may include pool and exchange accounts",
        )

    async def rugcheck(self, mint: str) -> SecondOpinion:
        headers = (
            {"X-API-KEY": self.settings.rugcheck_api_key}
            if self.settings.rugcheck_api_key
            else None
        )
        try:
            body = await self.get_json(
                "rugcheck",
                RUGCHECK_SUMMARY_URL.format(mint=mint),
                headers=headers,
            )
        except SourceError:
            return SecondOpinion(
                status="unavailable",
                fetched_at=_iso_now(),
                score_normalised=None,
                lp_locked_pct=None,
                risks=[],
            )
        if not isinstance(body, dict) or not body:
            self._mark_failure("rugcheck", "RugCheck returned no report")
            return SecondOpinion(
                status="unavailable",
                fetched_at=_iso_now(),
                score_normalised=None,
                lp_locked_pct=None,
                risks=[],
            )
        risks: list[dict[str, str]] = []
        raw_risks = body.get("risks")
        if isinstance(raw_risks, list):
            for risk in raw_risks:
                if isinstance(risk, dict):
                    risks.append(
                        {
                            "name": str(risk.get("name") or ""),
                            "level": str(risk.get("level") or ""),
                            "description": str(risk.get("description") or ""),
                        }
                    )
        score = body.get("score_normalised")
        return SecondOpinion(
            status="ok",
            fetched_at=_iso_now(),
            score_normalised=int(score) if isinstance(score, (int, float)) else None,
            lp_locked_pct=_optional_float(body.get("lpLockedPct")),
            risks=risks,
        )

    def _request_error(
        self,
        source: FactSource,
        url: str,
        status_code: int | None,
        *,
        rate_limited: bool = False,
    ) -> SourceError:
        host = urlsplit(url).hostname or "unknown"
        logger.warning(
            "token source request failed source=%s host=%s status=%s",
            source,
            host,
            status_code,
        )
        if source == "solana_rpc" and rate_limited:
            detail = "Couldn't check right now (Solana connection rate-limited)"
        elif status_code == 429:
            detail = "Couldn't check right now (rate limited)"
        else:
            detail = "Couldn't check right now"
        self._mark_failure(source, detail)
        return SourceError(source, detail)

    def _mark_success(self, source: FactSource) -> None:
        self.sources[source] = {
            "name": source,
            "fetched_at": _iso_now(),
            "ok": True,
            "error": None,
        }

    def _mark_failure(self, source: FactSource, detail: str) -> None:
        self.sources[source] = {
            "name": source,
            "fetched_at": _iso_now(),
            "ok": False,
            "error": detail,
        }


@dataclass(frozen=True)
class ParsedMint:
    token_program: Literal["spl-token", "token-2022"]
    mint_authority: str | None
    freeze_authority: str | None
    risky_extensions: list[str]
    transfer_fee_bps: int | None
    token_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class DiscoveredToken:
    mint: str
    name: str | None
    symbol: str | None


def validate_mint(mint: str) -> Pubkey:
    try:
        return Pubkey.from_string(mint)
    except (TypeError, ValueError) as error:
        raise ValueError("Mint must be a valid 32-byte base58 Solana public key.") from error


def metadata_pda(mint: str | Pubkey) -> Pubkey:
    mint_pubkey = validate_mint(mint) if isinstance(mint, str) else mint
    program = Pubkey.from_string(METADATA_PROGRAM)
    address, _ = Pubkey.find_program_address(
        [b"metadata", bytes(program), bytes(mint_pubkey)],
        program,
    )
    return address


def parse_token_mint(account: Mapping[str, Any] | None) -> ParsedMint:
    if account is None:
        raise NotTokenMint("Address is not a token mint.")
    owner = account.get("owner")
    if owner == SPL_TOKEN_PROGRAM:
        token_program: Literal["spl-token", "token-2022"] = "spl-token"
    elif owner == TOKEN_2022_PROGRAM:
        token_program = "token-2022"
    else:
        raise NotTokenMint("Address is not owned by the SPL Token or Token-2022 program.")

    data = account.get("data")
    parsed = data.get("parsed") if isinstance(data, Mapping) else None
    if isinstance(parsed, Mapping) and parsed.get("type") not in {None, "mint"}:
        raise NotTokenMint("Address does not contain parsed token mint data.")
    info = parsed.get("info") if isinstance(parsed, Mapping) else None
    if not isinstance(info, Mapping):
        raise NotTokenMint("Address does not contain parsed token mint data.")

    extensions = info.get("extensions", [])
    extension_states: dict[str, Mapping[str, Any]] = {}
    if isinstance(extensions, list):
        for extension in extensions:
            if not isinstance(extension, Mapping):
                continue
            name = extension.get("extension")
            state = extension.get("state", {})
            if isinstance(name, str):
                extension_states[name] = state if isinstance(state, Mapping) else {}

    risky_extensions = [
        extension for extension in RISKY_TOKEN_EXTENSIONS if extension in extension_states
    ]
    fee_config = extension_states.get("transferFeeConfig", {})
    newer_fee = fee_config.get("newerTransferFee", {})
    transfer_fee_bps_value = (
        newer_fee.get("transferFeeBasisPoints") if isinstance(newer_fee, Mapping) else None
    )
    transfer_fee_bps = int(transfer_fee_bps_value) if transfer_fee_bps_value is not None else None

    token_metadata: dict[str, Any] | None = None
    metadata_extension = extension_states.get("tokenMetadata")
    if metadata_extension is not None:
        token_metadata = {
            "update_authority": metadata_extension.get("updateAuthority"),
            "name": metadata_extension.get("name"),
            "symbol": metadata_extension.get("symbol"),
            "uri": metadata_extension.get("uri"),
        }

    return ParsedMint(
        token_program=token_program,
        mint_authority=_optional_string(info.get("mintAuthority")),
        freeze_authority=_optional_string(info.get("freezeAuthority")),
        risky_extensions=risky_extensions,
        transfer_fee_bps=transfer_fee_bps,
        token_metadata=token_metadata,
    )


def parse_metaplex_metadata(data: bytes) -> dict[str, Any]:
    offset = 0
    if len(data) < 65:
        raise ValueError("Metaplex metadata account is truncated.")
    key = data[offset]
    offset += 1
    update_authority = str(Pubkey.from_bytes(data[offset : offset + 32]))
    offset += 32
    mint = str(Pubkey.from_bytes(data[offset : offset + 32]))
    offset += 32
    name, offset = _read_borsh_string(data, offset)
    symbol, offset = _read_borsh_string(data, offset)
    uri, offset = _read_borsh_string(data, offset)
    if offset + 3 > len(data):
        raise ValueError("Metaplex metadata account is truncated.")
    seller_fee_basis_points = struct.unpack_from("<H", data, offset)[0]
    offset += 2
    creators_tag = data[offset]
    offset += 1
    if creators_tag == 0:
        creators: list[dict[str, Any]] | None = None
    elif creators_tag == 1:
        if offset + 4 > len(data):
            raise ValueError("Metaplex creators field is truncated.")
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        creators = []
        for _ in range(count):
            if offset + 34 > len(data):
                raise ValueError("Metaplex creator entry is truncated.")
            creators.append(
                {
                    "address": str(Pubkey.from_bytes(data[offset : offset + 32])),
                    "verified": bool(data[offset + 32]),
                    "share": data[offset + 33],
                }
            )
            offset += 34
    else:
        raise ValueError("Metaplex creators option has an invalid tag.")
    if offset + 2 > len(data):
        raise ValueError("Metaplex metadata account is truncated.")
    primary_sale_happened = bool(data[offset])
    is_mutable = bool(data[offset + 1])
    return {
        "key": key,
        "update_authority": update_authority,
        "mint": mint,
        "name": name.rstrip("\x00"),
        "symbol": symbol.rstrip("\x00"),
        "uri": uri.rstrip("\x00"),
        "seller_fee_basis_points": seller_fee_basis_points,
        "creators": creators,
        "primary_sale_happened": primary_sale_happened,
        "is_mutable": is_mutable,
    }


def decode_account_data(value: Any) -> bytes:
    if isinstance(value, list) and value and isinstance(value[0], str):
        value = value[0]
    if not isinstance(value, str):
        raise ValueError("Metadata account did not return base64 data.")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError("Metadata account returned invalid base64 data.") from error


def _read_borsh_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(data):
        raise ValueError("Metaplex string field is truncated.")
    size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    end = offset + size
    if end > len(data):
        raise ValueError("Metaplex string field is truncated.")
    return data[offset:end].decode("utf-8"), end


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_gecko_new_pool_tokens(body: Any) -> list[DiscoveredToken]:
    if not isinstance(body, dict):
        return []
    included_by_id: dict[str, Mapping[str, Any]] = {}
    included = body.get("included")
    if isinstance(included, list):
        for item in included:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                continue
            attributes = item.get("attributes")
            if isinstance(attributes, Mapping):
                included_by_id[str(item["id"])] = attributes

    tokens: OrderedDict[str, DiscoveredToken] = OrderedDict()
    pools = body.get("data")
    if not isinstance(pools, list):
        return []
    for pool in pools:
        if not isinstance(pool, Mapping):
            continue
        relationships = pool.get("relationships")
        base_token = relationships.get("base_token") if isinstance(relationships, Mapping) else None
        token_data = base_token.get("data") if isinstance(base_token, Mapping) else None
        token_id = token_data.get("id") if isinstance(token_data, Mapping) else None
        if not isinstance(token_id, str) or not token_id.startswith("solana_"):
            continue
        mint = token_id.removeprefix("solana_")
        try:
            validate_mint(mint)
        except ValueError:
            continue
        attributes = included_by_id.get(token_id, {})
        symbol = _optional_string(attributes.get("symbol"))
        if symbol is not None and symbol.upper() in {"SOL", "USDC", "USDT"}:
            continue
        if mint not in tokens:
            tokens[mint] = DiscoveredToken(
                mint=mint,
                name=_optional_string(attributes.get("name")),
                symbol=symbol,
            )
    return list(tokens.values())


def _pending_card(mint: str, name: str | None, symbol: str | None, seen_at: str) -> TokenCard:
    return TokenCard(
        mint=mint,
        name=name,
        symbol=symbol,
        first_seen_at=seen_at,
        checked_at=seen_at,
        facts=TokenFacts(
            mint_authority=unavailable_fact("solana_rpc", "Check pending"),
            freeze_authority=unavailable_fact("solana_rpc", "Check pending"),
            token_extensions=unavailable_fact("solana_rpc", "Check pending"),
            metadata_mutable=unavailable_fact("solana_rpc", "Check pending"),
            top10_share=unavailable_fact("solana_rpc", "Check pending"),
            liquidity_lock=unavailable_fact("dexscreener", "Check pending"),
            market=unavailable_fact("dexscreener", "Check pending"),
        ),
        second_opinion=SecondOpinion(status="unavailable", risks=[]),
    )


def _first_seen(card: TokenCard | None, fallback: str) -> str:
    return card.first_seen_at if card is not None else fallback


def _safe_image_url(pairs: list[dict[str, Any]]) -> str | None:
    main = select_main_pool(pairs)
    info = main.get("info") if isinstance(main, dict) else None
    image_url = info.get("imageUrl") if isinstance(info, dict) else None
    if not isinstance(image_url, str):
        return None
    parsed = urlsplit(image_url)
    return image_url if parsed.scheme == "https" and parsed.hostname else None


class SolanaTokenEngine:
    def __init__(self, checker: SolanaTokenClient) -> None:
        self.checker = checker
        self.tokens: OrderedDict[str, TokenCard] = OrderedDict()
        self.updated_at: str | None = None
        self.ready = False
        self._checked_at: dict[str, float | None] = {}
        self._pool_addresses: dict[str, set[str]] = {}
        self._top_retry_count: dict[str, int] = {}
        self._top_retry_at: dict[str, float] = {}
        self._rug_retry_at: dict[str, float] = {}
        self._rug_retry_attempted: set[str] = set()
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._queued: set[tuple[str, str]] = set()

    def response(self, limit: int) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "warming",
            "updated_at": self.updated_at,
            "sources": [dict(self.checker.sources[source]) for source in SOURCE_NAMES],
            "tokens": [
                card.model_dump(mode="json") for card in list(self.tokens.values())[::-1][:limit]
            ],
        }

    async def discover_once(self) -> list[str]:
        discovered: list[str] = []
        try:
            body = await self.checker.get_json(
                "geckoterminal",
                GECKOTERMINAL_NEW_POOLS_URL,
                params={"page": 1, "include": "base_token"},
            )
            for token in parse_gecko_new_pool_tokens(body):
                if token.mint in self.tokens:
                    continue
                seen_at = _iso_now()
                self.tokens[token.mint] = _pending_card(
                    token.mint,
                    token.name,
                    token.symbol,
                    seen_at,
                )
                self._checked_at[token.mint] = None
                self._queue_work("card", token.mint)
                discovered.append(token.mint)
            while len(self.tokens) > 200:
                expired_mint, _ = self.tokens.popitem(last=False)
                self._checked_at.pop(expired_mint, None)
                self._pool_addresses.pop(expired_mint, None)
                self._top_retry_count.pop(expired_mint, None)
                self._top_retry_at.pop(expired_mint, None)
                self._rug_retry_at.pop(expired_mint, None)
                self._rug_retry_attempted.discard(expired_mint)
            self.checker._mark_success("geckoterminal")
        except SourceError:
            pass
        return discovered

    async def refresh_markets_once(self) -> None:
        mints = list(self.tokens)
        for start in range(0, len(mints), 30):
            batch = mints[start : start + 30]
            try:
                pairs_by_mint = await self.checker.get_dex_pairs(batch)
            except SourceError as error:
                fetched_at = _iso_now()
                for mint in batch:
                    self._update_facts(
                        mint,
                        market=unavailable_fact("dexscreener", error.detail, fetched_at=fetched_at),
                        liquidity_lock=unavailable_fact(
                            "dexscreener", error.detail, fetched_at=fetched_at
                        ),
                    )
                continue
            for mint in batch:
                pairs = pairs_by_mint.get(mint, [])
                old_card = self.tokens.get(mint)
                if old_card is None:
                    continue
                old_market = old_card.facts.market.value
                old_market_value = old_market if isinstance(old_market, dict) else {}
                main = select_main_pool(pairs)
                previous_main_value = old_market_value.get("main_pool")
                previous_main = previous_main_value if isinstance(previous_main_value, dict) else {}
                previous_pool = previous_main.get("address")
                current_pool = main.get("pairAddress") if isinstance(main, dict) else None
                previous_gecko_liquidity = (
                    old_market_value.get("geckoterminal_liquidity_usd")
                    if previous_pool == current_pool
                    else None
                )
                market = market_value(
                    pairs,
                    geckoterminal_liquidity_usd=previous_gecko_liquidity,
                )
                if market is None:
                    market_fact = unavailable_fact(
                        "dexscreener",
                        "No trading pool found yet",
                        fetched_at=_iso_now(),
                    )
                    lock_fact = unavailable_fact(
                        "dexscreener",
                        "No trading pool found yet",
                        fetched_at=_iso_now(),
                    )
                else:
                    for field in (
                        "pool_count",
                        "total_liquidity_usd",
                        "total_volume_24h_usd",
                        "pools_checked_at",
                    ):
                        market[field] = old_market_value.get(field)
                    market_fact = Fact(
                        status="ok",
                        value=market,
                        source="dexscreener",
                        fetched_at=_iso_now(),
                    )
                    if (
                        previous_pool == current_pool
                        and old_card.facts.liquidity_lock.status != "unavailable"
                    ):
                        lock_fact = old_card.facts.liquidity_lock
                    else:
                        lock_fact = await self.checker.liquidity_lock(main)
                    self._pool_addresses[mint] = self._pool_addresses.get(mint, set()) | {
                        str(pair["pairAddress"])
                        for pair in pairs
                        if isinstance(pair.get("pairAddress"), str)
                    }
                self._update_facts(mint, market=market_fact, liquidity_lock=lock_fact)

    def schedule_retries(self, *, max_top_holder_retries: int = 10) -> None:
        now = monotonic()
        scheduled_top_retries = sum(kind == "top10" for kind, _ in self._queued)
        for mint, card in self.tokens.items():
            top_fact = card.facts.top10_share
            retry_at = self._top_retry_at.get(mint)
            retry_due = (
                self._top_retry_count.get(mint, 0) == 0 if retry_at is None else now >= retry_at
            )
            if (
                top_fact.status == "unavailable"
                and top_fact.detail != "Address is not a token mint."
                and retry_due
                and self._top_retry_count.get(mint, 0) < MAX_TOP_HOLDER_ATTEMPTS
                and scheduled_top_retries < max_top_holder_retries
                and ("top10", mint) not in self._queued
            ):
                self._queue_work("top10", mint)
                scheduled_top_retries += 1
            retry_at = self._rug_retry_at.get(mint)
            if retry_at is not None and now >= retry_at and mint not in self._rug_retry_attempted:
                self._queue_work("rugcheck", mint)

    async def worker(self) -> None:
        while True:
            kind, mint = await self._queue.get()
            try:
                if kind == "card":
                    existing = self.tokens.get(mint)
                    if existing is None:
                        continue
                    checked_at = self._checked_at.get(mint)
                    if checked_at is not None and monotonic() - checked_at < 60:
                        continue
                    try:
                        card = await self.build_card(
                            mint,
                            first_seen_at=existing.first_seen_at,
                            gecko_cross_check=False,
                            fetch_top10=False,
                        )
                    except NotTokenMint:
                        self._update_all_unavailable(mint, "Address is not a token mint.")
                        continue
                    self._store_card(card, checked=True)
                    if card.second_opinion.status == "unavailable":
                        self._rug_retry_at[mint] = monotonic() + 600
                    else:
                        self._rug_retry_at.pop(mint, None)
                elif kind == "top10":
                    stored_card = self.tokens.get(mint)
                    if (
                        stored_card is None
                        or self._checked_at.get(mint) is None
                        or stored_card.facts.top10_share.status != "unavailable"
                    ):
                        continue
                    fact = await self.checker.top10_share(
                        mint,
                        self._pool_addresses.get(mint, set()),
                    )
                    if _is_geckoterminal_rate_limit(fact):
                        self._top_retry_at[mint] = monotonic() + TOP_HOLDER_RATE_LIMIT_RETRY_SECONDS
                    else:
                        attempts = self._top_retry_count.get(mint, 0) + 1
                        self._top_retry_count[mint] = attempts
                        if fact.status == "unavailable" and attempts < MAX_TOP_HOLDER_ATTEMPTS:
                            self._top_retry_at[mint] = monotonic() + TOP_HOLDER_RETRY_SECONDS
                        else:
                            self._top_retry_at.pop(mint, None)
                            if fact.status == "ok":
                                self._top_retry_count.pop(mint, None)
                    self._update_facts(mint, top10_share=fact)
                elif kind == "rugcheck":
                    stored_card = self.tokens.get(mint)
                    if (
                        stored_card is None
                        or stored_card.second_opinion.status != "unavailable"
                        or mint in self._rug_retry_attempted
                    ):
                        continue
                    self._rug_retry_attempted.add(mint)
                    opinion = await self.checker.rugcheck(mint)
                    updated_card = self.tokens.get(mint)
                    if updated_card is not None:
                        self.tokens[mint] = updated_card.model_copy(
                            update={"second_opinion": opinion}
                        )
                    self._rug_retry_at.pop(mint, None)
            except Exception as error:
                logger.warning("could not refresh Solana token facts mint=%s", mint)
                if kind == "card":
                    self._update_all_unavailable(mint, "Couldn't check right now")
                elif kind == "top10":
                    rate_limited = (
                        isinstance(error, SourceError)
                        and error.source == "geckoterminal"
                        and "rate limited" in error.detail.casefold()
                    )
                    if rate_limited:
                        self._top_retry_at[mint] = monotonic() + TOP_HOLDER_RATE_LIMIT_RETRY_SECONDS
                    else:
                        attempts = self._top_retry_count.get(mint, 0) + 1
                        self._top_retry_count[mint] = attempts
                        if attempts < MAX_TOP_HOLDER_ATTEMPTS:
                            self._top_retry_at[mint] = monotonic() + TOP_HOLDER_RETRY_SECONDS
                        else:
                            self._top_retry_at.pop(mint, None)
                    self._update_facts(
                        mint,
                        top10_share=unavailable_fact(
                            "geckoterminal",
                            error.detail
                            if isinstance(error, SourceError)
                            else "Couldn't check right now",
                            fetched_at=_iso_now(),
                        ),
                    )
                elif kind == "rugcheck":
                    stored_card = self.tokens.get(mint)
                    if stored_card is not None:
                        self.tokens[mint] = stored_card.model_copy(
                            update={
                                "second_opinion": SecondOpinion(
                                    status="unavailable",
                                    fetched_at=_iso_now(),
                                    risks=[],
                                )
                            }
                        )
                    self._rug_retry_at.pop(mint, None)
            finally:
                self._queued.discard((kind, mint))
                self._queue.task_done()

    async def build_card(
        self,
        mint: str,
        *,
        first_seen_at: str | None = None,
        gecko_cross_check: bool = True,
        fetch_top10: bool = True,
        fetch_rugcheck: bool = True,
        geckoterminal_priority: Literal["background", "interactive"] = "background",
    ) -> TokenCard:
        validate_mint(mint)
        now = _iso_now()
        existing = self.tokens.get(mint)
        first_seen = _first_seen(existing, first_seen_at or now)
        facts = {
            "mint_authority": unavailable_fact("solana_rpc", "Couldn't check right now"),
            "freeze_authority": unavailable_fact("solana_rpc", "Couldn't check right now"),
            "token_extensions": unavailable_fact("solana_rpc", "Couldn't check right now"),
            "metadata_mutable": unavailable_fact("solana_rpc", "Couldn't check right now"),
            "top10_share": unavailable_fact("solana_rpc", "Couldn't check right now"),
            "liquidity_lock": unavailable_fact("dexscreener", "Couldn't check right now"),
            "market": unavailable_fact("dexscreener", "Couldn't check right now"),
        }
        parsed_mint: ParsedMint | None
        try:
            account = await self.checker.get_mint_account(mint)
            parsed_mint = parse_token_mint(account)
        except SourceError:
            parsed_mint = None
        if parsed_mint is not None:
            rpc_fetched_at = self.checker.sources["solana_rpc"].get("fetched_at") or _iso_now()
            facts["mint_authority"] = Fact(
                status="ok",
                value=parsed_mint.mint_authority,
                source="solana_rpc",
                fetched_at=rpc_fetched_at,
                detail=("Creator can't mint more" if parsed_mint.mint_authority is None else None),
            )
            facts["freeze_authority"] = Fact(
                status="ok",
                value=parsed_mint.freeze_authority,
                source="solana_rpc",
                fetched_at=rpc_fetched_at,
                detail=(
                    "Creator can't freeze token accounts"
                    if parsed_mint.freeze_authority is None
                    else None
                ),
            )
            facts["token_extensions"] = Fact(
                status="ok",
                value={
                    "risky": parsed_mint.risky_extensions,
                    "transfer_fee_bps": parsed_mint.transfer_fee_bps,
                },
                source="solana_rpc",
                fetched_at=rpc_fetched_at,
            )

        metadata: dict[str, Any] | None = None
        if parsed_mint is not None and parsed_mint.token_metadata is not None:
            metadata = parsed_mint.token_metadata
            mutable = metadata.get("update_authority") is not None
            facts["metadata_mutable"] = Fact(
                status="ok",
                value=mutable,
                source="solana_rpc",
                fetched_at=self.checker.sources["solana_rpc"].get("fetched_at") or _iso_now(),
            )
        elif parsed_mint is not None:
            try:
                metadata_account = await self.checker.get_metadata_account(mint)
                if metadata_account is None:
                    facts["metadata_mutable"] = unavailable_fact(
                        "solana_rpc",
                        "No metadata account",
                        fetched_at=_iso_now(),
                    )
                else:
                    metadata = parse_metaplex_metadata(
                        decode_account_data(metadata_account.get("data"))
                    )
                    facts["metadata_mutable"] = Fact(
                        status="ok",
                        value=bool(metadata["is_mutable"]),
                        source="solana_rpc",
                        fetched_at=self.checker.sources["solana_rpc"].get("fetched_at")
                        or _iso_now(),
                    )
            except SourceError as error:
                facts["metadata_mutable"] = unavailable_fact(
                    "solana_rpc",
                    error.detail,
                    fetched_at=_iso_now(),
                )
            except (ValueError, TypeError, KeyError):
                facts["metadata_mutable"] = unavailable_fact(
                    "solana_rpc",
                    "Couldn't read metadata account",
                    fetched_at=_iso_now(),
                )

        market_fact, lock_fact, pairs, dex_name, dex_symbol = await self.checker.market_and_lock(
            mint,
            gecko_cross_check=gecko_cross_check,
            priority=geckoterminal_priority,
        )
        facts["market"] = market_fact
        facts["liquidity_lock"] = lock_fact
        pool_addresses = {
            str(pair["pairAddress"]) for pair in pairs if isinstance(pair.get("pairAddress"), str)
        }
        if existing is not None:
            self._pool_addresses[mint] = pool_addresses
        if fetch_top10:
            facts["top10_share"] = await self.checker.top10_share(
                mint,
                pool_addresses,
                priority=geckoterminal_priority,
            )
        else:
            facts["top10_share"] = unavailable_fact("geckoterminal", "Check pending")
        opinion = (
            await self.checker.rugcheck(mint)
            if fetch_rugcheck or existing is None
            else existing.second_opinion
        )
        if parsed_mint is not None and metadata is None:
            name = dex_name
            symbol = dex_symbol
        else:
            name = _optional_string(metadata.get("name")) if metadata else dex_name
            symbol = _optional_string(metadata.get("symbol")) if metadata else dex_symbol
        if name is None and existing is not None:
            name = existing.name
        if symbol is None and existing is not None:
            symbol = existing.symbol
        card = TokenCard(
            mint=mint,
            name=name,
            symbol=symbol,
            image_url=_safe_image_url(pairs),
            token_program=parsed_mint.token_program if parsed_mint is not None else None,
            first_seen_at=first_seen,
            checked_at=_iso_now(),
            facts=TokenFacts(**facts),
            second_opinion=opinion,
        )
        return card

    async def get_card(self, mint: str, *, max_age_seconds: float = 60.0) -> TokenCard:
        validate_mint(mint)
        checked_at = self._checked_at.get(mint)
        if (
            mint in self.tokens
            and checked_at is not None
            and monotonic() - checked_at < max_age_seconds
        ):
            return self.tokens[mint]
        existing = self.tokens.get(mint)
        if existing is not None and ("rugcheck", mint) in self._queued:
            return existing
        fetch_rugcheck = (
            existing is None
            or existing.second_opinion.status == "ok"
            or self._checked_at.get(mint) is None
        )
        retry_at = self._rug_retry_at.get(mint)
        retry_due = (
            retry_at is not None
            and monotonic() >= retry_at
            and mint not in self._rug_retry_attempted
        )
        if retry_due:
            fetch_rugcheck = True
            self._rug_retry_attempted.add(mint)
        card = await self.build_card(
            mint,
            first_seen_at=existing.first_seen_at if existing is not None else None,
            gecko_cross_check=True,
            fetch_rugcheck=fetch_rugcheck,
            geckoterminal_priority="interactive",
        )
        if existing is not None:
            self._store_card(card, checked=True)
        return card

    def _store_card(self, card: TokenCard, *, checked: bool) -> None:
        if card.mint in self.tokens:
            self.tokens[card.mint] = card
        if checked:
            self._checked_at[card.mint] = monotonic()
        top_fact = card.facts.top10_share
        if top_fact.status == "ok":
            self._top_retry_count.pop(card.mint, None)
            self._top_retry_at.pop(card.mint, None)
        elif _is_geckoterminal_rate_limit(top_fact):
            self._top_retry_at[card.mint] = monotonic() + TOP_HOLDER_RATE_LIMIT_RETRY_SECONDS
        elif top_fact.detail != "Check pending":
            self._top_retry_at.setdefault(card.mint, monotonic() + TOP_HOLDER_RETRY_SECONDS)
        if (
            card.second_opinion.status == "unavailable"
            and card.mint not in self._rug_retry_attempted
        ):
            self._rug_retry_at.setdefault(card.mint, monotonic() + 600)
        elif card.second_opinion.status == "ok":
            self._rug_retry_at.pop(card.mint, None)

    def _update_facts(self, mint: str, **updates: Fact) -> None:
        card = self.tokens.get(mint)
        if card is None:
            return
        self.tokens[mint] = card.model_copy(update={"facts": card.facts.model_copy(update=updates)})

    def _update_all_unavailable(self, mint: str, detail: str) -> None:
        updates = {
            fact_name: unavailable_fact(
                "solana_rpc" if fact_name not in {"market", "liquidity_lock"} else "dexscreener",
                detail,
                fetched_at=_iso_now(),
            )
            for fact_name in FACT_NAMES
        }
        self._update_facts(mint, **updates)

    def _queue_work(self, kind: str, mint: str) -> None:
        work = (kind, mint)
        if work in self._queued:
            return
        self._queued.add(work)
        self._queue.put_nowait(work)
