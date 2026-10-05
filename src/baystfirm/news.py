from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict, deque
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
from statistics import median
from time import monotonic, struct_time
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlsplit

import feedparser
import httpx
from pydantic import BaseModel, Field, field_validator

from baystfirm.models import EventType, MarketEvent

if TYPE_CHECKING:
    from baystfirm.storage import EventStore

logger = logging.getLogger(__name__)

NewsKind = Literal["official", "filing", "market_event", "token_event"]
NewsSource = Literal["sec", "cftc", "federal_reserve", "bank_of_canada", "sec_edgar", "baystfirm"]

NEWS_NOTE = (
    "Headlines link to the original publisher. Market and token events are measured by Baystfirm "
    "from exchange and on-chain data. Facts, not investment advice."
)
OFFICIAL_POLL_SECONDS = 15 * 60
OFFICIAL_RETENTION_DAYS = 30
SEC_THROTTLE_SECONDS = 0.2
SEC_TICKER_CACHE_SECONDS = 24 * 60 * 60
SEC_SUBMISSIONS_CACHE_SECONDS = 10 * 60
SEC_SUBMISSIONS_CACHE_MAX = 200
LARGE_LIQUIDATION_USD = 250_000.0
LIQUIDATION_BURST_USD = 1_000_000.0
LIQUIDATION_WINDOW_SECONDS = 5 * 60
LIQUIDATION_COOLDOWN_SECONDS = 30 * 60
STABLECOIN_PRICE_WINDOW_SECONDS = 2 * 60
STABLECOIN_OFF_PEG_THRESHOLD_PCT = 0.5
STABLECOIN_REARM_THRESHOLD_PCT = 0.2
STABLECOIN_REARM_SECONDS = 10 * 60
STABLECOIN_MINIMUM_REFIRE_SECONDS = 60 * 60
STABLECOINS = frozenset({"USDT", "USDC", "PYUSD", "DAI", "USDE", "FDUSD"})
OFFICIAL_FEEDS: tuple[tuple[NewsSource, str, str], ...] = (
    ("sec", "U.S. SEC", "https://www.sec.gov/news/pressreleases.rss"),
    ("cftc", "U.S. CFTC", "https://www.cftc.gov/RSS/RSSGP/rssgp.xml"),
    ("federal_reserve", "Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    (
        "bank_of_canada",
        "Bank of Canada",
        "https://www.bankofcanada.ca/content_type/press-releases/feed/",
    ),
)
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
FILING_FORMS = frozenset({"8-K", "10-Q", "10-K", "6-K", "20-F", "40-F"})
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")
SOLANA_MINT_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join(" ".join(parser.parts).split())[:300]


def stable_news_id(source: str, url: str) -> str:
    return sha256(f"{source}\0{url}".encode()).hexdigest()


def owned_news_id(rule: str, key: str, window_start: datetime | str) -> str:
    start = window_start.isoformat() if isinstance(window_start, datetime) else window_start
    return sha256(f"{rule}\0{key}\0{start}".encode()).hexdigest()


class NewsItem(BaseModel):
    id: str
    kind: NewsKind
    source: NewsSource
    source_label: str
    title: str
    url: str | None
    published_at: datetime
    symbols: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> str:
        return clean_title(value)

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class NewsRequestError(RuntimeError):
    pass


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _datetime_from_struct(value: object) -> datetime | None:
    if not isinstance(value, struct_time):
        return None
    try:
        return datetime(*value[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _utc_datetime(value)
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
    return _utc_datetime(parsed)


def _entry_published_at(entry: Mapping[str, Any]) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = _datetime_from_struct(entry.get(key))
        if parsed is not None:
            return parsed
    for key in ("dc_date", "date", "published", "updated"):
        parsed = parse_datetime(entry.get(key))
        if parsed is not None:
            return parsed
    return None


def _http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def parse_official_feed(
    source: NewsSource,
    content: bytes,
    *,
    fetched_at: datetime | None = None,
) -> list[NewsItem]:
    feed = feedparser.parse(content)
    if feed.get("bozo") and not feed.get("entries"):
        error = feed.get("bozo_exception")
        raise NewsRequestError(f"Could not parse feed: {error or 'invalid document'}")
    label = next((label for feed_source, label, _ in OFFICIAL_FEEDS if feed_source == source), "")
    now = _utc_datetime(fetched_at or datetime.now(UTC))
    items: list[NewsItem] = []
    for raw_entry in feed.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            continue
        title = clean_title(raw_entry.get("title"))
        url = _http_url(raw_entry.get("link"))
        if not title or url is None:
            continue
        published_at = _entry_published_at(raw_entry) or now
        items.append(
            NewsItem(
                id=stable_news_id(source, url),
                kind="official",
                source=source,
                source_label=label,
                title=title,
                url=url,
                published_at=published_at,
                symbols=[],
                details={},
            )
        )
    return items


def normalize_news_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    value = symbol.strip()
    if not value:
        return None
    return value if SOLANA_MINT_PATTERN.fullmatch(value) else value.upper()


class NewsService:
    def __init__(
        self,
        store: EventStore,
        user_agent: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.store = store
        self.user_agent = user_agent
        self._client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self._owns_client = client is None
        self._feed_sources: dict[str, dict[str, Any]] = {
            source: {
                "source": source,
                "label": label,
                "url": url,
                "last_success_at": None,
                "last_error": None,
            }
            for source, label, url in OFFICIAL_FEEDS
        }
        self._sec_source: dict[str, Any] = {
            "source": "sec_edgar",
            "label": "SEC EDGAR",
            "url": SEC_TICKERS_URL,
            "last_success_at": None,
            "last_error": None,
        }
        self._ticker_cache: tuple[float, dict[str, dict[str, Any]]] | None = None
        self._submissions_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._sec_throttle_lock = asyncio.Lock()
        self._next_sec_request_at = 0.0

    def official_sources(self) -> list[dict[str, Any]]:
        return [dict(self._feed_sources[source]) for source, _, _ in OFFICIAL_FEEDS]

    def sec_source(self) -> dict[str, Any]:
        return dict(self._sec_source)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def poll_official_once(self) -> None:
        for source, _, url in OFFICIAL_FEEDS:
            status = self._feed_sources[source]
            try:
                if (urlsplit(url).hostname or "").endswith("sec.gov"):
                    await self._wait_for_sec_request()
                response = await self._client.get(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": (
                            "application/rss+xml, application/atom+xml, application/xml, text/xml"
                        ),
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                items = parse_official_feed(source, response.content)
                for item in items:
                    await self.store.append_news(item)
                status["last_success_at"] = datetime.now(UTC).isoformat()
                status["last_error"] = None
            except Exception as error:
                status["last_error"] = str(error) or type(error).__name__
                logger.warning("News feed poll failed for %s: %s", source, error)
        await self.store.prune_news(datetime.now(UTC) - timedelta(days=OFFICIAL_RETENTION_DAYS))

    async def poll_official_forever(self) -> None:
        while True:
            await self.poll_official_once()
            await asyncio.sleep(OFFICIAL_POLL_SECONDS)

    async def filings_for_ticker(
        self, ticker: str, limit: int
    ) -> tuple[list[NewsItem], str | None]:
        normalized = ticker.strip().upper()
        try:
            companies = await self._company_tickers()
            company = companies.get(normalized)
            if company is None:
                return [], (
                    f"No SEC filer found for {normalized} "
                    "(non-US companies may not file with the SEC)."
                )
            cik_value = company.get("cik_str")
            if not isinstance(cik_value, int | str):
                raise NewsRequestError("SEC company record has an invalid CIK.")
            try:
                cik = int(cik_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise NewsRequestError("SEC company record has an invalid CIK.") from error
            submissions = await self._submissions(cik)
            company_name = str(
                submissions.get("name") or company.get("title") or normalized
            ).strip()
            filings = submissions.get("filings")
            recent = filings.get("recent") if isinstance(filings, Mapping) else None
            if not isinstance(recent, Mapping):
                raise NewsRequestError("SEC submissions response has no recent filing data.")
            forms = recent.get("form")
            accessions = recent.get("accessionNumber")
            filing_dates = recent.get("filingDate")
            primary_documents = recent.get("primaryDocument")
            if not (
                isinstance(forms, list)
                and isinstance(accessions, list)
                and isinstance(filing_dates, list)
                and isinstance(primary_documents, list)
            ):
                raise NewsRequestError("SEC submissions response has invalid filing arrays.")
            accepted_values = recent.get("acceptanceDateTime", [])
            descriptions = recent.get("primaryDocDescription", [])
            items_values = recent.get("items", [])
            if not isinstance(accepted_values, list):
                accepted_values = []
            if not isinstance(descriptions, list):
                descriptions = []
            if not isinstance(items_values, list):
                items_values = []
            items: list[NewsItem] = []
            for index, raw_form in enumerate(forms):
                if not isinstance(raw_form, str):
                    continue
                form = raw_form.strip()
                base_form = form.removesuffix("/A")
                if base_form not in FILING_FORMS:
                    continue
                accession = accessions[index] if index < len(accessions) else None
                document = primary_documents[index] if index < len(primary_documents) else None
                if (
                    not isinstance(accession, str)
                    or not isinstance(document, str)
                    or not document.strip()
                ):
                    continue
                accession_path = accession.replace("-", "")
                document_path = quote(document.strip(), safe="")
                url = f"{SEC_ARCHIVES_URL}/{cik}/{accession_path}/{document_path}"
                description = descriptions[index] if index < len(descriptions) else None
                title = f"{company_name}: Form {form}"
                if (
                    isinstance(description, str)
                    and description.strip()
                    and description.strip().casefold() != form.casefold()
                ):
                    title += f" ({description.strip()})"
                filing_items = items_values[index] if index < len(items_values) else None
                if base_form == "8-K" and isinstance(filing_items, str) and filing_items.strip():
                    title += f" · Items {filing_items.strip()}"
                accepted = accepted_values[index] if index < len(accepted_values) else None
                filing_date = filing_dates[index] if index < len(filing_dates) else None
                published_at = parse_datetime(accepted)
                if published_at is None:
                    published_at = parse_datetime(filing_date)
                    if published_at is not None:
                        published_at = published_at.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                if published_at is None:
                    continue
                items.append(
                    NewsItem(
                        id=stable_news_id("sec_edgar", url),
                        kind="filing",
                        source="sec_edgar",
                        source_label="SEC EDGAR",
                        title=title,
                        url=url,
                        published_at=published_at,
                        symbols=[normalized],
                        details={},
                    )
                )
            items.sort(key=lambda item: item.published_at, reverse=True)
            return items[:limit], None
        except Exception as error:
            note = f"SEC filings unavailable for {normalized}: {str(error) or type(error).__name__}"
            logger.warning("SEC filing lookup failed for %s: %s", normalized, error)
            return [], note

    async def _company_tickers(self) -> dict[str, dict[str, Any]]:
        now = monotonic()
        if self._ticker_cache is not None and self._ticker_cache[0] > now:
            return self._ticker_cache[1]
        payload = await self._get_sec_json(SEC_TICKERS_URL)
        values = payload.values() if isinstance(payload, Mapping) else ()
        companies: dict[str, dict[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping):
                continue
            ticker = value.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                companies[ticker.strip().upper()] = dict(value)
        self._ticker_cache = (now + SEC_TICKER_CACHE_SECONDS, companies)
        return companies

    async def _submissions(self, cik: int) -> dict[str, Any]:
        key = f"{cik:010d}"
        now = monotonic()
        cached = self._submissions_cache.get(key)
        if cached is not None and cached[0] > now:
            self._submissions_cache.move_to_end(key)
            return cached[1]
        payload = await self._get_sec_json(SEC_SUBMISSIONS_URL.format(cik=key))
        self._submissions_cache[key] = (now + SEC_SUBMISSIONS_CACHE_SECONDS, payload)
        self._submissions_cache.move_to_end(key)
        while len(self._submissions_cache) > SEC_SUBMISSIONS_CACHE_MAX:
            self._submissions_cache.popitem(last=False)
        return payload

    async def _get_sec_json(self, url: str) -> dict[str, Any]:
        await self._wait_for_sec_request()
        try:
            response = await self._client.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise NewsRequestError("SEC returned an invalid JSON response.")
        except (httpx.HTTPError, ValueError, NewsRequestError) as error:
            self._sec_source["last_error"] = str(error) or type(error).__name__
            raise NewsRequestError(self._sec_source["last_error"]) from error
        self._sec_source["last_success_at"] = datetime.now(UTC).isoformat()
        self._sec_source["last_error"] = None
        return payload

    async def _wait_for_sec_request(self) -> None:
        async with self._sec_throttle_lock:
            delay = self._next_sec_request_at - monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_sec_request_at = monotonic() + SEC_THROTTLE_SECONDS


def parse_tickers(value: str) -> list[str]:
    tickers = [item.strip() for item in value.split(",")]
    if not 1 <= len(tickers) <= 5 or any(not TICKER_PATTERN.fullmatch(item) for item in tickers):
        raise ValueError("Provide 1–5 comma-separated valid ticker symbols.")
    return [ticker.upper() for ticker in tickers]


def format_usd_amount(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        amount = f"{value / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
        return f"${amount}B"
    if absolute >= 1_000_000:
        amount = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"${amount}M"
    if absolute >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def _venue_label(venue: str) -> str:
    return {
        "coinbase": "Coinbase",
        "kraken": "Kraken",
        "bybit": "Bybit",
        "okx": "OKX",
        "binanceus": "Binance.US",
    }.get(venue.lower(), venue.replace("_", " ").title())


def _symbol_base_quote(symbol: str) -> tuple[str, str] | None:
    parts = [part.strip().upper() for part in symbol.split("-") if part.strip()]
    parts = [part for part in parts if part not in {"PERP", "SPOT"}]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


class MarketNewsObserver:
    def __init__(self) -> None:
        self._liquidations: dict[str, deque[tuple[datetime, float, str, str]]] = {}
        self._last_liquidation_emit: dict[tuple[str, str], datetime] = {}
        self._stablecoin_prices: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
        self._stablecoin_active: set[tuple[str, str]] = set()
        self._stablecoin_calm_since: dict[tuple[str, str], datetime] = {}
        self._stablecoin_last_fired: dict[tuple[str, str], datetime] = {}

    def observe(self, event: MarketEvent) -> list[NewsItem]:
        if event.event_type is EventType.LIQUIDATION:
            return self._observe_liquidation(event)
        if event.event_type is EventType.TRADE:
            item = self._observe_stablecoin_trade(event)
            return [item] if item is not None else []
        return []

    def _observe_liquidation(self, event: MarketEvent) -> list[NewsItem]:
        if event.price is None or event.size is None:
            return []
        base = event.base_asset.strip().upper()
        if not base:
            parsed = _symbol_base_quote(event.symbol)
            base = parsed[0] if parsed else ""
        if not base:
            return []
        multiplier = event.metadata.get("contract_multiplier")
        try:
            multiplier_value = float(multiplier) if multiplier is not None else 1.0
            notional = float(event.price) * float(event.size) * multiplier_value
        except (TypeError, ValueError, OverflowError):
            return []
        if not (notional > 0 and notional < float("inf")):
            return []
        observed_at = _utc_datetime(event.exchange_timestamp)
        venue = _venue_label(event.venue)
        items: list[NewsItem] = []
        if notional >= LARGE_LIQUIDATION_USD and self._allow_liquidation_emit(
            base,
            "large_liquidation",
            observed_at,
        ):
            item_key = f"{base}:{event.venue.lower()}:{event.symbol}"
            items.append(
                NewsItem(
                    id=owned_news_id("large_liquidation", item_key, observed_at),
                    kind="market_event",
                    source="baystfirm",
                    source_label="Baystfirm (measured)",
                    title=(
                        f"{format_usd_amount(notional)} {base} position liquidated on "
                        f"{venue} ({event.symbol})"
                    ),
                    url=None,
                    published_at=observed_at,
                    symbols=[base],
                    details={
                        "rule": "large_liquidation",
                        "notional_usd": notional,
                        "venue": venue,
                        "symbol": event.symbol,
                        "contract_multiplier": multiplier_value if multiplier is not None else None,
                        "window_seconds": LIQUIDATION_COOLDOWN_SECONDS,
                    },
                )
            )

        cutoff = observed_at - timedelta(seconds=LIQUIDATION_WINDOW_SECONDS)
        history = [
            record
            for record in self._liquidations.get(base, ())
            if cutoff <= record[0] <= observed_at
        ]
        history.append((observed_at, notional, venue, event.symbol))
        history.sort(key=lambda record: record[0])
        self._liquidations[base] = deque(history)
        total = sum(value for _, value, _, _ in history)
        if total >= LIQUIDATION_BURST_USD and self._allow_liquidation_emit(
            base,
            "liquidation_burst",
            observed_at,
        ):
            venues = sorted({item_venue for _, _, item_venue, _ in history})
            window_start = history[0][0]
            items.append(
                NewsItem(
                    id=owned_news_id("liquidation_burst", base, window_start),
                    kind="market_event",
                    source="baystfirm",
                    source_label="Baystfirm (measured)",
                    title=(
                        f"{format_usd_amount(total)} of {base} positions liquidated in 5 minutes "
                        f"({', '.join(venues)})"
                    ),
                    url=None,
                    published_at=observed_at,
                    symbols=[base],
                    details={
                        "rule": "liquidation_burst",
                        "notional_usd": total,
                        "venues": venues,
                        "window_seconds": LIQUIDATION_WINDOW_SECONDS,
                    },
                )
            )
        return items

    def _allow_liquidation_emit(self, base: str, rule: str, observed_at: datetime) -> bool:
        key = (base, rule)
        previous = self._last_liquidation_emit.get(key)
        if (
            previous is not None
            and (observed_at - previous).total_seconds() < LIQUIDATION_COOLDOWN_SECONDS
        ):
            return False
        self._last_liquidation_emit[key] = observed_at
        return True

    def _observe_stablecoin_trade(self, event: MarketEvent) -> NewsItem | None:
        pair = _symbol_base_quote(event.symbol)
        if (
            pair is None
            or event.price is None
            or not (event.price > 0 and event.price < float("inf"))
        ):
            return None
        base, quote = pair
        if base not in STABLECOINS or quote not in STABLECOINS | {"USD"}:
            return None
        observed_at = _utc_datetime(event.exchange_timestamp)
        key = (event.venue.lower(), event.symbol.upper())
        cutoff = observed_at - timedelta(seconds=STABLECOIN_PRICE_WINDOW_SECONDS)
        history = [
            sample
            for sample in self._stablecoin_prices.get(key, ())
            if cutoff <= sample[0] <= observed_at
        ]
        history.append((observed_at, float(event.price)))
        history.sort(key=lambda sample: sample[0])
        self._stablecoin_prices[key] = history
        median_price = median(price for _, price in history) if len(history) >= 3 else None
        active = key in self._stablecoin_active
        if active:
            if (
                median_price is not None
                and abs(median_price - 1.0) <= STABLECOIN_REARM_THRESHOLD_PCT / 100
            ):
                calm_since = self._stablecoin_calm_since.setdefault(key, observed_at)
                last_fired = self._stablecoin_last_fired[key]
                if (observed_at - calm_since).total_seconds() >= STABLECOIN_REARM_SECONDS and (
                    observed_at - last_fired
                ).total_seconds() >= STABLECOIN_MINIMUM_REFIRE_SECONDS:
                    self._stablecoin_active.discard(key)
                    self._stablecoin_calm_since.pop(key, None)
                    active = False
            else:
                self._stablecoin_calm_since.pop(key, None)
        if active or median_price is None:
            return None
        deviation_pct = (median_price - 1.0) * 100
        if abs(deviation_pct) < STABLECOIN_OFF_PEG_THRESHOLD_PCT:
            return None
        previous_fire_at = self._stablecoin_last_fired.get(key)
        if (
            previous_fire_at is not None
            and (observed_at - previous_fire_at).total_seconds() < STABLECOIN_MINIMUM_REFIRE_SECONDS
        ):
            return None
        self._stablecoin_active.add(key)
        self._stablecoin_last_fired[key] = observed_at
        self._stablecoin_calm_since.pop(key, None)
        direction = "above" if deviation_pct > 0 else "below"
        window_start = history[0][0]
        venue = _venue_label(event.venue)
        venue_readings: list[dict[str, Any]] = []
        for (venue_name, symbol_name), samples in self._stablecoin_prices.items():
            if symbol_name != key[1]:
                continue
            recent_prices = [
                price for sample_at, price in samples if cutoff <= sample_at <= observed_at
            ]
            if len(recent_prices) >= 3:
                venue_readings.append(
                    {
                        "venue": _venue_label(venue_name),
                        "median_price": median(recent_prices),
                        "trade_count": len(recent_prices),
                    }
                )
        venue_readings.sort(key=lambda reading: str(reading["venue"]).casefold())
        cross_market_median = median(float(reading["median_price"]) for reading in venue_readings)
        agreeing_count = sum(
            1
            for reading in venue_readings
            if abs((float(reading["median_price"]) - 1.0) * 100) >= STABLECOIN_OFF_PEG_THRESHOLD_PCT
            and (float(reading["median_price"]) - 1.0) * deviation_pct > 0
        )
        return NewsItem(
            id=owned_news_id("stablecoin_off_peg", f"{key[0]}:{key[1]}", window_start),
            kind="market_event",
            source="baystfirm",
            source_label="Baystfirm (measured)",
            title=(
                f"{base} trading {abs(deviation_pct):.2f}% {direction} 1 {quote} "
                f"on {venue} ({event.symbol}); "
                f"{agreeing_count}/{len(venue_readings)} venues off by ≥0.5%"
            ),
            url=None,
            published_at=observed_at,
            symbols=[base],
            details={
                "rule": "stablecoin_off_peg",
                "median_price": median_price,
                "trade_count": len(history),
                "deviation_pct": deviation_pct,
                "venue": venue,
                "symbol": event.symbol,
                "window_seconds": STABLECOIN_PRICE_WINDOW_SECONDS,
                "venue_readings": venue_readings,
                "cross_market_median": cross_market_median,
                "venue_count": len(venue_readings),
                "agreeing_count": agreeing_count,
            },
        )
