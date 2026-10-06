# Baystfirm

Baystfirm is the backend foundation for a crypto-native realtime intelligence terminal. It
normalizes public market streams, preserves event provenance, replays historical observations, and
evaluates bounded market-state classifiers before they are allowed to surface as trusted signals.

The first release is intelligence-only. It does not connect wallets or execute trades.

## Product contract

Every classification must include:

- a bounded observable state;
- a calibrated probability and fixed horizon;
- source evidence and freshness;
- explicit abstention when evidence is insufficient;
- shadow-mode evaluation before user-visible promotion.

Stablecoins are first-class instruments and signals. The service will monitor their price, peg
health, liquidity, cross-venue flow, and surrounding spot, perpetual, DeFi, and tokenized-asset
markets.

## Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/baystfirm serve --reload
```

CI runs these same checks on every PR (.github/workflows/ci.yml).

The API is served at `http://127.0.0.1:8000`. The realtime normalized stream is available over
WebSocket at `/v1/stream` and as server-sent events at `/v1/stream/sse` (optional
`?symbols=USDC-USD,BTC-USD`; `market_event` and `classification` events, a keepalive comment every
15 seconds). `/v1/snapshot` returns the latest event per venue and instrument and the latest
classification per classifier and instrument, so a client can render before the stream delivers.

Replay captured events through the shadow classifiers without sleeping:

```bash
.venv/bin/baystfirm replay --database var/baystfirm.db --speed 0
```

Replay stored events, label every prediction with the state observed one horizon later, score
each classifier against the promotion gate, and record the run (served at `/v1/evaluation/gate`):

```bash
.venv/bin/baystfirm evaluate-replay --database var/baystfirm.db
```

Evaluate externally labeled newline-delimited prediction records:

```bash
.venv/bin/baystfirm evaluate evaluation-records.jsonl
```

The promotion gate also requires at least 0.5 macro recall across labels present in the expected outcomes; abstentions count as misses.

Import official Binance spot aggregate trades into the event database, timestamped CSV ticks, or
both. Repeat `--date` for each UTC archive day; at least one output is required:

```bash
.venv/bin/baystfirm import-archive --symbol BTCUSDT \
  --date 2025-09-01 --date 2025-09-02 \
  --database var/baystfirm.db --ticks-dir var/ticks
```

The importer verifies Binance's published SHA-256 checksum before reading each cached or downloaded
archive. Chart-state predictions use `gnsis_chart_momentum` and are graded against realized
`short_horizon_momentum` outcomes; all learned outputs remain shadow and uncalibrated until
validated.

## Indicators and strategy backtests

Request up to six talipp indicators with repeated `indicator` parameters. Supported forms are
`sma:<period>`, `ema:<period>`, `rsi:<period>`, `macd:<fast>,<slow>,<signal>`,
`bb:<period>,<mult>`, `atr:<period>`, and `stoch:<period>,<smoothing>`. Periods are 2–200,
Bollinger multipliers are 0.5–5, and MACD fast must be less than slow. Indicator arrays align with
the returned candles; `null` values at the beginning are expected during warm-up:

```text
/v1/candles?venue=coinbase&symbol=BTC-USD&interval=1h&indicator=sma%3A20&indicator=rsi%3A14
```

`POST /v1/backtest` evaluates your rule against historical closed candles. It is a deterministic,
intelligence-only simulation and does not execute trades. Price operands can use `open`, `high`,
`low`, `close`, or `volume`. Conditions can be combined at the rule level, or placed in one-level
`all`/`any` groups of 2–5 conditions. If `after_bars` is omitted for a rule with a take-profit or
stop-loss level, the backtester exits at the close 500 bars after entry when neither level has
been hit:

```json
{
  "rule": {
    "name": "Example rule",
    "venue": "coinbase",
    "symbol": "BTC-USD",
    "interval": "1h",
    "conditions": [
      {
        "combine": "all",
        "conditions": [
          {
            "left": {"kind": "price", "field": "volume"},
            "op": "above",
            "right": {"kind": "value", "value": 100}
          },
          {
            "left": {"kind": "price", "field": "close"},
            "op": "above",
            "right": {"kind": "indicator", "spec": "bb:20,2", "output": "middle"}
          }
        ]
      }
    ],
    "expect": "up",
    "exit": {"after_bars": 24}
  },
  "bars": 1000,
  "fee_bps": 10,
  "slippage_bps": 5,
  "holdout_pct": 20
}
```

The response retains trade statistics, completed/open trade records, and baseline results. It also
reports the echoed `costs`, an `equity` curve with total return and maximum drawdown, `buy_and_hold`
return/drawdown (always long, regardless of `expect`), UTC `by_year` statistics, and `holdout`
in-sample/recent statistics. Set `holdout_pct` to `0` to omit the holdout result.

`POST /v1/backtest/batch` evaluates the same rule on its `venue`/`symbol` and on 1–19 additional
instruments in `also`, using the same bars and cost settings:

```json
{
  "rule": {
    "name": "Example rule",
    "venue": "coinbase",
    "symbol": "BTC-USD",
    "interval": "1h",
    "conditions": [
      {
        "left": {"kind": "price", "field": "close"},
        "op": "above",
        "right": {"kind": "value", "value": 0}
      }
    ],
    "expect": "up",
    "exit": {"after_bars": 24}
  },
  "bars": 1000,
  "fee_bps": 10,
  "slippage_bps": 5,
  "holdout_pct": 20,
  "also": [{"venue": "coinbase", "symbol": "ETH-USD"}]
}
```

Batch results stay in request order—the rule's instrument first—and are not ranked by performance.
The summary emphasizes breadth, including how many instruments had trades and beat the baseline.
Coins can move together, so instruments are not independent tests; judge how broadly your rule
holds up rather than selecting the best result. An invalid or unavailable instrument is reported
as an error for that result while the rest of the batch continues.

## Momentum regime replay track record

Authenticated `GET /v1/track-record/backtest` replays the `momentum_regime` classifier on the last
seven days of closed 1-minute candles, using one eligible exchange per configured instrument. It
initially returns `computing`, then reports pooled horizon groups and each source; the background
replay refreshes every six hours. Live classifications use trades merged from all venues, and
overlapping predictions are scored individually, so replay results can differ and intervals are
optimistic.

## Solana token discovery

Authenticated `GET /v1/solana/tokens/new?limit=50` lists the latest Solana tokens discovered from
GeckoTerminal; `limit` is 1–200. `GET /v1/solana/tokens/{mint}` returns an individual token card.
`GET /v1/solana/search?q=...` searches DEX Screener and returns up to 10 grouped Solana tokens;
results describe pools found by the search and do not imply endorsement. The token page can use
`GET /v1/solana/tokens/{mint}/candles?interval=1h&limit=300` for USD GeckoTerminal candles at
`1m`, `5m`, `15m`, `1h`, `4h`, or `1d` intervals, with the same repeated indicator specs as
`/v1/candles`.
The feed refreshes every 60 seconds, keeps at most 200 tokens, and loads cards in the background.
Cards show sourced, timestamped facts from standard Solana JSON-RPC calls, DEX Screener,
GeckoTerminal, and the Raydium pool API. `top10_share` uses GeckoTerminal's token-info holder
distribution by default; it may include pool and exchange accounts, which GeckoTerminal does not
identify. A custom RPC URL tries account-level holder checks first and falls back to
GeckoTerminal. Feed card builds skip the optional GeckoTerminal liquidity cross-check; individual
detail cards retain it. RPC is limited to 4 requests/s (largest-account requests, when a custom RPC
is used, to 1 per 2 seconds); DEX Screener to 4/s; the keyless GeckoTerminal API is budgeted at
approximately 10 calls/minute with a shared token bucket. Background discovery and holder work
leave a three-token reserve for interactive detail and candle requests. GeckoTerminal 429 responses
pause shared callers with Retry-After or exponential backoff; interactive calls wait at most five
seconds for a token. Raydium is limited to 2/s and RugCheck to 1/s.
Market facts include the selected main pool's price/liquidity/volume plus all-pool count, total
liquidity, total 24-hour volume, and the time of the full pool read.

Our on-chain checks are presented first. RugCheck is explicitly a second opinion; Baystfirm does
not issue an overall safe/unsafe verdict and does not provide wallets or trade execution. The RPC
provider can be changed with `BAYST_SOLANA_RPC_URL`; its full URL is never logged or returned
because provider URLs may contain credentials. Set `BAYST_RUGCHECK_API_KEY` to send RugCheck's
optional `X-API-KEY` header. `BAYST_SOLANA_TOKENS=false` disables the background discovery loop.

## News and events

Authenticated `GET /v1/news` aggregates headline, source, publication time, and the original link
from the U.S. SEC, U.S. CFTC, Federal Reserve, and Bank of Canada press-release feeds. Each feed is
polled sequentially every 15 minutes with a 10-second timeout; individual feed failures are
reported without stopping the others. `GET /v1/news/filings?tickers=AAPL,GOOG` fetches recent SEC
EDGAR filings on demand. Official releases and filings link to their publisher; article bodies are
not copied or stored.

Baystfirm also reports measured events from its own data: liquidations of at least $250,000, or
rolling five-minute liquidation totals of at least $1,000,000, with 30-minute per-rule cooldowns;
stablecoin prices at least 0.5% from $1 across three or more trades in two minutes; and Solana
tokens first seen within 24 hours with at least $50,000 liquidity. A same-pool Solana liquidity drop
is reported when liquidity falls by at least half from $10,000 or more, at most once per token per
six hours. Stablecoin events re-arm after ten calm minutes within 0.2% of $1 and at least one hour
between alerts. Each stablecoin alert includes the median for every venue with at least three
recent trades, the cross-market median, and the number of venues deviating in the same direction.
These are measured facts, not investment advice or an overall safety verdict.

Set `BAYST_NEWS=false` to disable the official-feed poller. All news requests use
`BAYST_SEC_USER_AGENT`, which should identify the application and provide a contact email.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BAYST_DATABASE_PATH` | `var/baystfirm.db` | SQLite event and classification store |
| `BAYST_VENUES` | `coinbase,kraken,bybit,okx` | Public stream adapters to run |
| `BAYST_SYMBOLS` | Majors, stablecoins, and perpetuals | Canonical instruments; each venue subscribes to the ones it lists |
| `BAYST_SHADOW_MODE` | `true` | Prevent unvalidated classifications from becoming trusted |
| `BAYST_API_KEY` | _(empty)_ | When set, every `/v1` route requires `Authorization: Bearer <key>` (`/health` stays open) |
| `BAYST_NEWS` | `true` | Enable the official press-release feed poller |
| `BAYST_SEC_USER_AGENT` | `Baystfirm/0.1 aubincorinaldiecooper@gmail.com` | Declared User-Agent for news and SEC requests |
| `BAYST_SOLANA_RPC_URL` | `https://api.mainnet-beta.solana.com` | Swappable standard JSON-RPC endpoint; never logged or returned |
| `BAYST_RUGCHECK_API_KEY` | _(empty)_ | Optional RugCheck API key, sent as `X-API-KEY` |
| `BAYST_SOLANA_TOKENS` | `true` | Enable the background Solana token discovery loop |

## Implemented foundation

- normalized public trade adapters for Coinbase spot, Kraken spot, OKX spot, Bybit spot, and Bybit
  linear perpetuals;
- durable SQLite/WAL events with source provenance and measured receive latency;
- API access and a realtime WebSocket for events and classifications;
- deterministic replay through the same bounded classifiers used live;
- stablecoin peg (USDT, USDC, PYUSD, DAI, USDe, FDUSD, with stablecoin-quoted pairs converted to
  USD) and short-horizon momentum shadow classifiers, each emitting at most once per second per
  instrument;
- automatic outcome labeling from replayed history and stored per-classifier evaluation runs;
- promotion metrics and gates for accuracy, calibration, false alerts, coverage, and latency.

See `docs/ARCHITECTURE.md` for the trust boundaries and production gaps.
