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
The feed refreshes every 60 seconds, keeps at most 200 tokens, and loads cards in the background.
Cards show sourced, timestamped facts from standard Solana JSON-RPC calls, DEX Screener,
GeckoTerminal liquidity cross-checks, and the Raydium pool API. RPC is limited to 4 requests/s
(largest-account requests to 1 per 2 seconds); DEX Screener to 4/s, GeckoTerminal to 1 per 2.5s,
Raydium to 2/s, and RugCheck to 1/s.

Our on-chain checks are presented first. RugCheck is explicitly a second opinion; Baystfirm does
not issue an overall safe/unsafe verdict and does not provide wallets or trade execution. The RPC
provider can be changed with `BAYST_SOLANA_RPC_URL`; its full URL is never logged or returned
because provider URLs may contain credentials. Set `BAYST_RUGCHECK_API_KEY` to send RugCheck's
optional `X-API-KEY` header. `BAYST_SOLANA_TOKENS=false` disables the background discovery loop.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BAYST_DATABASE_PATH` | `var/baystfirm.db` | SQLite event and classification store |
| `BAYST_VENUES` | `coinbase,kraken,bybit,okx` | Public stream adapters to run |
| `BAYST_SYMBOLS` | Majors, stablecoins, and perpetuals | Canonical instruments; each venue subscribes to the ones it lists |
| `BAYST_SHADOW_MODE` | `true` | Prevent unvalidated classifications from becoming trusted |
| `BAYST_API_KEY` | _(empty)_ | When set, every `/v1` route requires `Authorization: Bearer <key>` (`/health` stays open) |
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
