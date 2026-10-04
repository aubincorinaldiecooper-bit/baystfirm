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

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BAYST_DATABASE_PATH` | `var/baystfirm.db` | SQLite event and classification store |
| `BAYST_VENUES` | `coinbase,kraken,bybit,okx` | Public stream adapters to run |
| `BAYST_SYMBOLS` | Majors, stablecoins, and perpetuals | Canonical instruments; each venue subscribes to the ones it lists |
| `BAYST_SHADOW_MODE` | `true` | Prevent unvalidated classifications from becoming trusted |
| `BAYST_API_KEY` | _(empty)_ | When set, every `/v1` route requires `Authorization: Bearer <key>` (`/health` stays open) |

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
