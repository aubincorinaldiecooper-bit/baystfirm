# Architecture

## Scope

Baystfirm is an intelligence-only crypto market service. It does not custody assets, connect
wallets, submit orders, or generate open-ended investment recommendations.

The first vertical slice covers public spot and perpetual trade streams, with stablecoin peg health
as a first-class classifier. The same event contract is intended to accept order books, funding,
liquidations, on-chain observations, DeFi state, reserve attestations, and tokenized assets later.

## Data plane

```text
Coinbase spot ─┐
Kraken spot ───┼─ venue adapters ─ normalized MarketEvent ─ SQLite/WAL
Bybit perps ───┘                              │
                                             ├─ realtime WebSocket
                                             └─ bounded classifiers
                                                        │
                                                        ├─ shadow Classification
                                                        └─ replay/evaluation
```

Adapters contain source-specific subscription and parsing logic only. They do not classify the
market. Every event preserves the source venue, native symbol, exchange time, receive time,
sequence or trade identifier, and a hash of the source payload.

SQLite is intentionally used for the local proof so replay and evaluation remain reproducible.
Production scale will require a durable event bus and analytical time-series store, but the event
and classifier contracts should remain unchanged.

## Intelligence plane

The initial classifiers are deliberately bounded:

- `stablecoin_peg`: requires fresh observations from at least two venues and classifies pegged,
  watch, or depegged states;
- `short_horizon_momentum`: classifies a thirty-second trade window as upward, downward, or
  range-bound.

Both are versioned rule baselines, not trained models. They remain marked `uncalibrated` and
`shadow` until replay and live shadow evaluation pass the promotion gate.

## Evaluation plane

Evaluation accepts explicit prediction records with expected labels. It reports:

- sample count and non-abstained coverage;
- accuracy and false-alert rate;
- Brier score and expected calibration error;
- p95 classification latency.

No classifier becomes trusted merely because it runs. The default gate requires at least 500
examples, 90 percent accuracy, five percent maximum false alerts, 0.10 maximum calibration error,
and 250 millisecond p95 latency. These are initial engineering thresholds and should be tightened
per classifier after real datasets exist.

The required release progression is:

```text
offline replay -> historical evaluation -> live shadow -> promotion gate -> user-visible signal
```

Any classifier revision returns to shadow mode.

## Trust boundaries

- Public market streams are untrusted input and must be schema-validated.
- Source timestamps and receive timestamps remain distinct.
- A missing or single-venue stablecoin observation causes abstention.
- The API currently has no authentication and is suitable only for local development.
- Production requires authenticated clients, encrypted transport, rate limits, durable streaming,
  dead-letter handling, observability, and source-specific terms review.
