# research/109 — Intraday Stocks System Discovery: RESULTS

## Verdict: **NO EDGE — intraday cash single-name trading cannot clear its own costs. The >=20% CAGR intraday goal is NOT achievable from price/indicator signals in this data.**

Wave 1: 9 families x 26 cells, 150 names, 1.94M trades, 2015-2023, 10bps RT
costs, time-matched-slot controls, halves-stability gate.
- 24/26 cells net-negative (t -2..-18). Notable fake: VWFADE25_L +124bps IS
  driven entirely by 2018-21 COVID whipsaws (halves +15/+199), Val NEGATIVE.
- Sole structural survivor: **CPRTR_S** (narrow CPR + open below BC + red
  first 15m -> short to close): excess +9.3bps t=5.3 IS, halves 9.0/9.6,
  names+ 0.64, Val excess +5.5 t=2.4 — REAL information, but net ~0 (the
  edge is smaller than the cost floor).

Wave 2 (CPRTR_S refinement): width {0.2/0.3/0.5%} x exit {EOD, 30m-high
trail, 1% stop}: **every cell net-negative in Val (-2..-6.5bps)**; the trail
hurts (whipsaw), the stop is neutral. Exit engineering cannot manufacture
margin the signal does not have.

## Program-level conclusion (r/89 + r/109: ~44 intraday constructions)
Intraday equity signals in this market carry at most ~5-10bps of real
information per trade against ~10bps of unavoidable friction. The arithmetic
for 20% CAGR (+8bps/day net on full capital) is therefore out of reach for
single-name intraday cash. Where the house DOES have live, validated edges:
multi-day books (HA 2-green, momentum, breakout-trail; gap-ORB 4-day
revival candidate awaiting decision) and index-options structures (NAS).
Recommendation: route the >=20% CAGR ambition through those + the futures
leverage dial on validated curves — not through intraday stock trading.

Ledger: wave1 26 + wave2 9 = 35 cells (program total 456). OOS 2024+ never
touched — remains virgin for any future intraday family with a genuinely
different information source (order flow, events, cross-sectional intraday).
