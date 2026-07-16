# A7 First-Candle Coin-Toss Break — Fixed-% RR 1:1.5, 5-Min IS Screen

STATUS: DONE — NO EDGE

EXP-A7 of research/81, proposed by user 2026-07-16: trade the SAME stocks
every day; first 5-min candle's range break in EITHER direction (whichever
breaks first = the day's trade), fixed-% stop on STOCK PRICE (not OR size),
target = 1.5× stop (RR 1:1.5). Portfolio-of-stocks; optimize stage-by-stage
only if the base shows an edge.

## Locked mechanics

- Universe (stage 1): NIFTY + 9 deep F&O names; IS 2015-02-01→2021-09-30.
  Breadth stage only if G1 gate approached.
- Signal: after bar 1 (09:15–09:20), the FIRST 5-min close above bar-1 high
  → long, or below bar-1 low → short — whichever occurs first; one
  trade/symbol/day. Entry next bar open.
- Stop = signal close × (1 − s) long / (1 + s) short; target = 1.5×s
  mirrored. s ∈ {0.5%, 0.75%, 1.0%}.
- Time-stop ts ∈ {1 (same-session close), 2} — user intent reads intraday,
  but study meta-finding (multi-day drift) earns the ts=2 check.
- Costs futures-proxy 3 bps. Long/short legs reported separately (study-wide
  finding: shorts bleed — this tests whether 1:1.5 asymmetry rescues them).

## Grid (LOCKED): s {0.5,0.75,1.0}% × ts {1,2} = 6 cells (ledger +6)

Gate: standard G1 (pooled net t ≥ 3, ≥55% syms positive, coherent in s).
Breakeven context: RR 1:1.5 coin-toss needs >40% win rate + cost cover.
Falsification: all cells net-negative → coin-toss first-candle = NO EDGE;
no re-grids without new experiment ID.

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-16 ~18:10 IST | Pre-registered | runner: `scripts/run_a7_cointoss.py` |

## Findings

(after run)

## VERDICT (2026-07-16 ~18:20 IST): NO EDGE

35k trades: gross is a literal coin toss (-1.4..+3.2 bps); net -9..-13 bps
with t -5..-19; 0-20% symbols positive; win 39-45% vs ~43% breakeven at
RR 1:1.5 net. No s%, no leg, no hold-length works — nothing to optimize.
First-candle direction carries no information; the lead system wins on the
exact opposite choices (30-60min range, gap-up filter, long-only, multi-day
hold). Pre-registered falsification met. STATUS: DONE
