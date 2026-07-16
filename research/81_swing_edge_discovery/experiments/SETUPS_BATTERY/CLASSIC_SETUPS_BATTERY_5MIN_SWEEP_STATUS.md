# Classic Setups Battery — O=H/O=L, CPR-Open, PDH/PWH Break, MA Cross (5-min)

STATUS: DONE (see Verdict)

Experiments **EXP-D2, EXP-C2, EXP-A5, EXP-A6** of research/81, requested by
user 2026-07-16 ("were these considered?"). Universe: NIFTY + 9 deep F&O
names. IS window 2015-02-01→2021-09-30 ONLY (Val/OOS untouched). Costs
futures-proxy 3 bps. Engine + conventions identical to prior experiments.
Discipline note: not "all possible combinations" — each setup gets a small
locked coarse grid; only survivors earn filter conditioning (F1→F2 pattern).
Breadth replication on the backfilled universe follows for survivors.

## EXP-D2 — Open=High / Open=Low day-bias break (6 cells)

After the first 6 bars (09:15–09:45): if session open ≈ running LOW (within
0.1%) → long bias; entry on next-bar-open after a 5-min close above the
6-bar high; stop = 6-bar low. O≈H mirror short. ts {1,2,4}.
G0: O=L means zero selling pressure from the open — initiative buyers;
counterparty = fade traders. Gate: standard G1.

## EXP-C2 — CPR opening-candle setups (6 cells)

CPR from prev session H/L/C: P=(H+L+C)/3, BC=(H+L)/2, TC=2P−BC.
Long: first 5-min candle GREEN and closes above TC → entry next bar,
stop = P. Short mirror below BC. ts {1,2,4}.
Filter axes RESERVED for a follow-up only if base signals: CPR width
narrow/wide (trailing 20th pctile), width vs previous day.
G0: open above value area + initiative candle = acceptance above value;
in-house prior research/67 (daily narrow→calm) informs the LATER filter, not
the base. Gate: standard G1.

## EXP-A5 — Previous day / week range break (8 cells)

PDH break: first 5-min close > prev session high → long next bar,
stop = PDH − 0.5×(PDH−PDL). PDL mirror short. PWH/PWL same with prev WEEK
range, stop offset 0.33×week-range. ts {2,4}.
G0: range edges = resting liquidity; acceptance beyond = continuation.
NB: A1 already killed N-DAY-range breaks on daily bars; this tests the
INTRADAY-timed version — different fill, tighter risk. Gate: standard G1.

## EXP-A6 — MA crossover entries, multi-TF (12 cells)

EMA fast×slow cross as entry (long on up-cross, short mirror), stop = slow
EMA at signal: 15-min EMA9/21, 60-min EMA9/21, daily EMA10/20. ts {2,4}.
G0: crossover = trend ignition proxy. In-house priors are NEGATIVE
(indicators hurt MQ; RSI-regime added nothing) — this is a fair burial or a
surprise. Gate: standard G1.

**Battery ledger: +32 cells (→ ~160 study total).**

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-16 ~07:50 IST | Pre-registered | runner: `scripts/run_setups_battery.py` |

## Findings

(after run)

## VERDICT (2026-07-16 ~08:20 IST)

- **EXP-D2 O=L break long: G1 PASS** — ts4 net +21.2bps t=3.20, 8/10 syms,
  6/7 yrs, monotone in hold. Robustness: 9-cell plateau ALL positive
  (t 2.3-3.3), tighter open-tolerance = stronger edge (dose-response);
  WF 79% half-years positive (worst -14bps). Overlap with gap-ORB lead only
  36%; DISJOINT trades still +14.9bps (t=1.85) -> genuinely additive trigger.
  O=H short mirror NEGATIVE (consistent long-bias family asymmetry).
- **EXP-C2 CPR-open long: SIGNAL** — ts4 +10.3bps t=2.37, 7/10 syms, 6/7 yrs;
  short side dead; ts1/ts2 weak -> multi-day drift again. Third (weakest)
  trigger of the morning-strength family; CPR-width filters deferred until
  breadth confirms.
- **EXP-A5 PDH/PWH break: NO EDGE** — longs +4bps t<1; ALL shorts strongly
  negative. Intraday-timed daily-range break does not rescue family A1.
- **EXP-A6 MA crossovers: NO EDGE** — in-house prior confirmed; daily-long
  gross +31bps but n=284 t=0.88; every intraday crossover cell bleeds.

ARCHITECTURE consequence: one MORNING-STRENGTH CONTINUATION system, long-only,
=<4-session hold, entry triggers {gap-ORB, O=L break, CPR-open}; union book
with per-symbol-per-day dedup. A4 breadth (post-backfill) extended to
replicate D2_OL + C2_CPR alongside ORB before the single OOS touch.
STATUS: DONE
