# A2 Stock ORB → 2-4 Session Hold — 9 Deep-History F&O Names, 5-Min IS Screen

STATUS: PRE-REGISTERED

EXP-A2 of research/81. Family A's 5-min variant (daily variant A1 = NO EDGE).
Same mechanics as EXP-F1 but on single stocks: does the opening-range breakout
carry multi-day on liquid F&O STOCKS as it (weakly) does on the index?

## 1. The Base — locked

- **Universe:** HDFCBANK, ICICIBANK, RELIANCE, INFY, TCS, SBIN, ITC,
  HINDUNILVR, BHARTIARTL (deep 5-min, 2015/2018→present after backfill).
  KOTAKBANK EXCLUDED (known splice inconsistency until repaired).
- **Splits:** IS 2015-02-01→2021-09-30 (per-symbol from first bar if later),
  Val 2021-10→2023-12, OOS 2024+ locked. This screen touches IS only.
- **Signal/stop:** identical to F1 — OR = first W 5-min bars; first close
  beyond OR after window; entry next bar open; stop = opposite OR bound;
  no target. Costs FUTURES_PROXY 3 bps.
- **Time-stop:** 2 or 4 (ts=1 excluded on F1 evidence: intraday uniformly negative).

## 2. Plan — pre-registered grid (LOCKED)

W {6,12} × dir {L,S} × ts {2,4} = **8 cells** × 9 symbols (ledger +8 → 124).

**G1 gate:** pooled net t ≥ 3, ≥55% symbols positive, coherent across W.
**Falsification:** all cells net-negative → stock ORB swing = NO EDGE;
gap-conditioning (A3) only IF a positive base signal exists here.

## 3. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~22:30 IST | Pre-registered | runner: `scripts/run_a2_stock_orb.py` |

## 4. Findings

(after run)

## VERDICT (2026-07-15 ~22:40 IST): WEAK SIGNAL — same signature as F1

Longs +7.2/+7.6 bps net (t 1.5-1.6, gross +17), shorts strongly negative
(t -3.8..-5.4), ts4 > ts2, coherent across W — the ORB long multi-day drift
replicates on stocks but at ~1/3 the index's per-trade edge. Gate not met.
EXP-A3 (pre-registered here, ledger +2): gap_up>=0.25% conditioning on the
two long ts4 cells only — justified by F2's adopted filter; stock-level gap.
STATUS: DONE

## EXP-A3 RESULT (2026-07-15 ~22:45 IST): gap conditioning REPLICATES on stocks

W6_L_ts4 + stock gap_up>=0.25%: n=2074, gross +27.0bps, NET +17.4bps, t=2.87,
8/9 symbols positive. W12 variant: +15.5bps, t=2.33, 7/9. Same direction and
magnitude-lift as the index (F2) — cross-instrument replication of the
gap-up + ORB-long mechanism. Family verdict: STRONGEST candidate of the
study so far; G3 (walk-forward, MC, regime, param plateau; broad universe
post-backfill) queued next.
