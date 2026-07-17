# Follow-ups Battery — OR-Width Filter, Decay-Gate Forensic, B-Family 5-Min Timing

STATUS: DONE (see Verdicts)

Experiments **EXP-A8, EXP-H1, EXP-B3** of research/81 (post-conclusion
follow-ups, user-approved 2026-07-17). The ORB family's Val AND OOS are
CONSUMED — A8/H1 are therefore labeled exploratory/forensic: anything they
find validates only via paper-forward. B3 (family B) never touched Val/OOS,
so its normal gate pipeline remains available.

## EXP-A8 — OR-width quality filter on the locked ORB cell (exploratory)

Motivation: G5 forensics — narrow-OR trades lose, wide-OR win (equal-risk
sizing inverted the edge). Falsifiable prediction: net rises MONOTONICALLY
with OR-width bucket.
Mechanics: locked cell (gap≥0.25%, W12, long, ts4, stop=OR-low); OR-width =
(OR_high−OR_low)/day_open, bucketed by trailing-252-session causal percentile
into terciles. IS 2015-2021 only, F&O universe + NIFTY. **6 cells** (tercile ×
{stocks, index}). Gate: monotone + top-tercile net ≥ 1.5× unconditional.

## EXP-H1 — decay-gate forensic on the full ORB trade series (forensic)

Question: would a causal trade-level health gate have exited the family
before the 2025-26 decay? Full-period W12 gap25 trade series (2015→2026-07,
F&O universe; includes consumed-OOS window → FORENSIC label, no new claims).
Gates tested: trailing {6m, 12m} × metric {mean net > 0, t > 0}, evaluated
monthly, trade month m uses data ≤ m−1. **4 gate variants.** Report: full-
period gated vs ungated net, % of 2025-26 losses avoided, % of 2017-21 gains
kept, flip count (churn). Success = a gate that avoids ≥60% of 2025-26
bleed while keeping ≥70% of the good years — then propose as PAPER overlay.

## EXP-B3 — deep-z short-fade with 5-min entry timing (normal pipeline)

Family B (SIGNAL at daily, t=1.5, never Val/OOS-touched). Question: does
5-min entry timing lift the thin daily edge? Signal day t (causal): z20 ≥
+2.5 AND close < SMA200. Entry day t+1: (a) BASELINE next-open (=B2);
(b) CONFIRM: first 5-min close below the 09:15-09:45 low (skip day if never).
Stop 2.5×ATR14, target SMA20 (B-family locks), ts {2,4}. F&O universe,
IS 2015-2021, 3bp. **4 cells** + baseline. Gate: standard G1 vs the B2
baseline (t must beat baseline's on the same days).

**Ledger: +14 cells (A8 6, H1 4 forensic, B3 4).**

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-17 ~11:15 IST | Pre-registered | runner: `scripts/run_followups_battery.py` |

## Findings

(after run)

## VERDICTS (2026-07-17 ~11:40 IST)

- **EXP-A8 (OR-width filter): effect REAL, filter FAILS its gate.** Narrow
  tercile is the worst bucket in both sets (index +6.3 vs +42/+45 mid/wide;
  stocks +18.3 vs +27/+23) — confirms the G5 sizing forensic. But top-tercile
  lift = 1.03x unconditional (gate needed 1.5x) and stocks mid>wide breaks
  monotonicity. Keep as the SIZING lesson (never 1/stop-width), not a filter.
- **EXP-B3 (reversion 5-min timing): NO IMPROVEMENT.** n collapses to 144-235
  on the 5-min IS window; all cells within noise (best t=0.18). Family B stays
  a thin daily SIGNAL; the timing lever does not rescue it.
- **EXP-H1 (decay-gate forensic): NO WORKABLE GATE.** All four trailing gates
  (6m/12m x mean/t) were ON through the 2025-26 bleed (2024 strength kept the
  trailing window green as decay began -> avoided-bleed is NEGATIVE, i.e. the
  gate captured MORE of the bleed than the ungated book while skipping good
  months elsewhere; kept only 42-66% of 2017-21 gains). Lesson with LIVE-book
  relevance: monthly-granularity trailing-PnL health gates lag abrupt decay —
  do not rely on them as kill-switches.

STATUS: DONE — all three closed; no new deployable finding; two lessons banked.
