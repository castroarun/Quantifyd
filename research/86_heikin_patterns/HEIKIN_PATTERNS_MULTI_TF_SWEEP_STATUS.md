# research/86 — Heikin-Ashi Continuation Patterns (user chart idea), Multi-TF

STATUS: DONE — STRATEGY CANDIDATE (OOS passed; see results/RESULTS.md)

User spec (2026-07-20, from a SENSEX 30-min HA chart): after 2 consecutive
GREEN HA candles, a break of their high = long entry; mirror (2 red, low
break) = exit and/or short. Test across timeframes; assess related HA
patterns; find what works.

## Locked mechanics

- HA construction (causal): ha_close=(O+H+L+C)/4; ha_open=(prev_ha_open+
  prev_ha_close)/2. Color = ha_close vs ha_open.
- **Detection on HA, triggers on REAL prices** (HA highs are not tradeable):
  trigger level = max REAL high of the pattern bars; entry when a bar CLOSES
  above it → fill next bar open (house convention). Exits mirror.
- Exit = opposite-pattern break (user spec, stop-and-reverse structure —
  long flat on 2-red low-break; shorts tested as separate cells). No target,
  no time-stop (the mirror IS the exit); safety: positions force-closed at
  data end are dropped.
- Patterns (4): FLIP1 (first color flip, the classic HA system);
  2GREEN break (user spec); 3GREEN break; 2GREEN-NOWICK break (both pattern
  bars with no lower HA wick — HA-lore "strong trend" candles).
- TFs: 15m, 30m, 60m (resampled from clean 5-min), daily.
- Universe: intraday TFs → NIFTY50 + BANKNIFTY + 9 deep F&O names
  (2015-02→2021-09 IS; Val/OOS reserved). Daily TF → F&O ~80 (2005-2017 IS).
- Costs: futures-proxy, 1bp index / 3bp stocks.

## Grid (LOCKED): 4 patterns × {L, S} × 4 TFs = 32 cells (ledger +32)

Gate: standard G1 (pooled net t≥3, ≥55% names positive where multi-name,
coherent across TF). Priors stated: ALL equity/index shorts have failed at
every horizon in r/81-83 — short cells are expected negative and run for
completeness; long HA-flip ≈ a noisy trend filter, and the zoo showed
per-name trend timing subtracts return — the open question is whether the
BREAK-confirmation (user's addition) changes that.

## Status

| Date/time | Event |
|---|---|
| 2026-07-20 ~17:30 IST | Pre-registered; runner `scripts/run_ha_patterns.py` |

## Findings

(after run)

## VERDICTS (2026-07-20 ~18:20 IST)

- **SHORTS: all 16 short cells negative (t -2.7..-21) — 4th independent
  confirmation; the mirror-short leg of the user spec is dead. CLOSED.**
- FLIP1 (classic HA flip): intraday churn-negative (confirms zoo lesson);
  daily long +65bps t=13.8 but 6-day churny holds — inferior cousin of
  turtle exits, not pursued.
- Daily-TF break cells: huge (+12-17% per trade, t 9-11) but hold ~4-6
  MONTHS — that is turtle-family positional trend following rediscovered
  through HA smoothing, already covered by r/83. Not new.
- **NEW FINDING — the user's pattern with the no-wick refinement, LONG,
  30-min TF (2GREEN_NW_L_30m): IS +47bps t=3.74 (91% syms, 86% yrs) →
  VAL +36bps t=2.31 (82% syms, ALL years positive). WF-efficiency 0.77 —
  the best of the engagement (ORB was 0.53). Avg hold ~6 sessions.**
  60m variants positive but Val-underpowered. 15m weaker (costs).
- Next: breadth replication (full F&O 5-min universe, IS+Val — free of OOS)
  → then the one-time OOS decision (2024+, user gate).
