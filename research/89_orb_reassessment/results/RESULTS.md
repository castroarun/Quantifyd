# research/89 — ORB Reassessment: RESULTS

## Verdict: the SIGNAL is alive (regime-cyclic, modest); the LIVE IMPLEMENTATION was the failure. Revival only as a gated multi-day paper book.

## 1. Why the live book lost (autopsy, settled)
Same window (2024-01..2026-07), same 74 names:
| variant | net/trade | t |
|---|---|---|
| live replica (intraday, L+S, RSI) | -2.1bps | -2.0 |
| live replica long-only | -7.1bps | -5.4 |
| research config (same trigger, <=4-DAY hold) | **+14.3bps** | **+3.4** |
Plus live-only damage: shorts -Rs.18.4k of the -Rs.16.7k total;
17/46 trades exited by broker reconciliation (ops). **The intraday
squareoff was the killer deviation: EVERY intraday variant is negative in
EVERY period incl. 2015-21 (-1.5..-11bps). Intraday ORB after costs never
worked in this data. The edge was always multi-day continuation.**

## 2. Drought vs death (quarterly trace, locked config)
2026Q1 -84bps (t=-7.6) is the worst quarter on record — but comparable
regime crashes exist WITH RECOVERY: 2020Q1 -74 (COVID), 2022Q2 -62.
2025: all four quarters positive (small). 2026Q2: +29 (t=2.1). The r/81
'monotone decay' read was an annual-aggregation artifact of 2026Q1.
Correct characterization: **bull-dependent signal with recurring down-tape
drawdowns and modest post-2023 fade (IS ~25bps -> 2024-26 ~16-22bps).**

## 3. Revival grid (36 cells, never-died gate: net>=+10bps AND t>=2.0 in ALL of 2015-21 / 2022-23 / 2024-26)
Passers: the entire ts4 (4-day-hold) family; best = **W18 (90-min OR) x
gap>=0.4% x 4-day hold**: +27.4 / +20.5 / +21.6 bps (t 6.7 / 3.8 / 4.2),
62-73% of names positive per period. Parameter response is MONOTONE (wider
OR better, higher gap better, longer hold better) — the robust signature,
not a peak. ts2 weakly positive; eod (intraday) negative everywhere.

## 4. Honesty box
2024-26 is CONSUMED OOS (r/81, 2026-07-16): every number above touching it
is in-sample by construction. The never-died gate + monotone parameters are
the strongest evidence available, but the only legitimate validation path
is PAPER-FORWARD. The r/81 book-level failure also stands: ~5+ signals/day
x 4-day holds needs sleeve/capped construction + regime gate (2026Q1-type
quarters are down-tapes; the r/71 index>MA gate is the house pattern).

## 5. Recommendation
1. **Retire the intraday live implementation permanently** (never validated,
   negative in every era; also ops exits and shorts compounded it).
2. If revival wanted: **Rs-capped paper book** on W18_g40_ts4 long-only,
   equal-notional sleeves, NIFTY>50DMA entry gate, 90-day soak judged
   against +15-20bps/trade expectation. Deploy-ready on request.
3. Do not trade it live before the soak passes. No further backtest
   optimization — the grid is closed (ledger 391 cells).
