# VOLSURGE Multi-TF Sweep — Execution Log & Results

Live polling log (~4-min cadence) for the research/40 VOLSURGE backtest sweep
running on the VPS (`94.136.185.54`, commit `d3b85f7`, pid 1002876,
`/tmp/volsurge.log`, `venv/bin/python`).

- **Grid:** 1,440 signal cells/stock × 79 F&O stocks, 13 exit policies/signal.
- **Spec:** locked from 10 chart examples — see
  `VOLSURGE_PDR_BREAK_WEEKLY_CPR_INTRADAY_SWEEP_STATUS.md` (authoritative).
- **Run #1 failed** (git-reset clobbered the tracked live CSV → 0 rows);
  fixed by gitignoring sweep outputs (`d3b85f7`). **Run #2 is the valid run.**
- Edge metrics (Sharpe/WR/payoff/robustness) appear ONLY after the
  aggregation phase that follows all 79 stocks. No findings before then.

## Poll log (newest at bottom)

| Tick | IST time | Runner | Signal rows | Stocks done | Elapsed | Agg | Note |
|---:|---|---|---:|---|---|---|---|
| 1 | ~20:3x | ALIVE (pid 1002876) | 28,104 | 22/79 (DELHIVERY) | 16.6 min | no | Healthy, persisting. ~0.75 min/stock → ETA ~45 min more signal-gen |
| 2 | ~20:4x | ALIVE (pid 1002876) | 44,577 | 35/79 (HDFCLIFE) | 27.8 min | no | Healthy. HDFCBANK took 194s (8yr history). ~0.8 min/stock → ~35 min more signal-gen then aggregation |
| 3 | ~20:5x | ALIVE (pid 1002876) | 51,791 | 38/79 (HINDUNILVR) | 34.5 min | no | Healthy. 8yr-history names heavy (HINDUNILVR 305s, HDFCBANK 195s) — pace varies; remaining ~41 stocks mix heavy/light. ETA ~30-45 min more |
| 4 | ~21:0x | ALIVE (pid 1002876) | 60,791 | 41/79 (INDUSINDBK) | 39.0 min | no | Healthy, ~52% through stocks. Remaining incl heavy 8yr names INFY/ITC/KOTAKBANK/RELIANCE/SBIN/TCS. ETA ~45-50 min more signal-gen + aggregation |
| 5 | ~21:1x | ALIVE (pid 1002876) | 80,508 | 50/79 (M&M) | 56.3 min | no | Healthy, ~63% through. INFY/ITC/KOTAKBANK done (heavy); RELIANCE/SBIN/TCS still pending. ETA ~35-40 min more |
| 6 | ~21:2x | ALIVE (pid 1002876) | 92,287 | 62/79 (POWERGRID) | 63.5 min | no | Healthy, ~78% through. 17 stocks left (RELIANCE/SBIN/TCS heavy + light). ETA ~25-30 min then aggregation |
| 7 | ~21:3x | **DONE (exited)** | 121,278 | **79/79 (WIPRO)** | 84.4 min | **YES @21:19** | Signal-gen complete → aggregated: 70,148 ranked cells, 19 leaders, RESULTS.md written. **LOOP ENDS.** |

## Findings (final — from VPS results/RESULTS.md, 2026-05-15 21:19)

**121,278 signals** (long 81,524 / short 39,754), 79 F&O stocks, 2018→2026-05-15,
70,148 ranked cells.

### Honest headline: NO robust universe-wide edge — same verdict as research/34 VOLBO

- **Aggregate risk-adjusted return is ~0.12–0.16 avg Sharpe** across all cells
  (n≥15, mean>0) — i.e. essentially no edge on average. WR ~48–52%.
- **Only 1 of 79 stocks passes the robustness gate** (Sharpe≥0.5, n≥15,
  MidQ≥3): **TCS**. With 70,148 cells tested, ~1 survivor is roughly what
  multiple-comparison luck alone produces — treat as a candidate, NOT a proven
  edge. (research/34 had only 2/79 — consistent.)
- The 10 hand-picked chart examples were hindsight winners; across the full
  population the confluence is ~coin-flip. This is the ex#9 principle made
  concrete: necessary-but-not-sufficient, weak as a standalone systematic edge.

### The single promote candidate

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|
| **TCS** | 60min | sma200, cpr≤0.75, vm1.5, loose, no-clearroom, **carry** | **long** | **T_NO** | 18 | **2.56** | **94.4** | **1.085** |

Survives across 54 mid-quality variants → most internally consistent result.
Still n=18 and 1-in-79 → needs walk-forward / OOS before any capital.

### Secondary observations (directional, not actionable alone)

- **Strong LONG bias.** Shorts have ~zero edge (most short-Sharpe = 0 / neg);
  only ITC leans short. Contradicts the ex#4 symmetry hope — on data, the
  short side of this system does not work.
- **vm1.5 (loosest volume mult) + loose candle + carry** recurs in nearly
  every leader — the *stricter* volume/candle filters did NOT improve edge.
- **Timeframe:** 15min marginally best (avg Sharpe 0.158) but all ~equal/weak.
- **Exit:** Chandelier 1.5/2.0 best (~0.17); tight ATR stops & step-trail worst.
- Recurring leader shape: **long, daily-uptrend (sma200), narrow-ish CPR,
  vm1.5, loose, carried, no hard target** (T_NO/Chandelier).

### Recommended next step

Do NOT deploy. Either (a) walk-forward/OOS-validate the TCS-60min cell and the
"long + sma200 + vm1.5 + loose + carry + Chandelier" archetype on held-out
years, or (b) treat VOLSURGE as a discretionary screen (the scanner) rather
than an automated strategy. Full tables: `results/RESULTS.md` (VPS) — also
mirrored below.
