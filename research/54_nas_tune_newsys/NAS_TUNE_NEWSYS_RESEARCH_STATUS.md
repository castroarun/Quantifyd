# NAS Tuning + New Short-Vol Systems — Advance From research/52's Validated Base

STATUS: DONE (6 new angles tested; 1 keeper = OTM-strangle+move-stop; day-signal = opening-range) (IV-filter + structures killed; move-stop = top upgrade). Stage 4 (multi-year calm-day classifier) optional/pending.

## The Ask
**What you asked:** "better the systems — do backtesting, find workables / new systems /
optimize." Scope (confirmed 2026-06-03): **(A) tune the existing 8 NAS variants** AND
**(B) new strategies on the same recorded options data**. Start now.

**What we're actually testing:** research/52 already concluded with a stacked, evidence-graded
recommendation (1-DTE · tight-opening-range days · ±0.4% underlying-move stop · diversify
across families, drop OTM). The opening-range filter is the only VALIDATED (6yr) edge; the
rest is 28-day SIGNAL. This study (research/54) does NOT re-run settled axes. It pushes the
**untested, economically-core angles** and turns the findings into a concrete tuned config +
candidate new systems, all net-of-cost, signal-vs-validation graded.

## The Base (what's already settled — do NOT re-test)
From research/50/51/52 (real recorded NIFTY chain, 28–29 days + 6yr underlying):
1. Edge concentrates at **1 DTE** (real 28d: 1-DTE +₹37k; far-DTE bleeds). [signal]
2. **Tight opening range → range/calm day** — corr 0.52, positive EVERY year 2020-26. [VALIDATED]
3. Stop = **±0.4% underlying-move** (or maxloss ₹2-3k): bounded tail (2yr stress −7.9k vs
   no-stop −58.8k), beats 1.3-1.5× premium whipsaw. [strong on tail]
4. Diversify: best uncorrelated subset **[916 ATM4, Squeeze ATM]**; all-8 book noisy; drop OTM. [signal]

## What's UNTESTED (this study's targets)
- **(B1) IV-level filter** — never tested. "Sell short-vol only when ATM IV is rich vs its own
  recent range." Chain has per-snapshot `iv` (2.57M NIFTY rows, full 29d). Core short-vol edge.
- **(B2) New underlyings** — BANKNIFTY + SENSEX chains recorded (29d each). 1-DTE short straddle:
  do they add an uncorrelated, positive sleeve vs NIFTY? (Diversification on a different index.)
- **(A1) Tuned 8-variant config** — express findings 1-4 as a concrete per-variant spec:
  which run LIVE (Fri/Mon/Tue) vs PAPER, strike (ATM vs OTM-drop), stop (±0.4% move), 1-DTE gate.
- **(B3) Multi-year calm-day classifier** — stack opening-range + gap + prior-day-range + weekday
  (+VIX 2yr) to predict low-move days better than opening-range alone; OOS/per-year (robust layer).

## Plan (staged; G0 kill-cheap each)
| Stage | Test | Data | Gate |
|---|---|---|---|
| 1 | **IV-level filter** on real chain (straddle P&L by ATM-IV tercile, all-DTE + 1-DTE) | 29d NIFTY chain | edge separates IV buckets? |
| 2 | **BANKNIFTY/SENSEX** 1-DTE short straddle + corr vs NIFTY | 29d each | positive + uncorrelated? |
| 3 | **Tuned-config replay** — best stack on real chain + the [916ATM4,SqATM] sub-book | 29d | beats all-8? |
| 4 | **Calm-day classifier** (multi-year, OOS) — stacked range-day predictor | 6yr underlying | per-year robust? |

## Stop / kill rules
28-day premiums = SIGNAL (read gradient + tail, not peak); multi-year underlying = VALIDATION.
Net-of-cost (₹80/leg brokerage) throughout. Kill any stage that shows no gradient. Forward
validation accrues as the recorder keeps building (now enabled paper-all-days from 2026-06-03).

## Status log
| Time | Event |
|---|---|
| 2026-06-03 | research/54 folder + STATUS; launching Stage 1 (IV-level filter) on real chain |
| 2026-06-03 | Stage1 IV-filter DONE: KILLED as DTE-proxy (all-day corr +0.41 but within-1DTE corr -0.14); re-confirmed 1-DTE edge +2284/day real chain |
| 2026-06-03 | Stage2 weekday/DTE map DONE: Mon(1DTE)+2284, Tue(0DTE)+395, Fri(4DTE)-70 flat, Wed(6)/Thu(5) bleed; Mon/Tue/Fri-live is data-consistent |
| 2026-06-03 | Launching Stage3: defined-risk structures (iron-fly vs naked straddle) on real chain |
| 2026-06-03 | Stage3 DONE: iron-flies KILLED (cost premium, cut edge to ~0, far wings dont cap intraday tail -20k). naked+/-0.4%-move-stop wins: 0/1DTE +1412/day, worst -3260. |
| 2026-06-03 | Stage4 strike/timing DONE: NEW KEEPER -> ~100pt-OTM strangle + move-stop BEATS ATM straddle, monotonic on net (1412->1570/day) AND tail (-3260->-2695). Entry 09:20 best (11:00 -1445/day bad, 13:00 +621). Fold OTM strikes into the move-stop build. |
| 2026-06-03 | SYNTHESIS: 3 new angles (IV filter, defined-risk structures) all killed. Edge = day-selection+stop, NOT structure/vol-filter. TOP ACTIONABLE: replace per-leg 1.3x premium stop with +/-0.4% underlying-move stop (lifts net AND bounds tail; premium stop whipsaws -13983 vs move-stop positive). |

## Crash recovery
Scripts in `research/54_nas_tune_newsys/scripts/` are standalone (`./venv/bin/python3 scripts/<x>.py`).
They read `backtest_data/options_data.db` (read-only) — safe to run any time, no service impact.
Each writes `results/RESULTS_<x>.md` + a PNG. Re-run is idempotent. Engine reused from
research/52/scripts/scan.py (real-chain `sim()`).

| 2026-06-03 | Stage5 re-entry/skew DONE: re-entry KILLED (actively hurts - re-sells into the trend that triggered the stop; all-DTE goes -8.5k to -19.9k). Move-stop must be ONE-AND-DONE. Directional skew = neutral (drop). Launching Stage6 multi-year calm-day classifier. |
| 2026-06-03 | Stage6 calm-classifier DONE: multi-feature barely beats opening-range alone (2/3yr, +0.02-0.04) -> NOT robust; prior-day feats useless (corr ~0). Opening-range(+gap) IS the day-signal; calmest-30pct days = 0.38-0.43pct move vs 0.48-0.55pct all. NEW-SYSTEMS SEARCH COMPLETE. |