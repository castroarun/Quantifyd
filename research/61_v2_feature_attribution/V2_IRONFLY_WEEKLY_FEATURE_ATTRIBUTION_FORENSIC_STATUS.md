# V2 Iron Fly — Comprehensive Causal-Feature Screen to Kill Negative Weeks
STATUS: DONE — see results/RESULTS.md. Verdict: ONE factor (entry-time vol compression),
read two independent ways — narrow daily CPR (<0.10%) + inside-week. Stacked → Calmar 1.03→2.00,
DD −1.17L→−0.78L, 135/204 trades kept. All trend/momentum/RSI/MA/Ichimoku/pivot/range-break
features showed NO usable signal. Candidate overlay → forward-paper before live.

## The Ask
**What you asked:** "study all the negative and positive weeks from the baseline strategy
results again — weekly CPR range of the trading week and prior week, monthly/weekly RSI,
Bollinger bands, moving averages, ichimoku, monthly pivots/CPR, previous-week range breaks,
inside candles… assess comprehensively, find any pattern or combination that eliminates
some/most of the negative weeks (losing a few positive weeks is acceptable). Update the app
study."

**What we're actually testing:** Over the 204 locked-base V2 trades (2% wings + 2% move-stop,
ex-COVID; 2019-02→2026-05), compute a battery of **causal** technical features known at the
09:20 entry (prior-day / prior-completed-week / prior-completed-month only), and ask which —
singly or in small validated combinations — separate the losing cycles from the winners well
enough to be used as an **entry skip-filter on top of the VIX≥13 floor**, without overfitting.

## The Base (what's being attributed)
- **Trade set:** `C` from `/tmp/cd_data.py` = AlgoTest per-trade rows `(entry_date, entry_VIX,
  full_hold_P&L)` for the locked base (2% wings, 2% move-stop, PT40, 10 lots, net of costs).
  Ex-COVID = 204 trades. Headline screen on the **VIX≥13 subset (the live book)**; full set
  reported for sample size.
- **Target:** per-trade full-hold P&L (positive = good week for the short fly).
- **Success criterion:** a skip-rule that RAISES Calmar AND cuts MaxDD on VIX≥13, is
  **monotonic** (dose-response, not a magic middle bucket), **per-year consistent** (dropped
  bucket negative in most years), has a **mechanism**, and **passes walk-forward** (pick
  threshold on train half, apply blind to test half).

## Features (all causal; computed in `scripts/feature_screen.py`)
Daily NIFTY bars (Kite tok 256265, from 2017 for indicator warm-up). For each entry date:
1. Daily prior-day CPR width % (control — already known winner)
2. Weekly CPR width — this-week (= last completed week) and prior week; + narrowing/widening
3. Monthly CPR width; monthly pivot distance
4. RSI(14): daily, weekly, monthly (prior completed bar)
5. Bollinger (20,2): daily & weekly band-width % and %B (position)
6. Moving averages: close vs 20/50/200 DMA; weekly 20WMA; slopes
7. Ichimoku: price vs cloud + cloud thickness, daily & weekly
8. Prior-week range break: close above prior-week high / below prior-week low / inside
9. Inside candle: prior day inside, last week inside
10. Controls: entry VIX, 20d realized vol

## Plan / grid
- Univariate quartile attribution for every numeric feature + group means for categoricals.
- Auto-flag features whose extreme (Q1/Q4) bucket is negative, monotonic-ish, and negative in
  ≥4/7 years → candidate skip-rules.
- Walk-forward the flagged candidates (2019-22 ↔ 2023-26), report train-picked threshold
  applied blind to test.
- Test the best 2-3 in combination (guard: combinations overfit fastest on n≈200).
- **Multiple-testing honesty:** ~20+ feature variants tested → demand a stern bar; expect most
  to die. Report the full budget, not just survivors.

## Status (live log)
| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-10 ~10:1x | Folder + STATUS written, script being built | screen of 204 trades, ~20 causal features |

## Crash Recovery
- Trade data: `/tmp/cd_data.py` on VPS (regenerate from `research/60_v2_straddle_optimization`
  CSVs if missing — see that folder's STATUS). Daily bars pulled live from Kite each run.
- Re-run: `cd /home/arun/quantifyd && PYTHONPATH=. venv/bin/python3 /tmp/feature_screen.py`
  (needs fresh `backtest_data/access_token.json`; auto-login refreshes it ~08:5x on trading days).
- Read-only; places NO orders. Safe to run during market hours.

## Files
| File | Purpose | Committable |
|---|---|---|
| `scripts/feature_screen.py` | The attribution screen | yes |
| `scripts/cpr_late_entry_probe.py` | (prior turn) late-entry feasibility probe | yes |
| `results/RESULTS.md` | Honest verdict + survivors + caveats | yes |
| `results/screen_output.txt` | Raw screen dump | yes |

## Findings
(pending run)
