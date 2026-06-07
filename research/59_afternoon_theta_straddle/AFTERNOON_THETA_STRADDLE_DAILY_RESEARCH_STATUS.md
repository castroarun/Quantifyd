# Afternoon Theta Straddle — 12:15 Entry → 15:15 Exit, ±0.4% Move-Stop (NIFTY)

STATUS: DONE — verdict in results/RESULTS.md (SIGNAL, regime+DTE-conditional; keeper = DTE0 + VIX<16, no stop)

## The Ask

**What you asked:** "deploy cash post 12/12:15 pm as our cash wud be idle after that and
we might hav more opportunities with the theta decay. one baseline report of simple entry
at 12:15 and exit at 3:15 + adding a 0.4% stop on the underlying. attaching 2 reports for
ur assessment, u also applying it on the data we have."

**What we're testing:** A short ATM straddle entered at 12:15 IST and held to 15:15,
to harvest the afternoon theta on the idle post-noon cash (after the morning straddle
book has exited). Two variants:
- **Baseline** = sell ATM straddle 12:15, square off 15:15. No stop.
- **+0.4% SL** = same, but exit the whole straddle once the underlying moves ±0.4%
  from the 12:15 level.

Success criterion: does the afternoon window carry a positive, net-of-cost edge, and
does the ±0.4% move-stop improve risk-adjusted return (cut the tail) vs the baseline?

## The Base — what's being tested

- **Universe:** NIFTY index options, ATM (nearest 50-strike to 12:15 spot).
- **Sizing:** lot 65 × 10 lots = QTY 650 (same as the rest of the straddle book, so
  these numbers reconcile with /app/straddles). 1 premium point = ₹650.
- **Entry:** 12:15 PM, sell ATM CE + ATM PE.
- **Exit (baseline):** 15:15 PM, buy both back.
- **Exit (+SL):** whichever comes first — ±0.4% underlying move from 12:15, or 15:15.
- **Costs:** included per the user CSV (their platform's costs); for our-data replay we
  bake the same −160/round-trip used elsewhere.

## Plan

Two datasets, two variants each:

1. **User CSVs (primary, 2024-01-01 → 2026-06-05, ~598 trading days):**
   - `Trades (9).csv`  = baseline (all exits 15:15)
   - `Trades (10).csv` = baseline + 0.4% SL (early exits on stop)
   - Parse parent rows (net day P&L + VIX), compute: total, mean/day, win%, max DD,
     worst/best day, tail (P5), Sharpe, per-year, VIX-regime split. Compare 9 vs 10.

2. **Our recorded chain (cross-check, options_data.db, ~31 days since 2026-04-20):**
   - Replay 12:15→15:15 ATM straddle, baseline and ±0.4% stop, on the VPS recorder.
   - Confirm sign/shape consistent with the 2.5-yr CSVs on the overlapping recent window.

## Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-06-07 21:1x IST | Folder + STATUS written; CSVs located in Downloads | 598 trades each |

## Crash Recovery

- User CSVs: `C:\Users\arunc\Downloads\Trades (9).csv` (baseline) and `(10).csv` (+SL).
- Analysis script: `scripts/analyze_user_csvs.py` → writes `results/user_csv_summary.txt` + `results/*.csv`.
- Our-data replay (VPS): `scripts/afternoon_straddle_ourdata.py` reads
  `backtest_data/options_data.db`, writes `results/ourdata_afternoon.json`.
- Re-run analysis: `python research/59_.../scripts/analyze_user_csvs.py`.

## Files

| File | Purpose | Committable? |
|---|---|---|
| `AFTERNOON_..._STATUS.md` | This file | yes |
| `scripts/analyze_user_csvs.py` | Parse + assess the 2 user reports | yes |
| `scripts/afternoon_straddle_ourdata.py` | Replay on options_data.db (VPS) | yes |
| `results/user_csv_summary.txt` | Baseline vs +SL stats | yes |
| `results/RESULTS.md` | Final verdict | yes |

## Findings

### User CSVs (598 days, 2024-01 → 2026-06) — assessed

- **Baseline (hold to 15:15): −₹1.22 L total, PF 0.98, win 63.2%, median day +₹3,594,
  worst day −₹1.82 L, max DD −₹6.39 L.** Net NO EDGE — theta is real (wins most days)
  but a fat left tail of trend/high-vol days eats it.
- **+0.4% SL: WORSE, −₹2.16 L total.** It cuts the tail (worst −1.82→−1.01 L, maxDD
  −6.39→−5.22 L, P5 −52k→−29k) but whipsaws — exits early 45% of days, nets −₹94k worse
  than baseline. A move-stop on a short straddle trades disaster-risk for chronic bleed.
- **The lever is VIX, not the stop.** Baseline by regime: VIX<13 +₹55k, 13-16 +₹131k,
  16-20 −₹82k, **VIX≥20 −₹226k (mean −₹5,650/day)**. Gate at **VIX<16 → flips to +₹1.86 L
  over 459 days (+₹405/day)** by skipping the 139 high-vol days that cost −₹3.08 L.
- 2026 (the −₹6 L year) = the high-VIX stretch. The 0.4% stop is **most destructive in
  calm VIX<13 (+₹55k → −₹1.46 L)** — noise clips the trigger before mean-reversion.

**Verdict: SIGNAL gated by VIX, NOT a standalone strategy. Stop is the wrong fix;
the keeper is a VIX<16 gate (or defined-risk wings to cap the tail without whipsaw).**

### DTE breakdown (user CSVs) — per-year stability is decisive

Baseline VIX<16 by year × DTE: **only DTE 0 (expiry day) positive all 3 years**
(+126/+88/+37k = +₹2,514/day, 100 days). DTE 2 flipped −88k in 2026 (overfit); DTE 1/4
unstable. The "0.4% stop helps on DTE 0" claim is a **2024-only artifact** (stop hurts
DTE0 in 2025 & 2026). DTE reconstruction validated by sanity (2024 Thu/2026 Tue → DTE0).

### Our recorded chain (options_data.db, 30 days 2026-04→06, all high-VIX) — confirms

Baseline −₹1.32 L, +SL −₹1.53 L (stop worse). **True-DTE: expiry-day = −₹1.49 L/6 days
(−₹24,844/day)** → DTE0 is the WORST in high vol (gamma). Independently confirms the edge
is **DTE0 AND VIX<16**, not DTE0 alone, and that the move-stop hurts. (Print bug in the
runner emitted exit-1 after writing the JSON; fixed `%+,d`→`%+d`. JSON was intact.)
