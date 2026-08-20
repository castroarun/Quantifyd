# PCR / OI-Change Adjustment Signals — 85 Days of 1-min Full Chain, Both Venues

STATUS: **RUNNING** (launched 2026-08-20 by the ops session; executed by a research agent)

## 2. The Ask

**What Arun asked (2026-08-20):** "people do options adjustment systems based on PCR,
change in OI etc. Can you initiate a study and hand it over to an agent to study using our
backtest data and report it? You have 80+ days of options data to study and see if there
are correlation points."

**What we are actually testing.** Our live books are **short-volatility straddles**. So the
question that matters is not "does PCR predict direction" in the abstract, but:

> Does any PCR / OI-change signal, computable in real time from our recorded chain,
> tell us **early enough to act** that a short straddle is about to go wrong — and does
> acting on it beat doing nothing, net of costs?

Three sub-questions, in gate order:
1. **G1 — is there any relationship at all?** Information coefficient of each signal vs
   the forward move of the underlying, and vs the forward change in straddle premium,
   at 5 / 15 / 30 / 60-minute horizons.
2. **G2 — does an adjustment rule beat the baseline?** The baseline is **hold to the time
   exit** (research/114 showed holding beats nearly every stop we have tried, so this is
   a hard bar, not a strawman).
3. **G3 — is it robust?** Per-venue, per-DTE, per-month stability; random and
   date-matched controls; cost sensitivity.

## 3. The Base — data and definitions

- **Source:** `backtest_data/options_data.db :: option_chain` (read-only).
  Columns: `snapshot_time, symbol, expiry_date, strike, instrument_type, ltp, bid, ask,
  oi, volume, iv, delta, gamma, theta, vega, underlying_spot, lot_size`.
- **Granularity: 1 MINUTE** (not 3-second — verified: stamps are exactly on the minute).
  ~85 trading days, 2026-04-20 → 2026-08-20, NIFTY and SENSEX, full chain per snapshot.
- **Coverage caveat to measure first:** OI is non-null on ~78% of NIFTY rows and ~66% of
  SENSEX rows; IV ~74%. Exchange OI is published with lag and can be stale within a
  minute — quantify this before trusting any OI-derived signal.
- **Signals to construct** (all causal, using only data available at that minute):
  - PCR-OI (total, and within +/-N strikes of ATM), PCR-volume
  - dOI per strike over 5/15/30 min; ATM and wing OI build-up vs unwind
  - OI-weighted support/resistance (largest CE and PE OI strikes) and distance to them
  - Max-pain and its intraday drift
  - Optional cross-checks: IV skew, ATM IV change
- **Outcome variables:** forward underlying return; forward change in the ATM straddle
  premium (what actually hurts our books); and a "trouble" flag = straddle premium
  breaching +20% / +30% / +50% of its entry within the horizon.
- **Universe/period for the rule test:** the live constructions — 09:16 ATM straddle held
  to 15:15/15:20 — per venue and per DTE.

## 4. Plan — gates and the grid

| Gate | Question | Pass criterion |
|---|---|---|
| G0 | Is the OI data trustworthy? | staleness and coverage quantified per venue/DTE; a documented rule for handling stale/missing OI |
| G1 | Any signal at all? | \|IC\| >= 0.05 stable in sign across venues AND months for at least one signal/horizon |
| G2 | Beat do-nothing? | an adjustment rule (exit / hedge / roll the threatened side) beats hold-to-time-exit net of costs, on the same days |
| G3 | Robust? | holds per venue, per DTE, per month; survives random-entry and date-matched controls; monotonic in threshold, not a peak |

**Adjustment actions to test at a trigger** (each vs the hold baseline): exit the whole
straddle · exit the threatened leg only · roll the threatened leg out (premium-matched,
the ATM4 mechanic) · buy a protective wing · do nothing but tighten the combined stop.

**Multiple-testing discipline:** the grid is large. Pre-register the signal list, report
how many combinations were tried, and treat any single winner with a Bonferroni-style
haircut. A clean negative is a perfectly good outcome and must be reported as such.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-20 ~13:2x IST | Data recon: 1-min (not 3-sec), OI coverage measured | corrected the premise before launch |
| 2026-08-20 ~13:3x IST | STATUS written, agent launched | results to `results/RESULTS.md` |

## 6. Crash Recovery

- Everything read-only against `options_data.db`; no live state, no orders, nothing to undo.
- Scripts live in `research/115_pcr_oi_adjustments/scripts/`, outputs in `results/`.
- Re-run any script directly with `venv/bin/python3 <path>`; each writes its own CSV/JSON.
- If the agent dies mid-run, re-read this file and the partial CSVs in `results/`.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `PCR_OI_ADJUSTMENTS_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | signal build + IC + rule bake-off | yes |
| `results/*.csv` | per-signal IC, per-rule day detail | yes if small |
| `results/RESULTS.md` | verdict (NO EDGE / SIGNAL / STRATEGY) | yes |

## 8. Findings

(to be written by the research agent)
