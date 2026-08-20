# PCR / OI-Change Adjustment Signals — 85 Days of 1-min Full Chain, Both Venues

STATUS: **DONE** — verdict **NO EDGE** (see `results/RESULTS.md`)

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
| 2026-08-20 ~13:37 IST | **G0 PASS (with caveat)** — `g0_data_recon.py` over 85 days x 2 venues | NIFTY OI genuinely live (p50 **97.3%** of minute-pairs change); **SENSEX OI is published on a strict 3-min cadence** (median run-length exactly 3, change rate pinned at 35.7-38.1%). OI null 0% on the near expiry; OI *zero* 8.0% NIFTY / 29.5% SENSEX; IV null 24.0% / 13.4%. Chain window is ATM+/-20 strikes, not the literal full chain. RULE ADOPTED: all dOI horizons >= 5 min (so the 3-min cadence invalidates nothing), zero-OI strikes KEPT (dropping them would be liquidity selection). |
| 2026-08-20 ~13:41 IST | Feature build done — `build_features.py` | 60,092 minute rows, **168 day-venue units** (84 NIFTY + 84 SENSEX); 26 pre-registered causal signals + outcomes fret/fabs/fdprem at 5/15/30/60 min. |
| 2026-08-20 ~13:5x IST | G1 raw pass — `g1_ic.py`, 312 tests | 44 "passed" the pre-registered bar. Top: `pcr_oi_all` vs 60-min forward return **IC -0.4076, t -21.0**. FLAGGED AS IMPLAUSIBLE — an IC of -0.41 on a liquid index does not exist. Suspected price-anchor (finite-sample mean-reversion) + time-of-day artifacts. Did NOT advance to G2 on this. |
| 2026-08-20 ~14:0x IST | **G1b controls** — `g1b_controls.py` | Placebos + shuffled-day + partial IC off [spot deviation, minute]. **`placebo_negspot` (zero option information -- literally minus the intraday price deviation) scores raw IC +0.51, LARGER than every real signal, and collapses to +0.005 controlled.** PCR / wall / max-pain family loses 70-95% of its raw IC and several flip sign. |
| 2026-08-20 ~14:1x IST | **G1c full controls** — `g1c_controls2.py` | Controls extended to [spot dev, minute, premium level, trailing 15m return, trailing 30m return]; outcome normalised by entry premium. `atm_iv`->fdprem (-0.39) exposed as the premium-level artifact -> **-0.063** vs a random-walk placebo at +0.049. dOI family (contaminated by trailing return -- the r/109 reversal effect) -> \|IC\| <= 0.03. **Best partial IC anywhere in the PCR/OI/wall/max-pain family: `pcr_vol_all` +0.0773, t 3.55** (below the Bonferroni bar of 4.0) with a shuffled twin of -0.0699. **G1 FAIL — zero PCR/OI signals alive.** |
| 2026-08-20 ~14:2x IST | **G2 run anyway for a rupee answer** — `g2_rules.py` | 22 pre-registered triggers x 4 actions = 96 pooled arms vs HOLD, each with a random-trigger twin. HOLD nets **+Rs 238.3/lot/day** (NIFTY +282.9, SENSEX +193.7; win 66.7%). **0/96 arms beat HOLD at t>=3.9 while also beating their random twin; 67/96 do WORSE than triggering at a random minute.** Anti-monotonic in threshold (the rule improves the less it acts). Every arm pays the same 4 leg-sides as HOLD, so the failure is NOT a cost artifact. |
| 2026-08-20 ~14:3x IST | G1d targeted probe of the lone survivor — `g1d_skew_probe.py` | `skew` survives every control (IC -0.138 t -13.9 vs 5-min forward return, incl. adding trailing-5m to the controls) BUT scores **+0.000 vs \|forward move\| and +0.005 vs forward premium change** -> a pure DIRECTION signal with zero information about whether a short straddle is about to go wrong. Out of scope; logged as a lead, not a finding. |
| 2026-08-20 ~14:4x IST | **DONE — verdict NO EDGE** | `results/RESULTS.md` written; INDEX.md row added. Recommendation: drop this line of enquiry. |

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

### Verdict: **NO EDGE**

**Not one** PCR, dOI, OI-wall or max-pain construction carries information about where a
short straddle is going, once you control for things that are not option information at
all. The signals that looked strongest were the most contaminated.

**The single decisive exhibit.** A placebo containing zero option information --
`placebo_negspot`, literally minus the intraday price deviation -- scores a raw rank-IC of
**+0.51**, larger than every real signal in the study, and collapses to **-0.0006** under
control. The eye-catching raw G1 table (`pcr_oi_all` at IC -0.41, t -21) was measuring the
shape of a price path, not open interest. `pcr_oi_all` retains **3.6%** of its raw IC after
control; several signals flip sign.

**Why the raw numbers lied.** Two mechanical effects, both of which *also* produce the
"sign-stable across venues and months" property our G1 gate was asking for:
1. **Price-anchor / finite-sample mean-reversion bias.** PCR levels, wall distance and
   max-pain deviation are all monotone functions of where spot sits relative to a sticky
   intraday anchor. Regressing forward returns on such a variable inside one finite path
   is biased even for a pure random walk -- and biased with the *same sign every day*.
2. **Time-of-day.** Straddle premium decays monotonically through the session, so any
   signal with an intraday time trend correlates with forward premium change for free.

**The rupee answer.** HOLD-to-time-exit nets **+Rs 238/lot/day** (NIFTY +283, SENSEX +194,
win 66.7%, n=168). Of 96 pre-registered adjustment arms, **0 beat HOLD significantly** and
**67 did worse than pulling the same trigger at a random minute**. The arms are
anti-monotonic in threshold -- they improve strictly as they act less often, the signature
of a signal worth nothing. Crucially, every arm pays the **same 4 leg-sides as HOLD**
(exiting early adds none), so adjustment was given a zero-incremental-cost handicap and
still lost. Independently reproduces research/114 from a different direction.

**G0 gave us one durable operational fact worth keeping:** **SENSEX OI is published on a
strict 3-minute cadence** (median run-length exactly 3 on all 85 days), while NIFTY OI is
genuinely live minute-to-minute. Any future SENSEX OI signal finer than 3 minutes is
structurally impossible.

**One out-of-scope byproduct, flagged not claimed.** `skew` (PE IV 3 strikes below ATM
minus CE IV 3 strikes above) survives every control at IC -0.138 (t -13.9) vs the 5-minute
forward return -- but scores **+0.000 against |forward move| and +0.005 against forward
premium change**. It is a pure *direction* signal with precisely zero information about
whether a short straddle is about to go wrong, which is the only thing our books need. A
lead for a future directional study; it pays a delta-neutral book nothing.

### Recommendation

**Drop this line of enquiry.** Do not build a PCR/OI adjustment layer on the live books.
If ever revisited, do so after ~12 months of recorded chain spanning a volatility-regime
change, and only for the far-tail PCR this window cannot see. The genuinely open lever for
these books remains **sizing and per-DTE participation** (research/103, 104, 113, 114), not
intraday adjustment.

### Honest limitation

One regime, four months, 168 day-venue units. A negative result on 4 months is not proof of
a permanent absence of edge -- it is proof that nothing here is strong enough to detect in
4 months. That is a weaker claim, but it is more than enough to refuse to build on. Full
caveats in `results/RESULTS.md` section 8.
