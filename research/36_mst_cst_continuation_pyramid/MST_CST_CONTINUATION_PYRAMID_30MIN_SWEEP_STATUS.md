# MST CST Continuation & Pyramid Trigger — 30-min Research

**STATUS: DONE** — see [results/RESULTS.md](results/RESULTS.md)

## 1. The Ask

**What the user asked (verbatim):**
> Lets say MST is on and CST gives a counter signal to hedge.... now this counter signal could be short lived and the bigger trend could continue - now if we had established a condor, we would be in losses... so i want a trigger point where I double my existing debit spread or create a new one when this main trend gets re-established. Even if we need to validate right from scratch or to find better cst indicator/levels.

**What we are testing:**
1. **CST false-alarm rate:** how often does the trend continue meaningfully AFTER a CST fires? (Quantifies the user's intuition.)
2. **Pyramid trigger candidates:** which indicator-based signal best identifies "trend has resumed after a CST hedge" so we can double the directional debit spread?
3. Output: a "re-establishment" rule that, combined with the existing CST-for-hedge rule, defines a pyramiding policy: hedge on CST, double on re-establishment, hedge again on next CST, etc.

## 2. The Base — methodology

- Underlying: NIFTY50 30-min, 2024-03-01 → 2026-03-25
- MST: SuperTrend(ATR=21, multiplier=5.0) with break-of-extreme entry filter
- CST: Stochastic(14,3,3) cross-from-extreme rule (per research/35)
- Per-CST outcome label:
  - `TREND_CONTINUED` — continuation in MST direction ≥ 1×ATR AND > reversal
  - `CST_CORRECT` — reversal ≥ 1×ATR AND ≥ continuation
  - `NEUTRAL` — small move both ways

Trigger candidates evaluated:
- A: New 5-bar extreme after CST
- B: Stoch %K returns to ≥80 (long) or ≤20 (short) after dipping out of zone
- C: Break of post-CST running extreme (mini-MST style)
- D: Two consecutive closes beyond CST bar's high/low
- E: Stoch %K crosses %D in midzone (40-80 / 20-60)
- F/F2: Close above entry-anchored ATM+200 / ATM+100

Plus combinations: D AND B, D AND C, D OR B, D OR C.

Scoring: composite = coverage × max(0, avg_lead) × (1 − FP_rate).

## 3. Plan

| Step | Status |
|---|---|
| 1. Measure CST false-alarm rate (75 trends, ~400 CSTs) | ✓ |
| 2. Score 7 single triggers | ✓ |
| 3. Score 4 combinations (AND / OR) | ✓ |
| 4. Recommend pyramid trigger | ✓ |
| 5. Update MST design doc with pyramid state machine | (next session — separate edit) |

## 4. Status (live running log)

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-05-04 | Plan locked + CST continuation script written | Reuses research/35 helpers |
| 2026-05-04 | First-pass measurement done | **68% of CSTs are followed by trend continuation. 67% of FIRST CSTs are false alarms** — confirms user's intuition is correct |
| 2026-05-04 | 7 single triggers scored | D ("two closes beyond CST bar") wins on score (35.1) with 99% coverage and 31% FP |
| 2026-05-04 | 4 combinations scored | **D AND B wins on FP rate (12.6%)** with 80% coverage — recommended for pyramiding |
| 2026-05-04 | RESULTS.md written | Recommendation locked (2-year sample) |
| 2026-05-05 | NIFTY 30-min downloaded from 2020-01-01 (39 API calls, 20,372 bars) | Extended sample is 6.3 years, 302 trends, 1,495 CSTs |
| 2026-05-05 | Extended-period analysis run | Findings: false-alarm rate 64% (vs 67% in 2yr), D AND B FP rate 19% (vs 13%), trigger rank-order unchanged. RESULTS_EXTENDED.md written. |

## 5. Crash Recovery

If interrupted:
1. Run `python research/36_mst_cst_continuation_pyramid/scripts/measure_cst_continuation.py` — produces `cst_continuation_per_trigger.csv` and `cst_continuation_first_per_trend.csv`
2. Run `python research/36_mst_cst_continuation_pyramid/scripts/test_reestablishment_triggers.py` — depends on output of (1); produces `reestablishment_trigger_scores.csv`
3. Run `python research/36_mst_cst_continuation_pyramid/scripts/test_combined_triggers.py` — depends on output of (1) and (2); produces `combined_trigger_scores.csv`

Total runtime: < 30 seconds end-to-end. Idempotent; safe to re-run.

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `MST_CST_CONTINUATION_PYRAMID_30MIN_SWEEP_STATUS.md` | This doc | yes |
| `scripts/measure_cst_continuation.py` | Step 1 script | yes |
| `scripts/test_reestablishment_triggers.py` | Step 2 script | yes |
| `scripts/test_combined_triggers.py` | Step 3 script | yes |
| `results/cst_continuation_per_trigger.csv` | All 396 CST outcomes | yes (small) |
| `results/cst_continuation_first_per_trend.csv` | First CST per trend (the one that triggers condor) | yes |
| `results/reestablishment_trigger_scores.csv` | 7 single-trigger scores | yes |
| `results/combined_trigger_scores.csv` | 4 combo scores | yes |
| `results/RESULTS.md` | Final findings + pyramid policy | yes |

## 7. Findings — short version

The user's concern was empirically correct AND understated:

- **67% of the first CSTs in a trend are false alarms** (trend continues, condor's bear call gets crushed)
- When trend continues, it does so meaningfully: median 505 points (~9 ATR) of further travel beyond the CST close
- A pyramid policy can recapture this lost upside

Recommended re-establishment trigger (for pyramiding):

**D AND B — Two consecutive closes beyond CST bar's high/low AND Stoch %K back into OB/OS**
- Coverage: 80% of trend continuations correctly flagged
- Avg lead time: 48 bars (~24 hours) before MFE peak
- False positive rate: 13% (only 1 in 8 pyramids fires when trend was actually exhausting)
- Score: 33.8

Full writeup with pyramid state machine: [results/RESULTS.md](results/RESULTS.md).
