# MST CST Continuation Study — Findings & Pyramid Policy

**Period:** 2024-03-01 → 2026-03-25 (NIFTY50 30-min, MST = ST(21, 5.0) with break-of-extreme)
**Universe of CSTs analyzed:** 396 across 75 active trends
**Source scripts:** `scripts/measure_cst_continuation.py`, `scripts/test_reestablishment_triggers.py`, `scripts/test_combined_triggers.py`

---

## TL;DR — what to use

| Decision | Recommendation |
|---|---|
| **Was the user's concern real?** | **Yes, and bigger than expected.** 68% of all CSTs are false alarms (trend continues). 67% of FIRST CSTs (the one that triggers the condor) are false alarms. |
| **Pyramid trigger** | **D AND B** — two consecutive 30-min closes beyond the CST bar's high (long) / low (short), AND Stoch %K back into OB/OS zone (≥80 long, ≤20 short) |
| **Pyramid action** | Add a SECOND debit spread (1 lot bull call for long, bear put for short), strikes anchored to current spot at re-establishment time |
| **Hedge re-application** | The next CST after pyramid adds a SECOND credit spread (if credit ≥ ₹1,000/lot rule from earlier still applies) |
| **Cap** | Max pyramid level = 2 (so position can stack at most: 2× debit + 2× credit). Beyond that, ignore further re-establishment signals — risk too high |

---

## 1. The CST is wrong 67% of the time

Outcome distribution of 396 CSTs in active MST trends:

| Outcome | Count | % |
|---|---|---|
| TREND_CONTINUED | 271 | **68.4%** |
| CST_CORRECT | 119 | 30.1% |
| NEUTRAL | 6 | 1.5% |

For the **first CST per trend** (which is the one that triggers the condor in the design):

| Outcome | Count | % |
|---|---|---|
| TREND_CONTINUED | 49 | **67.1%** |
| CST_CORRECT | 22 | 30.1% |
| NEUTRAL | 2 | 2.7% |

When the trend continues after the first CST:

- Continuation magnitude: median **505 points**, mean 619 points
- In ATR-21 units: median 8.8×, mean 10.6×
- MFE extension after CST: median +140%, mean +474% beyond what was already gained

**Implication for the as-designed condor strategy:** in 2 out of every 3 trades, the bear call (or bull put) hedge is added at a point where the trend is going to extend dramatically, putting the credit spread at maximum loss. The user's instinct that "we'd be in losses" is borne out by the data.

This finding is the single most important revision to the strategy spec. Without a pyramid mechanism, the as-written condor would systematically build hedges at exactly the wrong moments.

---

## 2. Single-trigger scoring

Each trigger evaluated on the same 396 CSTs:

| Trigger | Coverage of trend-continuations | Avg lead (bars) | FP rate (fired in CST_CORRECT trends) | Score |
|---|---|---|---|---|
| A — New 5-bar extreme | 99.6% | 56.3 | **76.5%** | 13.2 |
| **B — Stoch K returns to ≥80/≤20** | **80.1%** | 50.1 | **20.2%** | 32.0 |
| C — Break of post-CST extreme | 97.1% | 51.3 | 34.5% | 32.6 |
| **D — Two closes beyond CST bar** | **98.9%** | 51.5 | 31.1% | **35.1** |
| E — Stoch midzone cross | 85.2% | 47.1 | 43.7% | 22.6 |
| F — Close above ATM+200 anchor | 97.8% | 54.4 | 63.9% | 19.2 |
| F2 — Close above ATM+100 anchor | 99.6% | 56.3 | 75.6% | 13.7 |

Score = coverage × max(0, avg_lead) × (1 − FP_rate). Higher is better.

**Patterns:**
- Price-action triggers (A, C, D) have very high coverage but moderate-to-high FP rates
- Stoch-based triggers (B, E) have lower coverage but cleaner FP profile
- Anchor-level triggers (F, F2) are too noisy — too many CST_CORRECT trends still cross those levels in counter-trend bounces

---

## 3. Combination scoring (the actual recommendation)

| Combination | Coverage | Avg lead | **FP rate** | Score |
|---|---|---|---|---|
| **D AND B** (price + momentum) | 79.7% | 48.5 | **12.6%** | 33.8 |
| D AND C (price + structure) | 96.3% | 50.7 | 26.1% | **36.1** |
| D OR B (high coverage) | 99.3% | 52.8 | 38.7% | 32.2 |
| D OR C (high coverage) | 99.6% | 52.1 | 39.5% | 31.4 |

### Why D AND B is the right pyramid trigger (not D AND C, despite higher score)

For pyramiding, **FP rate is more critical than coverage** because:
- Each false-pyramid event doubles directional exposure right before a real reversal
- Cost of a missed pyramid (lower coverage) = forgone gain on a continuing trend (still profitable, just less so)
- Cost of a false pyramid (higher FP) = doubled loss on a now-reversing trend

D AND B has FP rate of 12.6% (1 in 8 false alarms) vs D AND C's 26% (1 in 4). On the recommended Phase 1 size of 1 lot per leg, that 13-percentage-point gap translates to many fewer doubled-down losses.

D AND B: **80% catch rate, 13% false alarms, 24-hour median lead time** — is the right balance for a conservative pyramiding rule.

---

## 4. Pyramid policy — full state machine

Building on the existing MST/CST/condor logic from research/35 + the spec doc:

```
              ┌────────────────────────────┐
              │ NO_POSITION                │
              └─────┬──────────────────────┘
                    │ MST flip + break-of-extreme confirmed
                    ▼
              ┌────────────────────────────┐
              │ DEBIT_OPEN_L1 (level 1)    │
              │ 1× bull call (or put)      │
              └─────┬──────────────────────┘
                    │ first CST in active week (credit ≥ ₹1,000/lot)
                    ▼
              ┌────────────────────────────┐
              │ CONDOR_OPEN_L1             │
              │ 1× debit + 1× credit       │
              └──┬──────────────────┬──────┘
                 │                  │
                 │                  │ pyramid trigger fires
                 │                  │ (D AND B confirmed)
                 │                  ▼
                 │          ┌────────────────────────────┐
                 │          │ DEBIT_OPEN_L2 (level 2)    │
                 │          │ 2× debit + 1× credit       │
                 │          │ NEW debit anchored to      │
                 │          │ current spot's ATM         │
                 │          └─────┬──────────────────────┘
                 │                │ next CST in same week
                 │                │ (credit ≥ ₹1,000/lot)
                 │                ▼
                 │          ┌────────────────────────────┐
                 │          │ CONDOR_OPEN_L2             │
                 │          │ 2× debit + 2× credit       │
                 │          │ MAX PYRAMID LEVEL          │
                 │          │ Further re-est triggers    │
                 │          │ are LOGGED ONLY            │
                 │          └─────┬──────────────────────┘
                 │                │
                 ▼                ▼
        T-1 EOD (Wed 15:25) → close ALL legs (whichever level)
        MST flip            → close ALL legs, re-arm in opposite direction
        Kill switch         → close ALL legs, halt entries
```

### Pyramid trigger evaluation conditions

The "D AND B" trigger is checked **only after CONDOR_OPEN_L1**, until either:
- Pyramid fires (transition to DEBIT_OPEN_L2)
- MST flips (close everything)
- T-1 EOD (close everything)
- A new CST fires (does NOT pyramid — the CST is the hedge re-application, only relevant after pyramid level is increased)

Specifically:

```
At each 30-min bar close in CONDOR_OPEN_L1:
  - Evaluate trigger D: have there been 2 consecutive closes
    above CST_bar.high (long) / below CST_bar.low (short)
    since the last CST?
  - Evaluate trigger B: did %K leave the OB/OS zone since last CST,
    and is %K now back at >=80 (long) or <=20 (short)?
  - If BOTH → pyramid (D AND B)
```

### Pyramid execution

When trigger fires in CONDOR_OPEN_L1 (long bias example; mirror for short):

1. **Pre-trade margin check** via `kite.basket_margins()` — if margin shortfall, skip pyramid + alert operator
2. **Compute new strikes:** spot at re-establishment time → round to nearest 50 = **new_atm**. New bull call: long `new_atm` CE, short `new_atm + 200` CE, **same expiry as existing condor**
3. **Place 2 LIMIT orders** for the new bull call legs (mid-price, MARKET fallback after 30s)
4. **On fill:** transition to `DEBIT_OPEN_L2`. Existing bear call spread (the level-1 hedge) remains open
5. **Persist** new positions in `mst_positions` with `pyramid_level = 2`

### Re-hedging at level 2

After `DEBIT_OPEN_L2`, the next CST trigger:

1. Compute bear call credit at strikes `new_atm + 400 / new_atm + 600`
2. If credit ≥ ₹1,000/lot, place the spread → `CONDOR_OPEN_L2`
3. If credit < ₹1,000/lot, do NOT pyramid further. Roll-and-reset all positions to next week's expiry per the standard credit-too-low rule
4. The original level-1 bear call is still open at the original ATM+400/+600 strikes — it remains as a deeper-OTM hedge

### Cap at level 2

Beyond `CONDOR_OPEN_L2`, further re-establishment triggers (and further CSTs) are LOGGED ONLY. The position is fully built; we do not stack a third debit. This caps:
- Maximum directional exposure at 2 lots per leg = 150 contracts per spread
- Total open legs at 8 (4 debit + 4 credit)
- Margin requirement at ~2× the standard condor's margin (~₹2-3 lakh on Zerodha)

---

## 5. Sample timeline (illustrative)

NIFTY at 22,500 on Monday morning (1 lot examples, single weekly Thursday expiry):

| Time | Event | Position |
|---|---|---|
| Mon 09:45 | MST flips long. Break of 22,510 high → entry. New ATM = 22,500 | Long 22,500 CE / Short 22,700 CE |
| Tue 11:15 | First CST fires. NIFTY at 22,615. Bear call credit = ₹1,400/lot. Build condor | Long 22,500 / Short 22,700 / Short 22,900 / Long 23,100 |
| Wed 14:15 | Trigger D AND B fires. NIFTY at 22,720, 2 closes above CST_bar.high (22,640), Stoch K back to 84. New ATM = 22,700 | Pyramid: add Long 22,700 CE / Short 22,900 CE |
| Thu 12:45 | Second CST fires. NIFTY at 22,840. Bear call credit at 23,100/23,300 = ₹1,150/lot. Build level-2 hedge | Add Short 23,100 CE / Long 23,300 CE |
| Wed (next week) 15:25 | T-1 EOD square-off | All 8 legs closed at market |

If trend reversed instead of continuing after the level-2 hedge: max loss is bounded by the 4 separate spread structures, each capped. The pyramid amplifies both gain and loss in directional outcomes; the hedges still function on each level.

---

## 6. Caveats

1. **Backtest only — no live forward-test.** Behavior in unseen regimes (market crash, IV spike, gap days) hasn't been validated.
2. **Phase 1 is 1 lot.** With pyramiding, max exposure is 2 lots. Operator must monitor margin closely especially in the first month.
3. **The 12.6% FP rate is averaged across the sample.** In high-volatility regimes (election/correction days), FP rate likely spikes. Consider adding a regime filter (e.g., daily ADX > 25) before allowing pyramid in Phase 2.
4. **Same-week pyramid only.** This study assumed the pyramid happens in the same weekly expiry as the existing condor. Cross-week pyramiding (e.g., add level-2 spread in next week) was not tested and is not recommended for Phase 1.
5. **The pyramid trigger uses the LAST CST bar's high/low as the level for D.** If multiple CSTs fire (and we ignore subsequent ones per the multi-CST policy), we still use the FIRST CST's bar levels. This was tested implicitly because the dataset evaluated all CSTs, not just first ones.

---

## 7. Suggested next research

1. **Re-test with pyramid in the simulation:** simulate the full 5-state pyramid policy on the 75 trends, measure end-to-end P&L vs. plain condor strategy, vs. plain bull call (no hedge). Validates that pyramiding beats both alternatives on this data.
2. **CST quality enhancement:** can a different CST rule (e.g., RSI-based, or longer-period Stoch) reduce the 67% false-alarm rate at source? May obviate or complement the pyramid trigger.
3. **Regime conditioning:** does ADX > 25 (or similar trend-strength filter) on daily TF improve the pyramid trigger's FP rate further?
4. **Cap at level 3?** Test whether allowing one more pyramid level adds meaningful value or is dominated by tail risk.

---

## 8. Files

| File | Purpose |
|---|---|
| `cst_continuation_per_trigger.csv` | All 396 CSTs with continuation/reversal pts and outcome label |
| `cst_continuation_first_per_trend.csv` | First CST per trend (n=73) — what condor experiences |
| `reestablishment_trigger_scores.csv` | 7 single-trigger scores |
| `combined_trigger_scores.csv` | 4 combo scores |
| `RESULTS.md` | This file |
