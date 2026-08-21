# Non-Expiry TimeB — Can a Regime Filter or a Tighter Stop Get Us to 1:2.5?

STATUS: **RUNNING** (launched 2026-08-21 by the ops session; executed by a research agent)

## 2. The Ask

**What Arun asked (2026-08-21):** after seeing that the non-expiry TimeB windows risk
₹23–34k to typically make ₹3.9–5.5k — *"can we work on limiting the losses or aiming at 1:2.5
max on non-expiry days? Maybe that day's CPR width, previous day's CPR, that week's and/or
previous week's CPR width, gap ups/downs on that day, previous day's range or so — can any of
this help improve the probability?"*

**The measured problem (research/120 + the window-decay cut, all recorded days, 10 lots):**

| Day | Window | Typical profit (median) | Max loss | Reward : Risk |
|---|---|---|---|---|
| Mon NIFTY DTE1 | 13:00–14:00 | +₹3,883 (3.0% of credit) | −₹23,645 | 1 : 6.1 |
| Wed SENSEX DTE1 | 10:30–12:00 | +₹5,600 (4.7%) | −₹24,205 | 1 : 4.3 |
| Fri NIFTY DTE2 | 10:00–12:00 | +₹5,508 (4.0%) | −₹34,193 | 1 : 6.2 |

Expiry days (Tue 19.1%, Thu 35.5% of credit) are NOT in scope — they already earn their size.

**Target:** get non-expiry days to **≤ 1:2.5** — i.e. max loss ≈ ₹10–14k against the same
typical profit — **without destroying expectancy.** Two routes, both to be tested:

- **A · Condition the entry** — skip or downsize the day when a regime signal says a big move
  is likely. Candidates: today's CPR width, previous day's CPR width, this week's and previous
  week's CPR width, opening gap (up/down, magnitude), previous day's range (and range vs its
  own recent average), and any combination.
- **B · Tighten the stop** — a ladder from the deployed 20% down (15 / 12 / 10 / 8 / 6%), plus
  a rupee-cap variant sized to the 1:2.5 target.

## 3. Prior work that constrains this (read before designing)

- **research/67 — CPR daily vs weekly SIGN FLIP.** A *narrow weekly* CPR precedes trend; a
  *narrow daily* CPR precedes calm. They point opposite ways. Any filter that treats "narrow
  CPR" as one signal will be wrong half the time. The live CPR gate is already signed correctly.
- **research/114 / 116** — on expiry day every stop tested LOST to holding, and every ratchet
  made give-back worse. Do not assume tightening works; it must be shown, per window.
- **research/115** — a spectacular raw table there was pure artefact. Controls are mandatory.
- **research/120** — the calm/decay inversion: on Fridays the calmest windows LOSE and the
  dangerous ones EARN (Spearman +0.31, p=0.0011). **A filter that simply avoids volatility may
  therefore avoid the profit too.** This is the central risk of route A and must be tested head-on.

## 4. The methodology that makes this credible

**The binding constraint is sample size: n≈16 days per window.** Fitting filters directly on
16 days will manufacture a winner. So:

1. **Fit the conditioning relationship on the LONG sample, not the options sample.**
   `market_data.db :: market_data_unified`, SENSEX `minute` 2021-01-01 → 2026-08-21 (~1,350
   days) and daily OHLC for both venues for CPR/range/gap construction. Question: *does the
   signal predict the size of the subsequent intraday move in that window?* That is a
   many-hundred-day question and can be answered honestly.
   **Note: there is NO NIFTY 1-minute series (5-min only, ends 2026-07-17)** — state how NIFTY
   intraday is handled, or restrict NIFTY claims to daily-resolution evidence.
2. **Then apply the filter to the options sample as confirmation, not as the fitting set.**
   Report how the ≈16 days split and accept that this is corroboration with wide error bars.
3. **Controls (binding):** a filter that skips k% of days must beat *randomly* skipping k% of
   days. Report that comparison for every accepted filter.
4. **Monotonicity:** a real threshold effect strengthens smoothly; report the full response
   curve, not the best cut.
5. **Multiple testing:** pre-register the filter list and thresholds; report the count tried;
   haircut accordingly.
6. **Cost of skipping:** every skipped day forgoes its typical profit. Report net effect on
   total P&L, not just on the tail.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-21 ~14:0x IST | Non-expiry risk-reward shown to be 1:4–1:6; Arun asked for filters | brief written, agent launched |

## 6. Crash Recovery

Read-only on both DBs. `market_data.db` is 30 GB — always filter by symbol AND timeframe.
Reject frozen-chain holidays (<50 distinct spot prints — research/120's trap). Scripts in
`scripts/`, outputs in `results/`.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `NONEXPIRY_TIMEB_FILTERS_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | signal build, long-sample fit, options confirmation, stop ladder | yes |
| `results/*.csv` | response curves, per-day tables | yes |
| `results/RESULTS.md` | verdict + recommendation | yes |

## 8. Findings

(to be written by the research agent)
