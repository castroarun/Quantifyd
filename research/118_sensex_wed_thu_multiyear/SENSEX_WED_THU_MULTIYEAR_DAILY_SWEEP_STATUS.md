# SENSEX Wednesday vs Thursday — Every Day Characterised, Then 5 Years of Price Action

STATUS: **RUNNING** (launched 2026-08-20 by the ops session; executed by a research agent)

## 2. The Ask

**What Arun asked (2026-08-20):** "look into all days not just 8-July. Also go beyond the
options data we have — study the price pattern across the years of 1-min price action we
have; you will understand the spikes, calm periods, price moves, ranges etc. Always first
study with existing options data, use the price action data for years beyond that as
additional data to arrive at a consensus. Do this for Wednesday and Thursday for SENSEX."

**Why this matters right now.** research/114 concluded Thursday should HOLD (+2,630/lot,
92% win) and its Wednesday companion concluded the opposite (HOLD −1,112/lot). But
Wednesday's entire verdict rests on **one day**: 2026-07-08 at −16,502/lot. Remove it and
holding wins Wednesday comfortably. A rule that expensive cannot rest on n=1.

> **The question: is Wednesday genuinely fatter-tailed than Thursday, or did we build a
> rule on one unlucky day?**

## 3. The Base — a two-stage design, options first

**Stage A — options data (the rupee truth, but short).**
`options_data.db :: option_chain`, 1-min, 2026-04-20 → 2026-08-20. 12 clean Wednesdays and
12 clean Thursdays already replayed in research/114. Characterise **every single day**, not
just the losers: entry credit, terminal move, max adverse excursion and WHEN it happened,
whether the day was a trend or a spike-and-revert, and the resulting P&L under HOLD.

**Stage B — price action (the frequency, and it is long).**
`market_data.db :: market_data_unified`, symbol `SENSEX`, timeframe `minute`,
**2021-01-01 → 2026-08-20, 508,378 rows** — roughly 280 Wednesdays and 280 Thursdays.
Options are not available for these years, so measure the *underlying* behaviour that
determines a short straddle's fate:
- intraday range (H−L)/open, and realised vol
- **terminal |move| from the 09:16 level** — what a full-day short straddle actually pays
- **maximum adverse excursion** from 09:16, and the minute it occurred
- the same from **13:00** (the afternoon-window construction)
- spike vs drift: was the damage one violent move or a sustained trend?
- calm/normal/stressed regime buckets (e.g. by trailing realised vol or India VIX where available)

**Stage C — consensus.** Do the 12 recorded options-Wednesdays look representative of the
280? Is the Wednesday tail a structural feature or a 2026 artefact? Report agreement AND
disagreement between the two datasets explicitly.

## 4. Method notes that must not be skipped

1. **VERIFY WHICH DAY WAS EXPIRY IN EACH ERA.** BSE has moved the SENSEX weekly expiry day
   over the years. Comparing "Wednesday vs Thursday" across 2021–2026 is meaningless if the
   expiry day itself moved — the effect we care about is **DTE0 vs DTE1**, not the calendar
   name. Establish the expiry-day history first (from the data: which weekday shows the
   characteristic expiry-day decay/pin behaviour, cross-checked against
   `option_chain.expiry_date` for the 2026 period) and **label days by DTE, not weekday**.
   State the mapping used. If it cannot be established for the early years, say so and
   restrict the claim.
2. **Loss proxy for the option-less years.** We cannot price options before 2026, so define
   a short-straddle loss day as **|terminal move| > credit**, using the credit range we
   actually observe in 2026 (full-day Wed credits ≈ 465–720 points; Thu ≈ 170–260 at 13:00).
   Test a ladder of credit assumptions rather than one, and report sensitivity.
3. **Controls (binding in this repo):** compare against all other weekdays as the baseline,
   not against zero. A "Wednesday is dangerous" claim must show Wednesday differs from
   Mon/Tue/Fri, not merely that markets sometimes move.
4. **Tail statistics, not just means:** p05, p01, max adverse excursion distribution, and the
   frequency of moves beyond 1x, 1.5x, 2x credit. The mean is what one day already broke.
5. **Regime split:** report with and without 2021–22 (a structurally higher-vol era) so a
   conclusion is not smuggled in from a different volatility regime.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-20 ~15:3x IST | Wednesday result rested on one day; Arun asked for the deeper cut | brief written, agent launched |

## 6. Crash Recovery

Read-only on both DBs. `market_data.db` is 30 GB — always filter by symbol AND timeframe
(a bare LIKE scan timed out earlier today). Scripts in `scripts/`, outputs in `results/`.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `SENSEX_WED_THU_MULTIYEAR_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | per-day characterisation + multi-year price study | yes |
| `results/*.csv` | per-day tables, distributions | yes |
| `results/RESULTS.md` | verdict + what it means for the deployed rules | yes |

## 8. Findings

(to be written by the research agent)
