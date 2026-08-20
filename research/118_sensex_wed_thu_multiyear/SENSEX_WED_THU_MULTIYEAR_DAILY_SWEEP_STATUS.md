# SENSEX Wednesday vs Thursday — Every Day Characterised, Then 5 Years of Price Action

STATUS: **DONE** (2026-08-20). Verdict: **the Wednesday rule does not replicate and should be reverted; the Thursday HOLD rule stands but its risk picture was wrong by two orders of magnitude.** Full write-up in `results/RESULTS.md`.

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

**Final state: DONE.** Three datasets built and cross-validated; verdict written to
`results/RESULTS.md`; research index updated.

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-20 ~15:3x IST | Wednesday result rested on one day; Arun asked for the deeper cut | brief written, agent launched |
| 2026-08-20 ~17:5x IST | Stage A built | 1-min chain, 55 days (2026-06-03 .. 08-20), all five weekdays not just Wed/Thu |
| 2026-08-20 ~17:5x IST | Stage B built | SENSEX 1-min index, 1,354 days, 2021-01-01 .. 2026-08-20 |
| 2026-08-20 ~18:0x IST | **`bse_options_bhav` discovered** | real BSE daily option prices for SENSEX 2024-01-01 .. 2026-08-04, 289,859 rows — never used in any prior study. Stage A2 added on the spot: 618 days of REAL options instead of 55 |
| 2026-08-20 ~18:0x IST | **Expiry-day history derived from data** | weekly SENSEX expiry: **Friday 2024 -> Tuesday 2025 Jan-Aug -> Thursday 2025 Sep onward**. The brief's warning was correct; this became the study's natural experiment |
| 2026-08-20 ~18:1x IST | **Data corruption caught** | on rows where `trade_date == expiry_date` the BSE file overwrites `close`/`settle`/`underlying` with the settlement index. Naive use produced -Rs3,000,000/lot. DTE0 re-settled at intrinsic vs the 15:15 index; OI filter restricted to DTE>0 (OI legitimately goes to 0 on 2024 expiry rows and was silently deleting every expiry day) |
| 2026-08-20 ~18:2x IST | Stage A2 rebuilt clean | 618 days kept: DTE0 n=127, DTE1 n=123, ~110-131 per weekday |
| 2026-08-20 ~18:3x IST | Cross-validation passed | bhavcopy proxy vs 1-min truth on 38 overlapping days: corr **0.938**, sign agreement 95%, measured optimism **+Rs1,281/lot** (64 pts) — haircut applied to every A2 conclusion |
| 2026-08-20 ~18:4x IST | Analysis complete, RESULTS.md written | verdict below |

### Live findings as they emerged

- Stage A reproduced research/114 exactly, and showed Wednesday is the **only** weekday whose
  sign flips on leave-one-out (-1,112 -> **+287** without 2026-07-08).
- Stage B then contradicted the premise outright: across 1,354 days Wednesday has the **lowest**
  mean move, lowest p90/p95, and lowest max adverse excursion of any weekday. Thursday has the
  highest.
- Stage A2 settled it: over 125 real-option Wednesdays, Wednesday has the best win rate, the
  lowest standard deviation and **1 catastrophic day in 125 (0.8%)** against Friday's 4.0%.
- The tail turned out to live on **DTE0**: 8.7% of expiry days lost >500 points vs 3.3% on DTE1.

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

**VERDICT: NO EDGE in the Wednesday rule — research/114's Wednesday finding does not replicate
and should be reverted. The Thursday HOLD rule survives on expectancy but its risk picture was
badly wrong.**

### The headline

Wednesday is not the dangerous day. Over 5.6 years of 1-minute price action it is the
**calmest weekday of the SENSEX week** (lowest mean move, lowest p90/p95, lowest max adverse
excursion; calmest or second-calmest in five of six years and never the wildest). Over 2.6
years of **real BSE option prices** it has the **fewest catastrophic days of any weekday — 1 in
125 (0.8%)** against Friday 4.0%, Tuesday 3.8%, Monday 3.1%, Thursday 2.7%. That one day is
**2026-07-08**, the day research/114 built its rule on.

The fat tail is real but belongs to **DTE0 — expiry day, i.e. Thursday**: 8.7% of DTE0 days
lost more than 500 points against 3.3% on DTE1 and ~1% on DTE2+, and a DTE0 straddle closes
beyond its full credit **nine times as often** as a DTE1 one. **The deployed configuration
removes the stop on the fat-tailed day and keeps it on the thinnest-tailed day.**

### The single strongest piece of evidence

The catastrophic Wednesday of 2026-07-08 is *inside* the larger sample, contributing -999.5
points. In the same Thursday-expiry regime, across **46 Wednesdays**, the bucket still earns
**+105.7 points/day at an 80% win rate**. Twelve observations were not enough to see the
distribution; forty-six are.

### What the numbers say, restated on the live construction's terms

(A2 means less the measured Rs1,281/lot proxy optimism)

| bucket | n | adjusted Rs/lot/day | win% | worst |
|---|---|---|---|---|
| Wed & DTE1 (today's rule) | 46 | **-7** | 76 | -999.5 pts |
| Thu & DTE0 (today's rule) | 41 | **+1,105** | 71 | -1,003 pts |
| Wed (all) | 125 | +178 | 77 | -999.5 pts |
| Fri (all) | 124 | **-654** | 62 | -1,186 pts |

Held all day, a SENSEX ATM straddle is break-even to modestly positive on every weekday except
Friday. Wednesday-at-DTE1 is a coin flip around zero -- not a Rs1,112/lot loser.

### research/114 next to research/118

| rule | r114 (n=12, one quarter) | r118 (2.6 years, real options) |
|---|---|---|
| Wednesday HOLD | -Rs1,112, 67% win, worst -Rs16,502 | **-Rs7**, 65% win (n=46) |
| Thursday HOLD | +Rs2,709, 92% win, **worst -Rs127** | +Rs1,105, 68% win, **worst ~-Rs21,500** (n=41) |

### Recommendation

1. **Wednesday rule -> revert the justification, but do not edit an engine on this evidence.**
   The premise is void. research/114's own table already showed the retained per-leg 30% stop
   losing money on Wednesday (-Rs412/lot, 25% win, 6th of 17 variants) -- it only beat HOLD
   *because of* 2026-07-08. This study cannot price the stop over 2024-2026 (no intraday option
   data before 2026-04), so the correct next step is a **dedicated G2 study of the per-leg 30%
   stop on Wednesday using the 1-minute chain**, now that more Wednesdays have accumulated.
2. **Thursday rule -> keep `leg_sl_disabled_dtes: (0,)`, correct the risk assumption.** Positive
   expectancy in every independent cut. But the config comment's "92% win, worst -127" is a
   12-day artefact; the truth over 127 DTE0 days is **34% losers, 8.7% worse than -500 points,
   worst near -Rs21,500/lot**. Holding unstopped on DTE0 is an explicit decision to accept a
   gamma tail and must be **sized for a -Rs21,500/lot day**. Agrees with research/103: the real
   DTE0 lever is sizing.
3. **Friday is the weekday that actually looks bad** (adjusted -Rs654/lot, 62% win, 4.0%
   catastrophic days; worst weekday in the current era). Flagged as a lever, subject to the
   caveat that Friday's measured proxy bias was near zero and so its adjusted figure is the
   least trustworthy number in the study.

### Process lessons (both binding candidates)

- **A weekday rule on SENSEX must be validated against the expiry-era history.** BSE moved the
  SENSEX weekly expiry twice inside our own data (Friday -> Tuesday -> Thursday). A study that
  says "Wednesday" without saying which DTE that was in each era is comparing three different
  instruments.
- **`bse_options_bhav` must be settled at intrinsic on expiry days.** The file overwrites
  `close`/`settle`/`underlying` with the settlement index where `trade_date == expiry_date`;
  naive use gives -Rs3,000,000/lot. And OI filters must be restricted to DTE>0, or every expiry
  day -- exactly the days carrying the tail -- is silently deleted.
- This is the **second** time the same 2026 quarter has produced a false SENSEX-Wednesday alarm
  (research/104 at n=15, research/114 at n=12). A live rule set from n=12 inside one quarter is
  a rule set from noise.

Full write-up, all tables, sins accounting and honest caveats: **`results/RESULTS.md`**.
