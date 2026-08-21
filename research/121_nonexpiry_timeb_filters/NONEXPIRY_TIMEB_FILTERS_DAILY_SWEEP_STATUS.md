# Non-Expiry TimeB — Can a Regime Filter or a Tighter Stop Get Us to 1:2.5?

STATUS: **DONE** (2026-08-21) - verdict **NO EDGE on the filters / CONCLUDED on the target**: 1:2.5 is not reachable on non-expiry days by either route; the lever is SIZE, specifically Monday NIFTY. See section 8 and results/RESULTS.md.

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

**STATE: DONE.** All nine stages complete; verdict written to `results/RESULTS.md`.

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-21 ~14:0x IST | Non-expiry risk-reward shown to be 1:4–1:6; Arun asked for filters | brief written, agent launched |
| 2026-08-21 ~14:1x IST | Scope extended by the ops session | India VIX (level + shock), PCR (OI/volume), OI/ΔOI/walls/max-pain added to the pre-registered family; research/115's three established findings taken as given |
| 2026-08-21 | S1 day-level regime features built | NIFTY50 daily 3,875 rows 2011→2026; SENSEX daily 1,359 rows resampled from `minute` 2021→2026; INDIAVIX 2,882 |
| 2026-08-21 | S2 long-sample window outcomes | 4,068 SENSEX 1-min window-days + 8,279 NIFTY50 5-min window-days across the three window shapes |
| 2026-08-21 | S3 long-sample fit | 576 Spearman tests, 2,656 quintile rows, 7,308 skip rules each vs a 2,000-draw random-skip null |
| 2026-08-21 | **Key finding surfaced mid-run** | every signal predicts the RAW move; almost none survives dividing by the VIX-implied sigma (`vix_open` +0.50 → +0.07). The option market already prices the regime. |
| 2026-08-21 | S4 options-sample replay | 243 cell-days on the real 1-min chain; **harness validated — the Friday cell reproduces research/120 to the rupee** (+399/lot mean, worst −344/lot, 13/14) |
| 2026-08-21 | **Baseline discrepancy found** | the brief's max losses (−₹23.6k/−₹24.2k/−₹34.2k) are not reproducible on the live weekday; they look like an all-weekday pooling that mixes DTE0/far-DTE into a one-weekday cell. Own baseline substituted and the reconciliation reported in RESULTS §3. |
| 2026-08-21 | S5 stop ladder + move→premium map | SL20 fires on 0.2–0.6 % of long-sample days; the stop is decorative. 1:2.5 needs ~6 % (Wed) / ~3.5 % (Fri) / **unreachable** (Mon). |
| 2026-08-21 | S6 rupee confirmation + placebos | 540 skip rules vs an EXACT random-skip null: **10 winners against ~27 expected by chance, one of which is `placebo_noise`** |
| 2026-08-21 | S7 robustness | resolution-equivalence PROVED (5-min ≡ 1-min for max-excursion, 0 rows differing over 4,068 window-days); monotonicity curves; 6 combinations; size arithmetic |
| 2026-08-21 | S8 true tail (modelled) | recorded max excursion only reaches the long-run p90; true R:R 1:12.8–14.5 (Mon), 1:4.6–5.8 (Wed), 1:6.1–7.3 (Fri) |
| 2026-08-21 | **DATA TRAP found** | INDIAVIX **daily** bars carry `open(d)==close(d−1)` on **82.5 %** of rows — the overnight VIX shock is structurally zero there. Would have produced a false null. Rebuilt from the 5-minute series (4.1 % degenerate) in S9. |
| 2026-08-21 | S9 VIX family re-tested properly | shock has a real top-5 % tail effect (+22 % normalised) that is inert as a skip rule: over 36 shock rules the **maximum is never removed** |
| 2026-08-21 | STATUS → DONE, RESULTS.md written, INDEX.md row added, committed | verdict **NO EDGE / CONCLUDED — cut size, specifically Monday NIFTY** |

## 6. Crash Recovery

Everything is complete; nothing is running. To reproduce from scratch on the VPS:

```
cd /home/arun/quantifyd
nice -n 10 python3      research/121_nonexpiry_timeb_filters/scripts/s1_daily_features.py
nice -n 10 python3      research/121_nonexpiry_timeb_filters/scripts/s2_window_outcomes.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s3_longfit.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s4_options_sample.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s5_stops_and_tail.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s6_filters_confirm.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s7_robustness.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s8_true_tail.py
nice -n 10 venv/bin/python3 research/121_nonexpiry_timeb_filters/scripts/s9_vix_shock.py
```

S1→S2 must run before S3/S7/S8/S9; S4 must run before S5/S6/S8. Total wall time ≈ 12 min.
Both databases are opened **read-only**; nothing in `services/`, `app.py`, `frontend/` or the
live config was touched, and no restart was performed. Random seed 20260821 throughout.

Two data rules that must survive into any re-run:
- reject any recorded day with **< 50 distinct underlying prints** (frozen chain on an exchange
  holiday — 2026-05-01, **2026-05-28**, 2026-06-26);
- never take an overnight VIX change from `INDIAVIX` **daily** bars (see §8).

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `NONEXPIRY_TIMEB_FILTERS_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/s1_daily_features.py` … `s9_vix_shock.py` | the nine stages | yes |
| `results/RESULTS.md` | verdict + recommendation | yes |
| `results/daily_features_{NIFTY50,SENSEX}.csv` | day-level regime features | yes (0.8 MB) |
| `results/window_outcomes_*.csv` | long-sample window excursions (3 series) | yes (3.2 MB) |
| `results/longfit_{spearman,quintiles,skiprules}.csv` | the long-sample fit, reported whole | yes (1.5 MB) |
| `results/options_sample.csv` | per-day rupee replay incl. the whole stop ladder | yes |
| `results/prem_vs_move.csv` | the move→premium calibration pairs | yes |
| `results/{stop_ladder,tail_translation,true_tail}.csv` | Route B + the tail | yes |
| `results/{filter_confirm_pooled,filter_confirm_skip,combination_filters}.csv` | Route A confirmation | yes |
| `results/vix_{features_5min,family_ic,skiprules}.csv` | the rebuilt VIX family | yes |
| `results/*_report.txt` | printed transcripts of each stage | yes |

Total 5.8 MB — everything is committed; nothing needed gitignoring.

## 8. Findings

**VERDICT: NO EDGE on the filters — CONCLUDED on the target. A 1:2.5 reward:risk is not
reachable on non-expiry days by either route. The honest lever is SIZE, and the specific cut is
Monday NIFTY.** Full write-up in `results/RESULTS.md`.

**1 — The target is arithmetically out of reach, not merely hard.** Each cell sells 0.75–1.10 %
of spot and keeps 0.83 % / 2.56 % / 3.33 % of that credit as its median day. A combined-% stop
caps the loss at credit × stop, so 1:2.5 requires ~6 % of credit on SENSEX Wednesday, ~3.5 % on
NIFTY Friday, and **below 2 % — impossible — on NIFTY Monday**, where the ₹250/lot round trip is
itself 20 % of the median profit. On the long sample those stops fire on 17.7 %, ~45 % and 84 %
of days. Tightening also made the **worst day worse** on both cells where it engaged (Fri SL6
−₹3,440 → −₹15,210; Wed SL8 −₹11,440 → −₹15,620) — the third independent reproduction of
research/114 and research/116.

**2 — Every candidate signal is a volatility proxy, and the premium already prices it.** Raw
excursion is well predicted (Spearman up to +0.56), but divide the move by the day's VIX-implied
sigma — *was it big for what the market charged?* — and it collapses: `vix_open` +0.50 → +0.07,
`atr14` +0.48 → +0.15, `pdr_pct` +0.43 → +0.21. Skipping high-VIX days is **worse than random**
on the premium-relative outcome (rand-percentile 67–97 on all six cell × series combinations).
This is research/120's calm-window inversion moved from the time-of-day axis to the
day-selection axis, and research/115's `atm_iv` trap in a new costume.

**3 — Per-signal answers to what Arun named.** *Daily CPR*: research/67's sign confirmed
(+0.12…+0.31 raw) but flat across Q1–Q4 with a Q5-only jump, and ≈0 normalised. *Weekly CPR*:
**the r/67 sign flip does NOT reproduce** — normalised Q5/Q1 is 0.80 on SENSEX Wednesday (r/67's
sign) but 1.12 on NIFTY Monday and Friday (opposite). A weekly-CPR gate would be wrong on two
cells of three; do not build one. *Gap*: |gap| is the weakest vol proxy; the signed gap is the
more interesting read (down-gaps → larger normalised moves, −0.07…−0.27) but that is direction,
not risk. *Prev-day range*: real raw, mostly priced away normalised. *VIX shock* (rebuilt): a
genuine top-5 % tail effect (+22 % normalised, monotone in both % and points) that is **inert as
a skip rule — over 36 shock rules the maximum is never removed**. *PCR/OI*: complete coverage,
best IC in the study (`d_oi_atm_pct` +0.42 vs the drawdown path) but ≈0 against booked P&L and
the **worst** rule on Wednesday in the skip test; with no long sample it is untestable rather
than disproven.

**4 — The one partial survivor is `pre_range_bp` and it does not solve the problem.** The
session's own 09:15→entry range is the only signal with a monotone, sign-consistent,
premium-relative gradient (Q5/Q1 1.38–1.79) that beats the random-skip null on all six
combinations. Skipping its top quintile shaves **3–12 % off the p90 and leaves the maximum
untouched in four of six cases**, at the cost of a fifth of the trading days. The bad day is not
the hot-open day.

**5 — The placebo ends Route A.** 540 skip rules vs an exact random-skip null produced **10**
apparent winners against **~27 expected by chance** — fewer than noise — and one of the ten is
`placebo_noise`, a Gaussian random number, "beating 97 % of random skips" and "retaining 137 % of
P&L" on SENSEX Wednesday. ~11,550 cells were evaluated in total; nothing cleared all three
acceptance criteria (monotone curve **and** beats random **and** improves total P&L).

**6 — Neither sample's "worst day" should be used for sizing.** The recorded maximum excursion
(39–52 bp) barely reaches the long-run **p90**; the long-run maximum is 97–1,304 bp. Modelled on
the long sample, true R:R is **1:12.8–14.5 (Mon)**, 1:4.6–5.8 (Wed), 1:6.1–7.3 (Fri) — so the
brief's 1:6.1/1:4.3/1:6.2 is about right for Wed and Fri and far too kind for Monday.

**7 — Recommendation: change no rule; cut Monday NIFTY DTE1 from 8 lots to 3, or drop it.**
It is the only cell where the target is impossible rather than expensive; it has the worst true
R:R, the thinnest median (0.83 % of credit), and the highest cost drag (round trip = 20 % of
median profit), and it earns ₹998/day at 10 lots against Wednesday's ₹1,922 and Friday's ₹3,994.
R:R is invariant to size by construction, so size is the only dial that reduces how much a bad
day hurts without reducing expectancy per rupee of margin. **R:R is also the wrong success
metric for a short-premium window** — these cells win 71/71/93 % with positive expectancy;
demanding 1:2.5 of them is demanding an underwriter stop collecting small premiums against rare
large claims.

**8 — Two data findings worth carrying forward.**
*(a)* `INDIAVIX` **daily** bars carry `open(d)==close(d−1)` on **82.5 %** of rows — never compute
an overnight VIX change from them; use the 5-minute series (4.1 % degenerate).
*(b)* For the **maximum excursion inside a fixed window**, 5-minute bars are **exactly** equal to
1-minute bars (0 rows differing over 4,068 SENSEX window-days). The no-5-min rule bites on the
*path*, not on this statistic — which is what makes the NIFTY 5-minute long sample admissible
here.

**Next levers.** `pre_range_bp` is real but too small to be a *skip* rule — it might work as a
**sizing** rule (half-size on a hot morning), which captures the same 3–12 % without forfeiting a
day's premium. That is a cheaper, different test. Re-run §3/§5 after 2026-11 with ~28 days per
cell. And `CSL_TIMEB2_NIFTY`'s 2026-09-05 review should carry this answer for its DTE0
13:00–14:00 cell, which is the same shape as MON_NIFTY_DTE1.
