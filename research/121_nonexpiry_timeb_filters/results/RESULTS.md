# research/121 — Non-expiry TimeB: can a regime filter or a tighter stop get us to 1:2.5?

## VERDICT: **NO EDGE on the filters — CONCLUDED on the target. 1:2.5 is not reachable on non-expiry days by either route. The honest lever is SIZE, and the specific cut is Monday NIFTY.**

Three findings, in order of how much they should change what we do.

**1 — A 1:2.5 reward:risk is not a property these windows can have.** It is arithmetic, not
opinion. Each cell sells 0.75–1.10 % of spot in premium and keeps **0.83 % / 2.56 % / 3.33 %
of that credit** as its median day. A combined-% stop caps the loss at *that same credit times
the stop*. To make the cap 2.5× the median profit you need a stop of roughly **6 % of credit on
SENSEX Wednesday, 3.5 % on NIFTY Friday, and something below 2 % — i.e. impossible — on NIFTY
Monday**, where the ₹250/lot round-trip cost is *itself* 20 % of the median profit. On the long
sample those stops fire on **17.7 %, 57.5 % and 84 %** of days respectively. You do not get to
1:2.5; you get to a coin-flip that pays the round trip every time.

**2 — Nothing conditions the entry.** Every candidate signal — daily CPR, weekly CPR, gap,
previous-day range, ATR, VIX level, VIX shock, PCR, ΔOI, OI walls, max-pain — predicts the
**raw** size of the move inside the window, some of them strongly (Spearman up to **+0.56**).
Almost none of it survives asking the only question that matters: *was the move big **relative to
the premium the market charged for it***. Divide the excursion by the VIX-implied sigma and
`vix_open`'s +0.50 becomes **+0.07**; `atr14` +0.48 → +0.15; `pdr_pct` +0.43 → +0.21. **The
option market has already priced the regime you were going to filter on.** This is
research/120's Friday result — calm windows do not pay — generalised from the time-of-day axis
to the day-selection axis, and it is the reason Route A cannot work.

**3 — The controls caught exactly what research/115 said they would.** 540 skip rules were
scored against an *exact* random-skip null. **10** beat the 95th percentile while retaining
≥100 % of P&L — against **~27 expected by chance alone**. Fewer winners than noise. And one of
the ten is **`placebo_noise` — a Gaussian random number with no information in it whatsoever —
which "beat 97 % of random skips" and "retained 137 % of P&L" on SENSEX Wednesday.** That single
row is the study. Any table of "best filters" built on 14–17 days manufactures results of that
quality on demand.

> **Recommendation (§8): change no rule. Cut Monday NIFTY DTE1 — from 8 lots to 3, or drop it.**
> It is the cell where the target is arithmetically unreachable, it has the worst true R:R
> (**1 : 12.8–14.5**), the thinnest median (0.83 % of credit against 2.56 % and 3.33 %), and it
> earns ₹998/day at 10 lots against Wednesday's ₹1,922 and Friday's ₹3,994. Size is the only
> lever that reduces the rupee tail without reducing expectancy per rupee of margin.

---

## 1. What was asked, and what was actually tested

> **Arun (2026-08-21):** *"can we work on limiting the losses or aiming at 1:2.5 max on
> non-expiry days? Maybe that day's CPR width, previous day's CPR, that week's and/or previous
> week's CPR width, gap ups/downs on that day, previous day's range or so — can any of this help
> improve the probability?"* — extended mid-study to India VIX (level and shock), PCR (OI and
> volume), and OI / ΔOI / walls / max-pain.

The three live non-expiry cells, read from the frozen config
(`backtest_data/csl_paper_config.json`, refrozen 2026-08-20), not from memory:

| Cell | Book | Window | Stop |
|---|---|---|---|
| **MON_NIFTY_DTE1** | `CSL_TIMEB_NIFTY` DTE1 | 13:00 – 14:00 | combined SL 20 % |
| **WED_SENSEX_DTE1** | `CSL_TIMEB_SENSEX` DTE1 | 10:30 – 12:00 | combined SL 20 % |
| **FRI_NIFTY_DTE2** | `CSL_TIMEB_NIFTY` DTE2 | 10:00 – 12:00 | combined SL 20 % |

Two routes, both pre-registered before any run:

- **A · condition the entry** — 16 long-fittable day signals + 12 option-book signals + 2
  placebos, each swept across its whole response curve, plus 6 pre-registered combinations.
- **B · tighten the stop** — percent ladder 25/20/15/12/10/8/6 and a rupee-cap ladder
  2500/2000/1600/1400/1200/1000/800 per lot.

**The methodology that makes it credible.** The binding constraint is n ≈ 16 days per cell.
Fitting on 16 days manufactures winners, so the conditioning relationship is fitted on the
**long sample** (274–557 comparable days per window) against a question the long sample can
actually answer — *does the signal predict the size of the move inside that window?* — and the
options sample is used **only for confirmation in rupees**, never for selection.

---

## 2. Data, and the two traps found on the way in

| Stage | Source | Coverage | Role |
|---|---|---|---|
| Long | `market_data.db :: market_data_unified` **SENSEX `minute`** | 1,359 days, 2021-01-01 → 2026-08-20 | 1-minute truth |
| Long | same, **NIFTY50 `5minute`** | 2,762 days, 2015-02-02 → 2026-07-17 | NIFTY clock |
| Long | same, **INDIAVIX `5minute`** | 2,825 days, 2015-02-02 → 2026-07-17 | the VIX family |
| Rupee | `backtest_data/options_data.db :: option_chain`, 1-min | 84 NIFTY / 82 SENSEX usable days, 2026-04-20 → 2026-08-20 | rupee truth |

Both databases opened **read-only**. Holiday guard from research/120 applied (reject any day
with < 50 distinct underlying prints): it removed **2026-05-01, 2026-05-28 and 2026-06-26** —
05-28 is a *new* find, not in research/120's list. Today (2026-08-21) excluded as a partial
session.

### Trap 1 — the INDIAVIX **daily** series cannot measure an overnight shock

`market_data_unified` INDIAVIX daily bars carry **`open(d) == close(d−1)` on 2,800 of 3,395 bars
(82.5 %)**. The daily series is derived, not a real auction print. Computing "overnight VIX
change" from it yields **identically zero on four days in five** — which would have produced a
confident, completely false null ("VIX shocks predict nothing"). The whole VIX family was
rebuilt from the **5-minute** series, where `open(d) == close(d−1)` on only **4.1 %** of days
(`scripts/s9_vix_shock.py`). Every VIX number in this study comes from the rebuilt series.

### Trap 2 (a licence, not a trap) — 5-minute bars are *exact* for this statistic

The project rule bans 5-minute data in options backtests because it understates intraday
extremes. That rule is about the **path** — which minute a stop fires. The statistic this study
fits on is the **maximum excursion inside a fixed window**, and a window's high is the max of
its bar highs *at any resolution*. Proved rather than assumed: resampling the SENSEX 1-minute
series to 5 minutes over the same **4,068 window-days** gives `mean(exc₅ − exc₁) = 0.0000 bp`,
`max|diff| = 0.0000 bp`, **0 rows differing**. That is what licenses the NIFTY 5-minute long
sample here. The path work — the stop ladder — is done on the real 1-minute option chain.

### Costs
NIFTY 0.5 pt/leg-side × 65 + ₹30/leg-side = **₹250/lot** round trip; SENSEX 1.0 pt × 20 + ₹30 =
**₹200/lot**. Charged on every trade. Cost is a per-trade constant, so 2× sensitivity is exact
arithmetic.

---

## 3. The baseline, re-derived — and a correction to the brief

The brief's problem table could **not** be reproduced from the recorded chain. Replaying each
cell on its own live weekday, net of cost, at 10 lots:

| Cell | n | period | credit | median | mean | worst | win % | median as % of credit |
|---|---|---|---|---|---|---|---|---|
| MON_NIFTY_DTE1 | 17 | 04-27 → 08-17 | 180.7 pt (0.75 % of spot) | **+₹1,240** | +₹998 | **−₹4,840** | 71 % | 0.83 % |
| WED_SENSEX_DTE1 | 17 | 04-29 → 08-19 | 607.1 pt (0.79 %) | **+₹3,830** | +₹1,922 | **−₹11,440** | 71 % | 3.33 % |
| FRI_NIFTY_DTE2 | 14 | 05-08 → 08-14 | 264.9 pt (1.10 %) | **+₹3,575** | +₹3,994 | **−₹3,440** | 93 % | 2.56 % |

**The harness is right:** the Friday cell reproduces research/120 to the rupee — mean **+₹399/lot**
(r/120: +400), worst **−₹344/lot** (r/120: −344), 13 wins in 14.

The brief quoted medians of +₹3,883 / +₹5,600 / +₹5,508 and worsts of **−₹23,645 / −₹24,205 /
−₹34,193**. Those losses are 2–10× larger than anything the live weekday produced. The most
likely origin is **pooling all weekdays through the same clock window**, which mixes expiry-day
and far-DTE sessions into a cell that only ever trades one weekday:

| Same window, ALL recorded weekdays | worst ₹ at 10 lots by DTE |
|---|---|
| MON window 13:00–14:00 (NIFTY) | DTE0 −17,090 · **DTE1 −4,840** · DTE4 −38,450 · DTE6 −44,910 |
| WED window 10:30–12:00 (SENSEX) | DTE0 −24,610 · **DTE1 −11,440** · DTE2 −27,910 · DTE3 −8,260 |
| FRI window 10:00–12:00 (NIFTY) | DTE0 −23,170 · DTE1 −39,780 · **DTE4(=Fri) −3,440** · DTE5 −37,080 |

This is worth stating plainly because it cuts both ways: **the stated problem is partly an
artefact of pooling — and the observed tail is nonetheless far too small to trust**, for the
reason in §7. Do not use either number for sizing.

---

## 4. Route A — the long-sample fit (274–557 days per window)

Spearman correlation of each signal with the window's maximum excursion. `exc_bp` is the raw
move; `exc_norm` is the same move **divided by the day's VIX-implied 1-day sigma** — i.e. how
big it was *for what the option market charged*. Live weekday only; both index clocks shown.

| signal | exc_bp (SX 1-min, Mon/Wed/Fri) | exc_bp (NF 5-min) | **exc_norm (SX)** | **exc_norm (NF)** |
|---|---|---|---|---|
| `pre_range_bp` (session range to entry) | +0.56 / +0.42 / +0.38 | +0.56 / +0.46 / +0.40 | **+0.35 / +0.22 / +0.17** | **+0.37 / +0.27 / +0.22** |
| `vix_open` | +0.45 / +0.31 / +0.29 | +0.50 / +0.39 / +0.36 | +0.05 / −0.13 / −0.16 | +0.07 / −0.02 / −0.08 |
| `atr14_pct` | +0.42 / +0.28 / +0.21 | +0.48 / +0.38 / +0.37 | +0.10 / −0.09 / −0.15 | +0.15 / +0.04 / +0.02 |
| `pdr_pct` (prev-day range) | +0.43 / +0.32 / +0.28 | +0.40 / +0.43 / +0.35 | +0.26 / +0.05 / +0.07 | +0.21 / +0.18 / +0.13 |
| `pdr_rel` (range ÷ 20-day mean) | +0.22 / +0.20 / +0.17 | +0.12 / +0.23 / +0.12 | +0.24 / +0.11 / +0.19 | +0.16 / +0.19 / +0.15 |
| `cpr_today` (daily CPR width) | +0.31 / +0.23 / +0.12 | +0.25 / +0.30 / +0.19 | +0.24 / +0.03 / +0.03 | +0.16 / +0.14 / +0.08 |
| `wcpr_this` (weekly CPR width) | +0.17 / +0.01 / +0.18 | +0.22 / +0.15 / +0.17 | +0.09 / −0.11 / +0.07 | +0.10 / +0.03 / +0.03 |
| `gap_abs` | +0.19 / +0.10 / +0.09 | +0.23 / +0.13 / +0.10 | +0.07 / −0.07 / −0.05 | +0.07 / −0.00 / −0.04 |
| `gap_pct` (signed) | −0.24 / −0.02 / −0.01 | −0.10 / −0.02 / −0.06 | **−0.27 / −0.11 / −0.07** | −0.14 / −0.09 / −0.15 |

**Read the last two columns.** The whole predictive content of the volatility family —
VIX level, ATR, previous-day range, daily CPR — is *the level of volatility*, which is exactly
what the straddle's price already encodes. Normalise by it and the signal is gone. `vix_open`
goes from the second-strongest raw predictor to **flat or inverted**.

### The two partial survivors, and their full response curves

Reported as complete quintile curves, never a chosen cut (`results/longfit_quintiles.csv`):

**`pre_range_bp` — the session's own realised range from 09:15 to the window start.** The only
signal with a monotone, sign-consistent, premium-relative gradient:

| cell | exc_norm Q1 → Q5 | Q5/Q1 |
|---|---|---|
| MON (NF 5-min) | 0.19 → 0.23 → 0.27 → 0.28 → 0.34 | **1.79** |
| WED (SX 1-min) | 0.29 → 0.29 → 0.30 → 0.31 → 0.40 | **1.38** |
| FRI (NF 5-min) | 0.36 → 0.36 → 0.45 → 0.40 → 0.52 | **1.44** |

It also beats the random-skip null cleanly: skipping its top quintile puts the kept days' p90 in
the **bottom 0.0–3.9 %** of 2,000 equal-sized random skips, on all six cell × series
combinations. **And it still does not solve anything** — see the maximum column:

| cell / series | p90 exc_norm all → kept | **max all → kept** |
|---|---|---|
| MON SX 1-min | 0.42 → 0.37 | **0.90 → 0.86** |
| WED SX 1-min | 0.55 → 0.52 | **1.28 → 1.02** |
| FRI SX 1-min | 0.71 → 0.68 | **1.79 → 1.79** |
| MON NF 5-min | 0.43 → 0.40 | **1.08 → 1.07** |
| WED NF 5-min | 0.53 → 0.50 | **1.42 → 1.42** |
| FRI NF 5-min | 0.69 → 0.66 | **5.21 → 5.21** |

**Skipping one day in five shaves 3–12 % off the p90 and leaves the maximum untouched in four
of six cases.** The bad day is not the hot-open day. It is not predictable from anything here.

**`pdr_rel`** is the second survivor (exc_norm ρ +0.11 … +0.24, and it *strengthens* under
normalisation because it is already vol-normalised) — but it fails the random-skip null on 4 of
6 combinations (rand-percentile 7 – 78).

### Route A conclusions on the specific things Arun named

- **Daily CPR width — research/67's sign confirmed, but it is a volatility proxy.** `cpr_today`
  correlates positively with the raw excursion on all six combinations (+0.12 … +0.31), so
  narrow → calm holds. But the curve is **flat across Q1–Q4 and jumps only in Q5**
  (Mon NF: 24.9 / 23.3 / 24.9 / 25.2 / **38.8** bp) — a top-quintile effect, not a dose response —
  and normalised it is +0.03 … +0.24 with Q5/Q1 of 1.06–1.26.
- **Weekly CPR — the sign flip does NOT reproduce here, exactly as the brief warned.**
  research/67 predicts *wide weekly → contained*. Normalised Q5/Q1 is **0.80 on SENSEX
  Wednesday (r/67's sign) but 1.12 on NIFTY Monday and Friday (the opposite)**. A rule built on
  it would be wrong on two cells out of three. Do not use weekly CPR to gate these windows.
- **Gap.** |gap| is the weakest of the vol proxies (+0.09 … +0.23 raw, ≈0 normalised). The
  *signed* gap is the more interesting one — down-gaps precede larger premium-relative moves
  (−0.07 … −0.27) — but that is a directional read, not a risk filter, and it is a single-cell
  effect (SENSEX Monday −0.27).
- **Previous-day range.** Real on raw (+0.28 … +0.43), mostly priced away on normalised.
- **India VIX level.** Strong raw predictor (+0.29 … +0.50), **and skipping high-VIX days makes
  premium-relative risk WORSE than random** (rand-percentile 67 – 97 across the six
  combinations). This is research/115's `atm_iv` trap and research/120's calm-window trap in one:
  a volatility proxy regressed on a volatility outcome, where the payoff scales with the proxy.
- **India VIX shock (rebuilt from 5-min).** There *is* a genuine tail effect: on NIFTY Monday
  the top 5 % of overnight VIX jumps precede a raw excursion of **40.3 bp vs 27.3 bp** — monotone
  in both the percent and the absolute-point normalisation (27.3 → 29.5 → 31.2 → 35.9 → 40.3 bp
  across p50/p70/p80/p90/p95 of `vix_gap_pts`). Normalised, the same ladder is
  **0.263 → 0.268 → 0.275 → 0.286 → 0.322** — a ~22 % effect on 5 % of days. As a skip rule it is
  inert: over 36 shock rules tested, **the maximum is never removed** (Mon 1.081 → 1.081 on all
  nine; Fri 5.213 → 5.213 on eight of nine) and on Wednesday it is worse than random.
- **Combinations (6 pre-registered).** The best, *hot pre-open range OR big gap*, skips ~33 % of
  all trading days and moves the p90 premium-relative excursion by **−8.5 % (Mon), −3.3 % (Wed),
  −2.0 % (Fri)**, leaving the maximum at 1.081 → 1.068, 1.277 → 1.017, **5.213 → 5.213**. On the
  raw excursion the combinations look impressive (rand-percentile 0.0–1.3); on the
  premium-relative outcome they are mostly indistinguishable from random and on Friday four of
  six are *worse* than random.

---

## 5. Route A — the rupee confirmation, and the placebo that ends it

Same rules, applied to real option P&L on the 14–17 recorded days per cell, scored against an
**exact** random-skip null (every C(n, n−k) subset enumerated where feasible, else 20,000 draws).

**Cost of skipping — the number that kills it before any statistics.** Total net P&L retained
after skipping the top 4 of 14–17 days by each signal:

| cell | `cpr_today` | `pdr_rel` | `vix_open` | `gap_abs` | `pre_range_bp` |
|---|---|---|---|---|---|
| FRI_NIFTY_DTE2 | **53 %** | 59 % | 59 % | 78 % | 87 % |
| MON_NIFTY_DTE1 | 103 % | 103 % | 65 % | **39 %** | 45 % |
| WED_SENSEX_DTE1 | 47 % | 75 % | 55 % | 121 % | **25 %** |

Every signal's sign is different on different cells, and the good cases are as large as the bad
ones — the definition of noise. Skipping the hottest 4 Mondays by `gap_abs` retains **7 %** of
the book's P&L at 6 skips and **0.3 %** at 8.

**The placebo exhibit.** 540 skip rules (30 signals × 2 sides × 3 skip-fractions × 3 cells):

| | |
|---|---|
| Rules beating the 95th percentile of random **and** retaining ≥100 % of P&L | **10** |
| Expected under the pure null | **~27** |
| Of those 10, rules built on a signal with zero information | **1 — `placebo_noise`** |

`placebo_noise` is `numpy.random.normal`. On SENSEX Wednesday, skipping its bottom 6 days
"retained 136.8 % of P&L" and sat at the **96.9th percentile** of random skips. It is
indistinguishable from the study's "best" real filters, and it beat most of them. The correct
reading is that **there are fewer apparent winners than chance would produce**, and the ones
there are cannot be told apart from a random number generator.

**Information coefficients, pooled across the three cells (n = 48 day-cells).** The signals do
predict the *drawdown path* — `pdr_pct` +0.37 (p 0.010), `pdr_rel` +0.34 (p 0.020), `cpr_today`
+0.33 (p 0.024), `d_oi_atm_pct` +0.42 (p 0.004) against `exc_over_credit` — while being
**flat against the P&L that was actually booked**: the same four score −0.02, −0.09, +0.03,
+0.14 (all p > 0.29). With a stop that never fires, the outcome is decay, not the path. And
`d_oi_atm_pct`, the strongest IC in the study, is the **worst** rule on Wednesday in the skip
test (−27.8 % of P&L retained). It is not a signal.

**On PCR / OI specifically** (the scope extension). Coverage is complete (100 % non-null on
every book feature, 98 % on the overnight deltas), so this is a fair test of a genuinely
different question from research/115's — day-level pre-entry conditioning rather than intraday
adjustment timing. It fails for a different reason than research/115's: not contamination, but
**absence of a long sample**. The chain recorder starts 2026-04-20, so no PCR/OI signal can ever
be fitted on more than ~16 days per cell — precisely the regime where `placebo_noise` scores at
the 97th percentile. Nothing here can be validated with the data that exists. (Research/115's
established findings were respected and not re-litigated: the price-anchor contamination, the
strict 3-minute SENSEX OI cadence, and the premium-level trap on `atm_iv`, whose exact analogue
— `vix_open` — is the strongest false positive in §4.)

---

## 6. Route B — the stop ladder, and why 1:2.5 is arithmetic

Real 1-minute chain, live weekday, net of cost, ₹ at 10 lots.

| Cell | arm | total | median | worst | fire % (observed) | Δ P&L vs SL20 |
|---|---|---|---|---|---|---|
| **FRI_NIFTY_DTE2** (n=14) | NOSTOP … SL8 | 55,920 | 3,575 | −3,440 | 0 % | 0 % |
| | **SL20 (live)** | **55,920** | 3,575 | −3,440 | 0 % | — |
| | SL6 / RC1200 | 44,150 | 3,575 | **−15,210** | 7 % | **−21.0 %** |
| **MON_NIFTY_DTE1** (n=17) | every arm to SL6 | 16,980 | 1,240 | −4,840 | 0 % | 0 % |
| **WED_SENSEX_DTE1** (n=17) | NOSTOP … SL10 | 32,690 | 3,830 | −11,440 | 0 % | 0 % |
| | SL8 | 24,060 | 3,830 | **−15,620** | 12 % | **−26.4 %** |
| | SL6 | 19,110 | 3,830 | −12,630 | 24 % | **−41.5 %** |

Note the sign of the *worst* column: **tightening the stop made the worst day worse**, on both
cells where it engaged. A stop can only fire after a retrace, so it converts a day that would
have decayed back into a booked loss — the same mechanism research/116 found for ratchets
(median give-back rising monotonically as the rule tightens) and research/114 found on SENSEX
Thursday. Third independent reproduction.

### The arithmetic

The loss cap is `credit × stop% × lot + cost`. Against each cell's observed median profit:

| Cell | median profit | SL20 cap | R:R at SL20 | stop needed for 1:2.5 | that stop fires on … (long sample) |
|---|---|---|---|---|---|
| MON_NIFTY_DTE1 | ₹1,240 | ₹25,987 | **1 : 21.0** | below 2 % → **unreachable** (R:R at a 2 % stop is still 3.91) | 84 % of days |
| WED_SENSEX_DTE1 | ₹3,830 | ₹26,284 | **1 : 6.9** | ~6 % | **17.7 %** of days |
| FRI_NIFTY_DTE2 | ₹3,575 | ₹36,930 | **1 : 10.3** | ~3.5 % | **~45 %** of days |

Monday is unreachable because of the **cost floor**: at a 2 % stop the cap is ₹2,349 + ₹250
per lot, and the round trip alone is 20 % of the median profit. You cannot build a 1:2.5 trade
whose reward is barely four round trips.

Stop level → triggering underlying move → true firing frequency (the map is fitted on the real
chain and is **conservative**: on the recorded sample the actual premium path fired SL8 on 12 %
of SENSEX Wednesdays where the map predicts 0 %, because IV pops and spreads widen and the map
only knows the underlying — so every "long" column below is a lower bound):

| stop % of credit | move that triggers it | fires (long sample) MON / WED / FRI |
|---|---|---|
| 20 (live) | 133 / 132 / 224 bp | **0.2 % / 0.4 % / 0.6 %** |
| 10 | 66 / 70 / 110 bp | 4.6 % / 2.9 % / 2.6 % |
| 8 | 53 / 58 / 87 bp | 7.7 % / 6.5 % / 6.6 % |
| 6 | 39 / 46 / 65 bp | 17.7 % / 17.7 % / 13.6 % |
| 4 | 26 / 33 / 42 bp | 40.1 % / 32.9 % / 34.1 % |
| 3 | 19 / 27 / 30 bp | 59.7 % / 48.4 % / 57.5 % |

**The deployed SL20 is decorative** — it fires on between one day in 500 and one in 170. That is why the observed R:R
looks like 1:4–1:6 while the structural R:R is 1:7–1:21.

---

## 7. The real tail — why neither sample's "worst day" should be used for sizing

The 14–17 recorded days for each cell **never saw the tail**. Maximum underlying excursion
inside the live window:

| | recorded sample | long sample p90 | p99 | **max** |
|---|---|---|---|---|
| MON 13:00–14:00 | 39.4 bp (n=17) | 43–49 | 85–98 | **97 – 141 bp** |
| WED 10:30–12:00 | 51.6 bp (n=17) | 52–55 | 92–134 | **215 – 316 bp** |
| FRI 10:00–12:00 | 50.4 bp (n=14) | 68–72 | 125–159 | **160 – 1,304 bp** |

The recorded maximum barely reaches the long-run **p90**. Modelling each long-sample day's loss
(credit predicted from VIX at r 0.65–0.91, premium rise from the Theil-Sen map on the real
chain, capped at SL20 — **modelled, research/103 precedent, directional not decision-grade**):

| Cell | median profit | modelled loss p90 | p99 | max | observed worst | **true R:R (p99)** |
|---|---|---|---|---|---|---|
| MON_NIFTY_DTE1 | ₹1,240 | ₹10,191–10,245 | ₹15,881–18,001 | ₹18,691–25,006 | −₹4,840 | **1 : 12.8 – 14.5** |
| WED_SENSEX_DTE1 | ₹3,830 | ₹10,209–11,312 | ₹17,701–22,231 | ₹42,105–56,741 | −₹11,440 | **1 : 4.6 – 5.8** |
| FRI_NIFTY_DTE2 | ₹3,575 | ₹12,720–13,676 | ₹21,709–25,936 | ₹26,907–57,593 | −₹3,440 | **1 : 6.1 – 7.3** |

So the brief's headline R:R of 1:6.1 / 1:4.3 / 1:6.2 is **roughly right for Wednesday and Friday
and far too kind for Monday**, even though its individual rupee figures could not be reproduced.
The maxima exceed the SL20 cap on the mean credit because a 20 % stop on a high-VIX day is a
bigger rupee number — the cap scales with the premium, which is another way of saying the
"protection" grows in exactly the regime you wanted it to shrink.

---

## 8. What the evidence supports

**1 — Change no rule. Do not add a filter and do not tighten the stop.**
Route A produces fewer winners than chance and cannot beat a random number generator. Route B
either cannot reach the target (Monday) or reaches it by firing on a fifth to a half of all days
and costing 21–42 % of the book's P&L, while *worsening* the worst day. This is the third
independent reproduction of the research/114 / research/116 result: **on these books, every stop
tightening tested loses to holding.**

**2 — Cut Monday NIFTY DTE1. It is the specific, defensible action.**

| | MON_NIFTY_DTE1 | WED_SENSEX_DTE1 | FRI_NIFTY_DTE2 |
|---|---|---|---|
| median as % of credit | **0.83 %** | 3.33 % | 2.56 % |
| mean ₹/day @ 10 lots | **+998** | +1,922 | +3,994 |
| true R:R (p99 modelled) | **1 : 12.8 – 14.5** | 1 : 4.6 – 5.8 | 1 : 6.1 – 7.3 |
| round trip as % of median profit | **20 %** | 5 % | 7 % |
| 1:2.5 reachable at any stop? | **No, arithmetically** | at 6 %, firing 18 % of days | at 3.5 %, firing ~45 % |

Monday is the worst cell on every dimension, and it is the *only* one where the target is
impossible rather than merely expensive. Moving it from 8 lots to 3 cuts its rupee tail by 63 %
and costs ₹600/Monday of median profit. Dropping it entirely costs ~₹800/week at current size.

**3 — R:R is the wrong success metric for a short-premium window, and that is worth saying out
loud.** These cells win **71 %, 71 % and 93 %** of the time with positive expectancy. Demanding
1:2.5 from a 60–90-minute short straddle is demanding that an insurance underwriter stop
collecting small premiums against rare large claims. The thing Arun is actually uncomfortable
with is **how much a bad day costs in rupees**, and the lever for that is size:

| lots | MON cap | WED cap | FRI cap | R:R |
|---|---|---|---|---|
| 10 | ₹25,987 | ₹26,284 | ₹36,930 | 21.0 / 6.9 / 10.3 |
| 8 | ₹20,789 | ₹21,027 | ₹29,544 | **identical** |
| 5 | ₹12,993 | ₹13,142 | ₹18,465 | **identical** |
| 3 | ₹7,796 | ₹7,885 | ₹11,079 | **identical** |

R:R is invariant to size by construction. Size is the only dial that changes how much the bad
day hurts without changing expectancy per rupee of margin.

**4 — If Arun still wants one conditioning rule, it is `pre_range_bp`, and it is not worth it.**
Stand aside when the session's own 09:15→entry range is in its top quintile. It is the only
signal that survived every control — monotone, sign-consistent across both venues and all three
windows, and in the bottom 4 % of random skips on all six combinations. It buys a **3–12 %
reduction in the p90**, **no reduction in the maximum**, and costs a fifth of the trading days
(retaining 87 % / 45 % / 25 % of P&L on Fri / Mon / Wed respectively). Recommended only if the
goal is a calmer mark-to-market, not a better book.

---

## 9. Multiple-testing accounting

Everything tried is reported; nothing was selected after the fact.

| Stage | Cells evaluated |
|---|---|
| Long-sample Spearman (16 signals × 3 windows × 3 series × 2 weekday-filters × 2 outcomes) | **576** |
| Long-sample skip rules (2 sides × 7 cuts, each vs a 2,000-draw null) | **7,308** |
| Long-sample quintile response curves | 2,656 |
| VIX family, rebuilt (7 signals × 3 cells × 2 outcomes + 72 swept thresholds + 36 skip rules) | **150** |
| Options-sample ICs (30 signals × 2 outcomes × 4 scopes) | **240** |
| Options-sample skip rules vs an exact random-skip null | **540** |
| Pre-registered combinations (6 × 3 cells × 2 outcomes) | **36** |
| Stop / rupee-cap ladder (15 arms × 3 cells) | **45** |
| **Total** | **≈ 11,550** |

At 11,550 evaluations a 5 % threshold yields ~575 false positives by construction. The
acceptance bar was held at *monotone response curve* **and** *beats a random-skip null of equal
frequency* **and** *improves total P&L, not just the tail*. **Nothing cleared all three.** One
signal (`pre_range_bp`) cleared the first two and fails the third. The confirmation stage
produced 10 apparent winners against ~27 expected by chance, one of which is pure noise.

---

## 10. Sins accounting

| Sin | How it was controlled |
|---|---|
| **Look-ahead** | Every feature computable before the window opens: CPR from prior-period OHLC, gap and VIX open at 09:15, `pre_range` from bars strictly before the entry minute, trailing 20-day range average built from days strictly earlier. Strike chosen from the spot at the window's own start minute. Stop evaluated minute-forward only. Expiry derived per day from the chain, never assumed. |
| **Survivorship / sample selection** | Every recorded day used; the three exchange holidays removed by a **data rule** (frozen chain, < 50 distinct spot prints) not by looking at their P&L; today's partial session excluded. Long sample is the full index history, no selection. |
| **Overfitting / multiple testing** | ~11,550 cells, all reported (§9); full response curves rather than best cuts; a **placebo signal with zero information** included in the same family and shown to beat 97 % of random skips; winner count *below* the null expectation. |
| **Cost neglect** | Net everywhere, ₹250/lot NIFTY and ₹200/lot SENSEX charged on every trade including every skipped-day counterfactual. Cost is a per-trade constant so 2× sensitivity is exact. The cost floor is the *reason* Monday cannot reach 1:2.5 and is stated as such. |
| **Regime dependence** | The rupee sample is one benign quarter (May–Aug 2026) and is treated as confirmation only; every risk claim is read off 274–557 days spanning 2015–2026 (NIFTY) and 2021–2026 (SENSEX), which include 2020-style tails on the NIFTY clock (max 1,304 bp). |
| **Correlation / single factor** | The three cells are the same short-gamma bet on two 0.98-correlated indices; the study never sums their P&L or claims diversification. Each cell is judged standalone. |
| **Capacity / liquidity** | Not re-examined — unchanged from research/120's margin work; no sizing increase is proposed, only a decrease. |
| **Placebo / control** (research/115) | Two placebos carried through the whole confirmation: `placebo_noise` (Gaussian) and `placebo_prepath` (signed pre-window index move — zero option information but mechanically embedded in any intraday PCR). Every skip rule scored against an **exact** random-skip null of identical size. |
| **Volatility-proxy-on-volatility-outcome** (research/115's `atm_iv` trap) | The decisive control of this study: every outcome reported both raw and **divided by the day's VIX-implied sigma**. It is what turns `vix_open` from the second-best signal into nothing. |
| **Data provenance** | Two traps found and reported: the INDIAVIX daily `open` degeneracy (82.5 % of bars), and the resolution-equivalence proof that licenses the NIFTY 5-minute long sample for this statistic. |

---

## 11. Honest caveats

1. **The rupee sample is 14–17 days per cell, in one benign quarter.** Its win rates (71/71/93 %)
   and its worst days (−₹4,840 / −₹11,440 / −₹3,440) are not estimates of anything. research/118
   made exactly this error reading "worst −127" off 12 Thursdays when the true DTE0 worst was
   −21,500/lot. **Size for §7's numbers, not for §3's.**
2. **The tail table in §7 is MODELLED.** Credit is predicted from VIX by a 14–17-point regression;
   the premium rise is a Theil-Sen line through the real chain that explains r ≈ 0.82–0.88 on
   SENSEX but only r ≈ 0.13–0.33 per-day on NIFTY. Treat the loss percentiles as directional
   magnitudes, not decision-grade rupees. They are, however, **conservative** — the map fires
   stops less often than the real premium path did.
3. **The brief's problem table could not be reproduced** and this study substitutes its own
   baseline. If the brief's numbers came from a source not visible to me (live-book journal,
   a different pooling), the §3 correction should be checked before it is relied on.
4. **NIFTY has no 1-minute index series.** The equivalence proof in §2 removes this as a concern
   *for the maximum-excursion statistic only*. Any future question about the NIFTY intraday
   **path** still needs a 1-minute series we do not have.
5. **PCR / OI conclusions are weaker than the rest.** With no long sample they cannot be fitted
   honestly at all; the finding is "untestable with existing data", not "proven absent".
6. **Single-entry model.** One straddle sold at the start minute, covered at the end or on a
   combined-% stop, on 1-minute LTP with fixed slippage. It does not model the live books'
   5-second polling, two-poll dwell, or 50 % disaster backstop. Fills are optimistic on every
   stop arm — which biases §6 *in the stops' favour*, and they lose anyway.

---

## 12. Next levers

- **The one open question this study raises but cannot answer:** the `pre_range_bp` result is
  real (monotone, both venues, all three windows, beats the null on 6/6) and merely too small to
  matter as a *skip* rule. It might matter as a **sizing** rule — half-size when the morning is
  hot — which changes P&L by the same 3–12 % without forfeiting a day's premium. That is a
  different (and cheaper) test than the one asked for here.
- **Re-run §3 and §5 after 2026-11** with ~28 days per cell. The scripts are resumable and the
  holiday guard now catches 2026-05-28 as well.
- **A dedicated review of MON_NIFTY_DTE1's existence**, not just its size: 17 Mondays at
  ₹998/day at 10 lots against ₹1.65 L/lot of margin is a poor use of the margin line even before
  the tail is considered.
- **The `CSL_TIMEB2_NIFTY` review scheduled for 2026-09-05** should carry this study's answer for
  its DTE0 13:00–14:00 cell, which is the same shape as MON_NIFTY_DTE1.

---

**Reproducibility stamp.** Data snapshot 2026-08-21 (`market_data.db` 30.3 GB,
`options_data.db` 12.1 GB, both opened read-only on the VPS). Scripts, in order:
`s1_daily_features.py` (day-level regime features) · `s2_window_outcomes.py` (long-sample window
excursions) · `s3_longfit.py` (Spearman, quintiles, skip rules vs a 2,000-draw null) ·
`s4_options_sample.py` (real 1-min chain replay, stop ladder, PCR/OI features) ·
`s5_stops_and_tail.py` (stop ladder, move→premium map, R:R arithmetic) ·
`s6_filters_confirm.py` (rupee confirmation, exact random-skip null, placebos) ·
`s7_robustness.py` (resolution-equivalence proof, monotonicity, combinations, size arithmetic) ·
`s8_true_tail.py` (modelled long-sample loss distribution) · `s9_vix_shock.py` (INDIAVIX trap +
rebuilt VIX family). Costs NIFTY ₹250/lot, SENSEX ₹200/lot round trip. Random seed 20260821.
Live config read from `backtest_data/csl_paper_config.json` (refrozen 2026-08-20). **No live
config, service, engine or order path was touched.**
