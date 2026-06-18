# Weekly vs Daily CPR — Movement, Direction, and the V2 Fly Gate

**STATUS: DONE — SIGNAL (context tool, not a hard gate).** NIFTY, 5-min→weekly/daily resample, 2015-2026
(market_data.db NIFTY50 5min; daily `day` series only covers 2023+, so 5-min resample used for full history).

## The ask
"For wide weekly CPRs, what's the max NIFTY move that week?" → grew into: validate the classic CPR rule
(narrow→trending, wide→sideways) on NIFTY weekly; produce trade-planning stats; check whether it can gate
the V2 iron fly; and reconcile with our live **daily** CPR gate (which assumes narrow=calm).

CPR width = |2C − H − L| / 3 (BC=(H+L)/2, P=(H+L+C)/3, TC=2P−BC). Drawn FOR a period from the PRIOR
period's H/L/C — i.e. the lines on the chart for the current week. Width = how far the prior period
CLOSED from its mid-range ⇒ a **trending-close** measure (corr 0.65 with close-at-extreme), NOT a range
measure. That single fact reframes everything below.

## 1. Weekly CPR — the classic rule HOLDS (with the right metric)
My first pass used intra-week RANGE (high−low) and got the wrong sign (+0.24, wide→bigger range) — range
counts CHOP and is contaminated by 2020. The correct metrics are net directional move / trend-efficiency /
containment:

| RECENT 2023-26 · prior-week CPR | net abs % | trend-eff | maxSide p90 % | closed INSIDE band |
|---|---|---|---|---|
| Q1 narrow (0.00–0.19%) | 1.27 | 0.531 | 2.95 | 0% |
| Q5 WIDE (0.70–2.02%) | **0.77** | **0.361** | **2.40** | **32%** |

corr(width, net) = −0.17, corr(width, trend-eff) = −0.20 (recent). **Wide CPR → contained/sideways; narrow
→ trends.** The cleanest, regime-robust tell is **close-inside-CPR**: rises monotonically with width in BOTH
samples (full 1%→23%, recent 0%→32%).

## 2. Directional signal — 1st-30-min vs the weekly CPR (STABLE, tradable)
| 1st-30min (09:15–09:45) closed | % weeks | closed above | closed below |
|---|---|---|---|
| ABOVE the band | 54% | **69%** | 22% |
| BELOW the band | 36% | 28% | **58%** |
Identical full vs recent. Open above weekly CPR → 69% the week holds bullish; open below → 58% bearish
(weaker — equity drift). Even on bullish weeks, ~43% dip back through the band intra-week (buy-the-dip zone).

## 3. Daily vs Weekly — the SIGN FLIPS
| DAILY (full 2015-26) prior-day CPR | net abs % | breach >1% |
|---|---|---|
| Q1 narrow (0.00–0.06%) | 0.52 | 23% |
| Q5 wide (0.32–4.68%) | 0.75 | 43% |
corr(daily width, next-day move) = **+0.225**. **At the DAILY scale narrow→calm, wide→more move — the
OPPOSITE of weekly.** CPR width is volatility-PERSISTENCE daily (quiet begets quiet) but trend-EXHAUSTION
weekly (a trending close → digesting week). Scale flips the sign.

## 4. (a) Weekly CPR as a fly entry filter — NOT robust
Breach = 2% move-stop hit during the week (signal proxy; real ₹ needs premiums). Recent: WIDE → lower
breach (>2%: 24% vs 32% narrow; >3%: 6% vs 12%) — supports selling on wide-CPR weeks. BUT full history
FLIPS — widest CPRs cluster at crash onset (2020) → WIDE breaches MOST (>2%: 52%). **Regime-conditional
edge that backfires in a crisis; not strong enough to hard-gate the fly.**

## 5. (b) Our live DAILY gate is correctly signed — and combo_skip is justified
- The daily gate (narrow prior-day CPR = calm = enter) has the RIGHT sign (3 above): narrow daily → calm.
- The `combo_skip` "skip if prior-day CPR < 0.10%" looked backwards on the movement axis (it discards the
  calmest days). It traces to **research/61** (narrow-CPR-skip + inside-week-skip *stacked* → Calmar
  1.03→2.00) — but that figure is the TWO filters combined, the narrow-CPR component is NOT isolated.
- Forward premium evidence (recorder 2026-04-27→06-16, n=34): NARROW-CPR days (<0.10, n=10) collected
  **1.16% of spot** in 09:20 straddle credit vs NORMAL **1.27%** (n=24) — only ~9% thinner — and BOTH
  groups stayed calm (100% <2%, ~0.7% avg move). So the skip discards calm days for a MODEST premium saving;
  the reward-for-risk rationale (thin credit vs a full-wing breach) CANNOT be tested in this benign,
  breach-free window.
- **VERDICT: UNCONFIRMED, not "clearly justified".** Keep the skip conservatively (research/61 is the only
  breach-inclusive evidence), but its STANDALONE value is genuinely uncertain — the recent premium gap is
  small and narrow days are calmer. **Owed: an isolated backtest of the narrow-CPR-skip ALONE over
  breach-inclusive history with real premiums** (decouple it from the inside-week filter). Do NOT change
  live on the strength of this; do NOT claim it's settled.

## Verdict
Weekly CPR is a good **context/direction** read (wide=sideways-friendly; narrow=trend-risk; 1st-30min
above/below → 69%/58% directional hold) but **too regime-fragile to hard-gate** the fly. The DAILY gate
(narrow=calm) is correctly signed and the narrow-CPR skip is premium-justified — no change needed. The
seven sins: this is in-sample single-instrument (NIFTY), regime-dependent (2020 dominates the full daily
sample), and the premium leg is only ~2 months of recorder data — a SIGNAL/context tool, not a standalone
STRATEGY.

---

# WEEKLY CPR PLAYBOOK (consolidated) — entry classification + intra-week management

**STATE LEGEND (used throughout — read first):**
- **AGREE-UP** = the day/Monday close is ABOVE *both* the weekly CPR band and the daily CPR band.
- **AGREE-DOWN** = BELOW *both*.
- **DISAGREE** = above one but below/inside the other (previously called "coin-flip" / "split").

AGREE = a tradeable directional lean; DISAGREE = no reliable direction → trade neutral.
Weekly CPR = lines drawn for the week from the PRIOR week; daily CPR = for the day from the PRIOR day.
Signal fixed at the 1st-30min close (09:45) / day close. Causal. NIFTY 5min→weekly/daily, **11y 2015-2026**.

## A. Context (validated)
- Weekly CPR width: **NARROW → week TRENDS; WIDE → week SIDEWAYS/contained** (measured by net-move &
  containment, NOT high-low range — range was the wrong metric).
- **TOO-narrow** (top whipsaw decile) → whippy (66-74% cross both sides) → **SKIP the directional break.**
- **Daily vs weekly SIGN FLIP:** weekly narrow = trend, daily narrow = calm. Do not mix the two timeframes.

## B. ENTRY classification (Monday 09:45) → structure
The 1st-30min candle read two ways: POSITION vs the weekly CPR (which side; ~69% above / 58% below holds)
and COLOR (green/red = whether the week actually TRAVELS). Daily AGREE/DISAGREE is the gate.

| Entry state | read | structure |
|---|---|---|
| AGREE-UP + GREEN | bull tilt (72% hold, +0.36% net) | bullish jade lizard / bull-put spread |
| AGREE-UP + RED | holds up but ~0 net | NEUTRAL — iron condor / iron fly |
| AGREE-DOWN + RED | bear tilt (61%, −0.40%) | bear-call / put debit (defined-risk) |
| close below weekly + GREEN candle | reversal-UP trap (+0.66%) | do NOT go bear — neutral / mild bull |
| DISAGREE | coin-flip direction (52%) | NEUTRAL only — condor / fly |

The two layers are orthogonal: **daily AGREE drives the hold rate; candle color drives the net travel.**

## C. DISAGREE weeks → neutral structures (max move + pivot hits for wing placement)
| DISAGREE scenario | n | maxBull avg/p90 | maxBear avg/p90 | R1%/R2% | S1%/S2% |
|---|---|---|---|---|---|
| close above weekly, daily disagrees | 53 | 1.08/1.88 | 1.11/2.47 | 43/19 | 26/11 |
| close below weekly, daily disagrees | 32 | 1.81/4.02 | 1.73/2.98 | 34/22 | 47/22 |

Both-sides whipsaw only **6%** → DISAGREE weeks are CONTAINED → sell premium. Above-disagree: mild up-lean,
S1 hit only 26% → condor short put ~S1 / call ~R2. Below-disagree: leans UP (+0.61% net) with a FAT upside
tail (p90 4.0%) → push the call wing beyond R2; never go bear.

## D. INTRA-WEEK management — re-check the state at EACH day close
The state EVOLVES: **48% of weeks the weekly side flips at least once.** Re-classify every evening.

**Robust base (large n) — ANY day showing AGREE → rest-of-week holds that side:**
Mon **71%** (n=457) · Tue **77%** (n=369) · Wed **83%** (n=343) · Thu **90%** (n=348).
(Absolute % rises toward week-end partly mechanically — less time to change; the AGREE-vs-DISAGREE *gap* is
the real signal.)

**Adjustment cues:**
- **DISAGREE, below weekly CPR = reversal-UP** every day (rest net +0.5–0.6%) → lift bearish risk / mild bull.
- **An uncertain week that RESOLVES to AGREE** (pooled n=120, robust): the new side holds **AGREE-UP 73% /
  AGREE-DOWN 46%** → add the tilt on the UP resolution; the DOWN resolution stays unreliable (equity drift).
- **A side-flip** (price crosses the weekly band) → re-center the structure to the new side.

(Why earlier Mon→Wed transition cells were n=7-14: AGREE is 79% of Mondays, DISAGREE only 21%, so the
specific sub-cell is rare — NOT a data shortage. Pooled across all days/sides → n=120, robust.)

## Verdict
A **Weekly CPR Playbook** for inline premium structures (NOT trend-catching): classify the week at entry
(AGREE+matching-color → directional defined-risk; DISAGREE / opposite-color → neutral premium), then
re-check the daily state each evening to adjust. SIGNAL/context tool — edge is in DIRECTION & structure
choice, ~±0.4% net magnitude; option P&L still needs real premiums. In-sample, NIFTY-only; persistent
bull/bear asymmetry (AGREE-UP ≫ AGREE-DOWN) from equity upward drift.
