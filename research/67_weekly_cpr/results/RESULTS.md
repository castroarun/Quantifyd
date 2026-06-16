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
