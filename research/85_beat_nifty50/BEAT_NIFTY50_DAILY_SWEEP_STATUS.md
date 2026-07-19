# research/85 — Beat Nifty by ≥5pp Within Nifty 50 (three evidence-backed families)

STATUS: DONE (see Verdict)

User mandate 2026-07-17: strategies on Nifty-50 stocks that beat index returns
by ≥5pp CAGR; filters/technicals allowed. Fundamentals excluded (no data; MQ
prior: quality weights inert).

## Locked grid (ledger +10)

- **R1 momentum rotation (8 cells):** rank today's Nifty-50 constituents by
  trailing return {6m (126d), 12m-minus-1m (252-21d)}; hold top {5, 10}
  equal-weight; month-end rebalance (signals at month-end close, fills next
  session close); gate {none, NIFTYBEES>200DMA else 100% cash}.
- **R2 turtle-N50 (1):** r/83 Turtle-EQ book (S1+S2, 2N, equal-notional,
  cap 8, gated) restricted to the Nifty-50 subset. Rules locked from r/83.
- **R3 KC6-N50 (1):** live KC6 rules on Nifty 50 — entry close < KC(6,1.3)
  lower AND close > SMA200; exit at KC mid (standing target), SL 5%, TP 15%,
  max-hold 15 sessions; 10 slots equal-notional. Rules locked from the live
  system (no tuning).
- Costs: CASH_DELIVERY (rotation/KC6) / futures-proxy (turtle, as r/83).
- Splits: IS 2005-2018, Val 2019-2023 (reported), OOS 2024+ untouched.
- Benchmark: NIFTYBEES B&H. **Gate: net CAGR ≥ benchmark+5pp on IS AND Val,
  MaxDD ≤ benchmark's, per-year excess positive ≥60% of years.**

## Honest caveats (stated up front)

Today's Nifty-50 membership = survivorship bias (inflates momentum backtests;
r/75's survivorship-FREE +18pp on Nifty-250 is the reason to believe the
family anyway). 50 names = thin cross-section; concentration risk is the
price of excess. Results will carry a survivorship-haircut note.

## Status

| Date/time | Event |
|---|---|
| 2026-07-17 ~16:50 IST | Pre-registered; runner `scripts/run_beat_n50.py` |

## Findings

(after run)

## VERDICT (2026-07-17 ~17:15 IST)

**WINNER (passes ALL pre-registered gates): 12-1 MOMENTUM ROTATION, TOP-10,
monthly, ungated — FULL 25.9% CAGR vs 13.5% bench = +13.8pp excess (IS +14.0,
Val years +4.9/0/+37.5/+6.8/+22.8), MaxDD -55% = bench, beat 14/18 years.**
Top-5 variant: +16.8pp but DD -66% > bench (fails DD gate). Gated variants
HALVE DD (-34%) but fail the Val excess gate (gate whipsaw 2020/2022 —
consistent with prior gate findings). Formation 12-1 >> 6m throughout.
R2 turtle-N50: FAILS (-2.4pp; mega-caps trend too weakly — turtle needed
breadth). R3 KC6-N50: low-exposure overlay, not a mandate candidate (-11.7pp
vs full-invested bench at slot sizing; per-signal fine, win 61%).
NOTE: R1 Sharpe figures inflated (monthly-NAV annualization) — CAGR/DD/yearly
are the valid columns. Survivorship haircut applies (today's 50) — r/75's
survivorship-FREE +18pp on Nifty-250 momentum is the reason to believe the
family net of bias; even a 50% haircut clears +5pp. OOS 2024+ UNTOUCHED —
available for a one-time confirmation before deployment.
STATUS: DONE — STRATEGY-CANDIDATE (momentum top-10 12-1)

## OOS TOUCH (2026-07-17, user-authorized, CONSUMED for the momentum-N50 family)

Window 2024-01..2026-07 (~31 monthly decisions — short for a monthly system):
- top10_nogate (LOCKED): excess +0.4pp (2024 +10.0, 2025 -7.1, 2026 -1.0)
- top5_nogate: +4.2pp (2024 +31.0, then -9.3/-5.5) — front-loaded
- top10_gate200: +1.1pp at MaxDD -14%

**The +5pp mandate is NOT confirmed out-of-sample in the recent regime.**
Same shape as every study: strong 2024, fading 2025-26. HOWEVER: 2.5yrs is a
short OOS for monthly rotation; the IS record contains comparable droughts
(2018-19: -8/-7pp) followed by big recoveries (+37.5pp 2021, +22.8 2023) —
drought vs decay is indistinguishable at n~31 months. VERDICT downgraded:
STRATEGY-CANDIDATE -> **SIGNAL/WATCH** for the concentrated N50 version.
Recommendation: the family stays best expressed via the BROADER live
momentum-paper book (Nifty 200 — breadth is momentum's fuel); a N50 top-10
paper sleeve is reasonable for a patient investor who accepts multi-year
tracking droughts; no real-money deploy on this evidence.
No further OOS looks for momentum-N50.

## ADDENDUM (2026-07-17 ~18:15 IST): EXP-ST1 + EXP-ZOO (per-stock timing, +22 cells)

User ideas: (1) per-stock ST(7,3)+liquid on all-N50 and top-10-mcap;
(2) timing zoo (ST params, EMA crosses, P>EMA, Donchian, RSI, stoch, MACD).

**VERDICT: NO timing signal adds return over holding the same basket — all 20
zoo cells NEGATIVE vs same-basket B&H (-6.2 to -19.6pp/yr).** The seeming
+3.6pp of ST-case-1 vs NIFTYBEES is pure survivorship: today's N50 members
equal-weight B&H did 24.8% CAGR vs NIFTYBEES 13.5% (+11pp survivorship gift);
ST timing DESTROYED 7.7pp of that. Top-10-mcap case even worse (-8.2pp vs its
basket, beat 4/19 yrs). What timing DOES buy: drawdown — DONCH20/10 cuts DD
-57%->-16% (Calmar 0.89 vs 0.43) at -10.6pp CAGR; EMAX9/21 the best
CAGR-preserving (-6.2pp for DD -23%). Oscillators (RSI>50, stoch, MACD) are
pure churn destruction (18-57 flips/yr). Taxes NOT modeled — real results
worse. Confirms and extends the weekly-ST study's conclusion: per-stock
trend-timing is a DD-reduction tool priced in CAGR, never a return-enhancer;
if DD control is the goal, index-level gating is cheaper.
Ledger +22 (Bonferroni: best-of-20 Calmar 0.89 is selection-flattered).
