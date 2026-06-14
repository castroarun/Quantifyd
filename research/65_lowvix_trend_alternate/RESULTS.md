# Low-VIX Trend Alternate — productive fill for the fly/jade idle (VIX<13) windows

**STATUS: G1 DONE — SIGNAL** (a plain NIFTY long in VIX<13 beats debt). research/65, complement to research/64.

## The ask
research/64 showed all 3 premium-selling systems (fly + bull/bear jade) only trade VIX 13-22, sitting IDLE
~31% of the year — concentrated in low-VIX (<13) years (2017/2023/2025). Goal: a system active EXACTLY in
those idle windows so the cash is not dead (vs a debt fund).

## G1 first pass (NIFTY daily 2015-2026, causal, VIX<13 = 22% of days)
| System (active when VIX<13) | days | total | ann (in-window) | Sharpe | maxDD |
|---|---|---|---|---|---|
| **Long every low-VIX day** | 637 | +43% | ~15%/yr | 0.74 | -7% |
| + price>50DMA | 533 | +36% | ~15% | 0.72 | -7% |
| + price>200DMA | 631 | +38% | ~15% | 0.69 | -7% |
| + 20d momentum>0 | 488 | +30% | ~14% | 0.62 | -7% |
| BuyHold NIFTY (ref) | 2835 | +185% | 10.7% | 0.66 | -38% |

**VERDICT: a plain NIFTY long while VIX<13 earns ~15%/yr in-window (Sharpe 0.74, DD -7%) — ~2x debt's
6.5% for a small drawdown. TREND FILTERS DON'T HELP (they cut days + lower return) -> the signal is just
"low-VIX = calm melt-up, be long". Regime-complementary: you exit as VIX rises through 13 = when the
fly/jade switch on.** Net-of-cost: NIFTY future/ETF long, cheap, low turnover.

## Caveats / next (G2+)
- A low-VIX regime can END in a vol spike + price drop; the VIX>=13 exit catches most but an overnight gap
  can hurt (the -7% DD captures some). Refine the exit (VIX trigger / a stop).
- Robustness owed: per-year stability, OOS/walk-forward, exact entry/exit mechanics, net-of-cost, the
  BLENDED book (fly/jade + low-VIX long on one pool) — does it lift the combined Calmar?
- Sizing: this is the underlying long; size vs the fly margin so the blended book is coherent.

---

## G2 — blended book (REAL fly ₹ + REAL low-VIX long) — DONE; verdict = PROMISING but naive long too crude
Blend: V2 AlgoTest fly ₹ (VIX≥13) + low-VIX NIFTY long (₹8L notional, VIX<13) on one pool, vs fly+idle-in-debt.

| Year | fly ₹ | lowVIX days | lowVIX ret | blended ₹ | fly+debt ₹ |
|---|---|---|---|---|---|
| 2019 | +10,930 | 19 | -2.7% | -10,906 | +14,851 |
| 2020 | +166,432 | 4 | -1.4% | +154,901 | +167,257 |
| 2021 | +57,973 | 26 | +2.6% | +78,803 | +63,338 |
| 2022 | +201,312 | 2 | -1.0% | +193,010 | +201,725 |
| **2023** | +62,331 | **175** | **+24.5%** | **+258,107** | +98,442 |
| 2024 | +293,918 | 47 | +1.9% | +308,980 | +303,616 |
| **2025** | +104,912 | 130 | +4.7% | +142,844 | +131,737 |
| 2026* | -17,698 | 24 | -5.7% | -63,173 | -12,746 |
| TOTAL | | | | **+10.6L** | +9.7L |

**VERDICT: lift +₹94k/7yr but ENTIRELY from 2023 (+196k) + 2025 (+38k) — sustained low-VIX bull grinds where
the long fills the fly's thinnest years (2023 = 175 low-VIX days @ +24.5%, perfect complement). BUT the naive
"always long in low-VIX" LOSES in short/choppy low-VIX patches (2019 -2.7%, 2026 -5.7%) -> adds directional
VARIANCE (crude annual DD 13k->63k). G1's aggregate +15% MASKED this lumpiness; G2 caught it.** The
complementarity is real (2023), but the naive long is too crude.

## G3 (next) — refine the low-VIX entry/exit
Keep the 2023/2025 sustained-grind upside, cut the 2019/2026 chop drag: test a trend-confirmation entry +
a stop / VIX-rising exit (the simple MA filter in G1 cut return without fixing per-year — need better). Also:
proper daily/monthly blended equity (real Calmar, not annual), net-of-cost, sizing/leverage, OOS.
Caveat: DD here is annual-granularity; the fly's real intraday DD is ~-₹1.17L (V2 study).

---

## G3 — refined low-VIX long (trend / trailing stop) — DONE
| variant | days | total | Sharpe | maxDD | 2023 | 2026 |
|---|---|---|---|---|---|---|
| naive (VIX<13) | 637 | 43% | 0.74 | -7% | +24.5% | -5.7% |
| **+2% trail stop** | 623 | **49%** | **0.84** | -8% | +26.9% | -5.8% |
| +20DMA trend | 606 | 36% | 0.67 | -7% | +22.3% | -5.7% |
| +20DMA +2% trail | 515 | 39% | 0.77 | -6% | +21.6% | -3.7% |
| +50DMA +3% trail | 568 | 35% | 0.67 | -7% | +19.7% | -6.3% |

**A 2% TRAILING STOP is the keeper: total 43→49%, Sharpe 0.74→0.84, keeps 2023 (+27%). Trend filters cut
red years but sacrifice too much return.** The 2019/2026 drag is the VIX-EXIT LAG (regime ends in a 1-day
spike+drop; next-day daily exit catches it) — a DAILY stop can't dodge a single-day reversal.
**G3b (next): INTRADAY VIX exit using the 5-min INDIAVIX (downloaded 2015-26) — exit the long the moment
VIX crosses 13 intraday, not next-day. The real fix for the lag.** Then re-blend the best variant w/ the fly.

---

## FULL always-on book — all 3 (fly + bull jade + bear jade) + low-VIX long (2% trail), one pool — DONE
Per-year deployment by sleeve (single pool, priority low-VIX-long when <13, else fly/jade in 13-22):
AVG/yr: low-VIX long 45d | fly+jade 157d | idle 34d | **DEPLOYED 202d (85%)**.
Deployment ladder: neutral-only 43% -> all-3 fly/jade 69% -> +low-VIX long 85%. Idle ~15% now almost
entirely VIX>22 (2020 = 123 idle days / 51% deployed; the chaos regime neither sleeve covers).
Low-VIX long carries the low-VIX years (2017 +12%, 2023 +21%, 2025 +5.5%) = exactly when premium-sellers thin.
=> idle-cash problem largely SELF-SOLVED by the systems; only the VIX>22 tail remains (debt, or a future
high-VIX 3rd sleeve). ₹ status: fly real (V2), low-VIX long real (2% trail), JADES need their AlgoTest runs
for real ₹ -> combined ₹/Calmar pending jade premiums; coverage/utilization above is solid.

---

## G3b — INTRADAY VIX exit (5-min INDIAVIX) — DONE; the keeper
Exit the low-VIX long the MOMENT a 5-min VIX bar >=13 (vs naive next-day), 5-min NIFTY+VIX 2015-26:
| exit | days | total | Sharpe | maxDD | 2020 | 2022 | 2026 |
|---|---|---|---|---|---|---|---|
| naive next-day | 623 | 44% | 1.61 | -9% | -1.5 | -1.1 | -5.9 |
| **INTRADAY VIX>=13** | 623 | **58%** | **2.17** | **-7%** | -0.7 | -0.3 | -4.5 |
Intraday exit lifts total 44->58%, Sharpe 1.61->2.17, DD -9->-7%, trims every spike-drag year, keeps 2023
(+21%). The lag fix WORKS. (Sharpe basis = deployed-day 5-min returns, higher than the daily-series 0.74.)

**LOW-VIX LONG FINAL SPEC: long NIFTY (future/ETF) while VIX<13; 2% trailing stop; EXIT INTRADAY the moment
5-min VIX>=13.** Closes the lag drag of the naive daily exit.
