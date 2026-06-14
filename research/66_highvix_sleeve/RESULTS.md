# High-VIX (>22) Sleeve — close the last idle (the chaos tail)

**STATUS: G1 DONE — SIGNAL** (mean-reversion long, timed). research/66, complement to research/64 + 65.

## The ask
The full always-on book (fly + jades + low-VIX long) is deployed ~85%/yr; the remaining ~15% idle is
almost entirely VIX>22 (the chaos regime, e.g. COVID-2020 = 123 idle days). Goal: a sleeve for VIX>22.

## G1 first pass (NIFTY daily 2015-26, causal, VIX>22 = 10% of days)
| system (active VIX>22) | days | ann (in-window) | Sharpe | maxDD | win% |
|---|---|---|---|---|---|
| Long every VIX>22 (buy-fear) | 297 | 41% | 0.40 | -33% | 56% |
| **Long VIX>22 + VIX FALLING** | 119 | **81%** | 0.47 | **-10%** | 55% |
| Long VIX>22 + price>20DMA | 138 | 35% | 0.34 | -9% | 57% |
| Long VIX>22 + RSI<35 | 56 | 20% | 0.09 | -33% | 59% |
| **SHORT every VIX>22** | 297 | **-36%** | -0.40 | -56% | 44% |

**VERDICT: buy-the-fear WORKS — NIFTY mean-reverts UP from VIX spikes (long +41%, SHORT catastrophic
-36%/-56% DD). Timing is key: buying EVERY high-VIX day catches the falling knife (-33% DD); the keeper =
long when VIX>22 AND VIX FALLING (spike peaked) -> 81% in-window, -10% DD, OR price>20DMA (-9% DD). Wait
for the peak, then ride the bounce.** HIGH-RISK regime -> run DEFINED-RISK (debit spread), not naked long.
Sample-thin (COVID-2020 dominates) -> robustness owed (per-year, OOS, costs, the option structure).

## This closes the rotation
VIX<13 -> low-VIX long (G3b intraday exit) | VIX 13-22 -> fly + bull/bear jades | VIX>22 -> mean-rev long
(VIX-falling). A productive sleeve for EVERY VIX regime -> no structural idle. Next: robustness on each,
the defined-risk option version of the high-VIX long, and the real combined ₹/Calmar (needs jade premiums).

---

## ROBUSTNESS (walk-forward halves + per-year + net-of-cost 0.05%/trade) — DONE
| Sleeve | 2015-20 | 2021-26 | verdict |
|---|---|---|---|
| low-VIX long (2% trail) | +10% Sh0.46 DD-6% | +22% Sh0.76 DD-7% | ROBUST (positive both halves, Sharpe rising) |
| high-VIX long (VIX-fall) | +36% Sh0.68 DD-11% | +17% Sh0.57 DD-9% | SIGNAL but THIN (2020 dominates H1; H2 from 2021/22/26) |
Low-VIX edge concentrated in low-VIX years (2017 +13, 2023 +23); net-of-cost fine (low turnover). High-VIX
positive both halves so NOT just COVID, but 10% of days + clustered -> thin; -9/-11% DD is real.

## DEFINED-RISK high-VIX — simple underlying stop FAILS
5% stop cuts return (2021 +9->+2) WITHOUT capping DD (-11% from intra-5% gaps). True defined-risk = an
OPTION DEBIT SPREAD (caps loss at spread width) -> needs premiums (AlgoTest). So #3 is premium-blocked too.

## STATUS of the program (research/64+65+66)
Price-action research EXHAUSTED. Every VIX regime has a validated SIGNAL: <13 low-VIX long (robust),
13-22 fly+jades, >22 mean-rev long (thin). REMAINING WORK ALL NEEDS REAL OPTION PREMIUMS (user's AlgoTest
runs / forward recorder): (#2) combined ₹/Calmar incl jades, (#3) defined-risk debit-spread versions.
Coverage/utilization solid; blended ₹ pending. AlgoTest cards written: ALGOTEST_TEST_CARD + ALGOTEST_JADE_CARD.
