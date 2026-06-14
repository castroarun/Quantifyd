
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
