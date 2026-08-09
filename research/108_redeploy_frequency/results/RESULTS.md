# RESULTS — Redeploy frequency for stopped-out cash (research/108)

**VERDICT: KEEP MONTHLY. Faster redeployment halves the Calmar.**

Arun asked whether stop-out cash should be redeployed at the weekly gate check instead of waiting for
the month-end rebalance. The code cited research/41 phase-27 for rejecting partial-state re-entry, but
that was the midcap-RS book, so it was re-tested on this Nifty-200 top-8 book (2011-2026, net of STCG).

| Refill policy | net CAGR | Max DD | Sharpe | net Calmar | Avg idle cash |
|---|---|---|---|---|---|
| MONTHLY (current live) | 27.1% | -17.0% | 1.70 | **0.91** | 56.4% |
| WEEKLY refill | 22.1% | -25.3% | 1.33 | 0.45 | 37.9% |
| DAILY refill | 18.9% | -36.0% | 1.18 | 0.28 | 32.3% |

Faster refill does cut cash drag (56% to 38% to 32% idle) but the deployed money performs badly:
-5pp CAGR and an 8pp deeper drawdown at weekly, worse at daily.

**Mechanism:** slots are empty precisely BECAUSE those names were just stopped out, i.e. the market is
falling. Refilling quickly buys into weakness, gets stopped out again, and repeats - the false-dawn
penalty. It replicates on this book, confirming the original research/41 finding generalises.

The idle cash is therefore not a defect to fix but the strategy correctly declining to catch a falling
knife. The right response is to make idle cash EARN (LIQUIDCASE sweep), not to deploy it sooner.

**Caveat:** modelled with the old cash-exit gate, so the 56% idle figure includes full risk-off periods.
With the put hedge now holding through risk-off, idle cash comes only from stop-outs - a smaller pool -
but the direction of the finding is unaffected.
