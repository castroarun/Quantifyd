# Phase 26 — cash-flow allocation / de-allocation policy

**Verdict: the system is ROBUST to cash-flow policy. None of the 20
inflow×outflow rules materially helps or hurts — final-wealth spread
< 1%, post-tax XIRR within 0.4pp, drawdown identical. Marginal best:
deploy contributions into the highest-RS holdings (C3); raise
withdrawals cash-first then pro-rata (W1). Practical takeaway: do NOT
over-engineer the live deposit/withdrawal logic — the simplest rule is
within noise of the best.**

## Setup

Base = SMOOTHEST + keep-top8 (the chosen best) on the validated
Phase-22 weekly daily-marked engine, fresh VPS data → 2026-05-15,
initial = 1.0. Fixed deterministic flow schedule (identical across all
policies so every delta is attributable): monthly SIP +1% of initial;
lump deposits +0.50 (2017-06, 2023-01); lump withdrawals −0.40 (2019-09),
**−0.40 forced at the 2020-03-23 COVID trough (drawdown stress)**, −0.30
(2025-06). Metrics: time-weighted CAGR (unitised), money-weighted XIRR
(investor-experienced), post-tax@20% (lot-level), daily MaxDD on
per-unit value.

## Grid (5 inflow × 4 outflow = 20)

| | TWR | XIRR | XIRR post-tax | dailyDD | final |
|---|---|---|---|---|---|
| all 20 combos | 33.5–33.6% | 32.1–32.2% | 26.5–26.9% | −20.2% | 46.7–47.2× |
| **best — C3/W1** | 33.6 | 32.2 | **26.9** | −20.2 | **47.17** |
| simplest — C1/W1 | 33.5 | 32.1 | 26.5 | −20.2 | 46.80 |

Inflow rules: C1 park-till-rebalance · C2 deploy-now equal-weight ·
**C3 deploy-now into highest-RS holdings (marginal best)** · C4
deploy-now pro-rata · C5 regime-aware. Outflow rules: W1 cash-first
then pro-rata · W2 lowest-RS-first · W3 tax-aware (LTCG lots first) ·
W4 trim-overweight. 

## Read

- **Inflows:** deploying promptly into the strongest holdings (C3) is
  the best, but only by ~0.4pp post-tax / ~0.8% final wealth over 12y
  vs parking (C1). The parked-cash drag (6.5% debt vs ~33% strategy) is
  real but tiny because rebalances absorb it within weeks.
- **Outflows:** the rule is **irrelevant** — W1/W2/W3/W4 differ in the
  3rd decimal. Even tax-aware lot selection (W3) gives no edge: the
  monthly rebuild + 12-year horizon washes out lot-level STCG timing.
- **Drawdown stress (2020-03-23 forced withdrawal):** no policy was
  penalised — daily MaxDD is −20.2% across the board, identical to the
  no-flow keep-top8. Forced selling into the COVID trough did not
  meaningfully scar any policy because the withdrawal is small vs the
  book and the system re-equal-weights the following month anyway.

## Live recommendation

Robustness is the headline. For implementation simplicity adopt
**C3 + W1**: new cash → top up the highest-RS current holdings on the
next dealing day; withdrawals → idle cash first, then pro-rata trim.
But **C1 + W1 (park till rebalance / cash-first) is statistically
indistinguishable and operationally simplest** — a safe default that
needs no special engine machinery beyond the existing monthly rebuild.

Artifact: `phase26_cashflow_policy.csv`. Runner
`scripts/26_cashflow_policy.py` (unitised TWR + bisection XIRR +
lot-level tax on the keep-top8 base).
