# RESULTS - Fresh-deposit deployment timing (research/112)

**VERDICT: SIGNAL - deploy fresh cash IMMEDIATELY as an EVEN top-up of names already held.
Do NOT use it to fill empty slots, and do NOT park it to month-end. Small but perfectly
consistent edge (12/12 phases).**

Arun asked which policy we had concluded for fresh deposited cash. Neither existing study covered
it: research/108 (stop-out cash) says MONTHLY, research/41 ph-27 (all-cash gate re-entry) says
WEEKLY. A deposit carries no adverse selection, so neither verdict transferred by analogy.

Period 2011-01-01 -> 2026-08-01 (15.6y), Nifty-200 top-8 book, net of 20%% STCG + 15bps round-trip,
idle cash at 6.5%%. Quarterly deposits of 5%% of a common reference NAV; every arm receives IDENTICAL
rupee cash flows on IDENTICAL dates, so the arm that ends richer wins. Re-run across 12 weekly
phase offsets of the deposit calendar.

| Deployment policy | med gain_x | med MaxDD | med idle | beats park |
|---|---|---|---|---|
| park | 1.566 | -22.8%% | 49.8%% | 0/12 |
| weekly | 1.531 | -22.2%% | 49.8%% | 1/12 |
| immediate | 1.560 | -22.4%% | 49.8%% | 3/12 |
| immediate_topup | 1.579 | -22.5%% | 49.3%% | 12/12 |

`gain_x` = terminal NAV / total contributed. It is a money-weighted ratio, NOT the book CAGR -
deposits scale with NAV, so contributed compounds alongside the book. Use it only to rank arms.

**Mechanism (consistent with research/108, not contradictory):**

- Filling EMPTY SLOTS quickly loses. Slots are empty because those names were stopped out, i.e. the
  market is falling - the false-dawn penalty. Both `immediate` (slot-fill first) and `weekly`
  underperform park, and `weekly` is worst because it does slot-filling most often.
- Topping up NAMES ALREADY HELD wins. Those names are positively selected: they survived the
  Donchian stop. Adding to survivors carries no adverse selection, and it beats letting the cash
  idle at 6.5%%.

The unifying rule: **deploy fresh cash into what is working; never rush it into what just broke.**

**Effect size - be honest:** +0.8%% terminal wealth over 15.6 years (~5bps/yr). The 12 phases share
one price history, so 12/12 is a consistent direction, NOT 12 independent trials. This is a
tiebreaker, not a material return driver. Idle cash barely moves (49.8%% -> 49.3%%) because the
dominant idle pool is gate risk-off periods and stop-outs, not deposits.

**Live status:** the deposit API `immediate` mode already implements exactly the winning policy -
`per = amount / len(pos)` over EXISTING positions only, no slot filling (services/momentum_paper.py
~line 1335). The UI control "Immediate equal top-up" is therefore the correct default for deposits,
while `park` should be reserved for a risk-OFF gate.

**Recommended change:** make `immediate` the default deposit mode when the gate is risk-ON.
