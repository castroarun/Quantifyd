# RESULTS — Breakout book: gate & check-frequency optimisation (research/109)

**VERDICT: MOVE THE BOOK TO A WEEKLY CADENCE.** Same CAGR, materially shallower drawdown.
Changing the gate threshold is a dead end; changing how OFTEN the book acts is the real lever.

research/71 settled the exit rule, concurrency and that a NIFTY>200DMA gate is mandatory — but it only
tested 200DMA *versus no gate*, always checking entries and the trail DAILY. This closes both gaps.

## Headline

| Config | CAGR | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|
| CURRENT (ma200, daily entry, daily exit check) | 18.9% | -33.7% | 1.03 | 0.56 |
| **WEEKLY entry + WEEKLY exit check (same gate)** | **18.9%** | **-27.3%** | **1.03** | **0.69** |

Same return, 6.4pp less drawdown, Calmar +23%.

## Gate variants (daily cadence) — threshold tweaks do NOT help

| Gate | CAGR | MaxDD | Calmar |
|---|---|---|---|
| ma200 (current) | 18.9% | -33.7% | 0.56 |
| ma150 | 17.5% | -30.7% | 0.57 |
| ma200 hysteresis +/-3% | 16.4% | -28.7% | 0.57 |
| ma200 hysteresis +/-1% | 16.1% | -29.8% | 0.54 |
| ma200 checked weekly | 18.4% | -30.6% | 0.60 |
| ema200 | 14.9% | -34.5% | 0.43 |
| ma200 + rising slope | 13.6% | -30.7% | 0.44 |

Hysteresis was the intuitive fix for the boundary-whipsaw problem (the live book is currently paralysed
with NIFTY 0.19% below its 200-DMA). It buys almost nothing (0.57 vs 0.56). Faster/slower MAs and EMA
are worse. The 200-day SMA cross is already the right threshold.

## Frequency grid

| Entry check | Exit check | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| daily | daily (current) | 18.9% | -33.7% | 0.56 |
| daily | weekly | 17.2% | -29.2% | 0.59 |
| weekly | daily | 16.8% | -34.8% | 0.48 |
| **weekly** | **weekly** | **18.9%** | **-27.3%** | **0.69** |

Both halves must move together: weekly entries with daily exits is the WORST cell (0.48) because it
takes fewer positions while still being shaken out by daily noise.

**Mechanism:** daily checking buys single-day spikes and exits on intraweek wobble. A weekly close
filters both — entries are confirmed by a week of strength, and winners survive normal noise. Once the
book acts weekly the gate-check frequency becomes irrelevant (weekly-gate + weekly-cadence is identical
at 0.69), confirming the cadence is the lever, not the threshold.

## Caveats

- Inherits research/71's **survivorship bias** — 18.9% absolute is optimistic; the RELATIVE improvement
  (same return, less drawdown) is the trustworthy part.
- 12 configs tested. Mild multiple-testing risk, but the winner is a SIMPLER rule than the current one
  rather than a tuned parameter, which is the low-risk kind of finding.
- Gross of tax; ~45-50 day holds are short-term, so 20% STCG applies.
