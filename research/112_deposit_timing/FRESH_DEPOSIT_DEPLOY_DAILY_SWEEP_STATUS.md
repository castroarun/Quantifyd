# Fresh-Deposit Deployment Timing — When Should New Cash Enter the Momentum Book?

STATUS: DONE

## 1. The Ask

**What Arun asked:** "i thought we have discussed on the deployment of fresh cash process
earlier and it will be either done as and when the cash comes in or as part of the weekly
cycle (not sure which 1 we concluded)"

**What we are actually testing:** The book has TWO studied cash policies that point in
OPPOSITE directions, and fresh deposits fall under neither:

| Cash situation | Policy | Evidence |
|---|---|---|
| Stop-out cash (partial book, slots emptied by Donchian) | MONTHLY | research/108 - weekly halves net Calmar 0.91 -> 0.45 |
| All-cash gate re-entry (gate flips back risk-ON) | WEEKLY | research/41 ph-27 - Calmar 1.72 |
| **Fresh deposited cash** | **UNTESTED** | this study |

The research/108 monthly verdict rests on ADVERSE SELECTION: slots are empty precisely
because those names were falling, so fast refill buys weakness (the false-dawn penalty).
A deposit carries no such selection - it arrives for exogenous reasons (Arun had spare
cash). So the monthly rule may not transfer. That is the question.

## 2. The Base

- Book: Nifty-200 universe, rsblend (6m/12m RS vs NIFTYBEES), top-8 equal weight,
  top-22 anti-churn buffer, 15-day Donchian EOD stop, weekly NIFTYBEES<100-SMA cash gate,
  month-end rotate-only rebalance. Identical across all arms.
- Period: 2011-01-01 -> 2026-08-01 (15.6y). Net of STCG 20% + 15bps round-trip. Idle cash 6.5%.
- Deposit schedule: every quarter, sized at 5% of a COMMON REFERENCE NAV (the park arm)
  so every arm receives IDENTICAL rupee cash flows on IDENTICAL dates. Without this the
  richer arm would deposit more and the comparison would be circular.
- Robustness: the whole calendar is re-run at 12 weekly phase offsets, so the verdict is a
  DISTRIBUTION over 12 deposit calendars, not one lucky schedule.
- Success criterion: median terminal NAV across the 12 phases (identical cash in => richer
  arm wins), with max DD and net Calmar as the risk check. A winner must beat park on the
  MEDIAN and not be materially worse on drawdown.

## 3. Arms

| Arm | When deposited cash is deployed |
|---|---|
| `park` | held in liquid; enters at the next month-end rebalance (CURRENT LIVE BEHAVIOUR) |
| `weekly` | at the next weekly gate check if risk-ON: fill empty slots first, then top up held |
| `immediate` | same day if gate risk-ON: fill empty slots first, then top up held |
| `immediate_topup` | same day if gate risk-ON: top up EXISTING holdings only (what the deposit API `immediate` mode does today) |

If the gate is risk-OFF every arm parks (matches the live code).

## 4. Plan

4 arms x 12 phase offsets = 48 backtest paths. Ranking snapshots precomputed once.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-14 14:40 IST | Folder + STATUS written, study launched | 48 paths |
| 2026-08-14 14:47 IST | All 48 paths complete | immediate_topup wins 12/12 |
| 2026-08-14 14:52 IST | RESULTS.md written, verdict SIGNAL | see results/RESULTS.md |

## 6. Crash Recovery

- Progress: `tail -f /home/arun/quantifyd/research/112_deposit_timing/results/run.log`
- Alive? `pgrep -af run_deposit_timing`
- Resume (idempotent, overwrites results): `cd /home/arun/quantifyd && nohup ./venv/bin/python3 research/112_deposit_timing/scripts/run_deposit_timing.py > research/112_deposit_timing/results/run.log 2>&1 &`
- Partial results land in `results/deposit_timing.csv` (one row per arm x phase, written incrementally).
- Safe to inspect: everything under `results/`. Do NOT edit `scripts/` mid-run.
- This study places NO orders and touches NO live state. The live book is unaffected.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_deposit_timing.py` | Study runner | yes |
| `FRESH_DEPOSIT_DEPLOY_DAILY_SWEEP_STATUS.md` | This file | yes |
| `results/deposit_timing.csv` | Per arm x phase results | yes (small) |
| `results/run.log` | Progress log | yes (small) |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings

**VERDICT: SIGNAL — deploy fresh deposits IMMEDIATELY as an EVEN top-up of names already HELD.**

| Policy | med gain_x | med MaxDD | beats park |
|---|---|---|---|
| park (was live default) | 1.566 | -22.8% | 0/12 |
| weekly | 1.531 | -22.2% | 1/12 |
| immediate (slot-fill first) | 1.560 | -22.4% | 3/12 |
| **immediate_topup (held names only)** | **1.579** | -22.5% | **12/12** |

Mechanism, and it RECONCILES r/108 rather than contradicting it: filling EMPTY SLOTS fast loses
(those slots are empty because those names were stopped out — the false-dawn penalty), but topping
up names already HELD wins (they survived the Donchian, so they are positively selected). Unifying
rule: **deploy fresh cash into what is working; never rush it into what just broke.**

Effect is small and honestly stated: +0.8% terminal over 15.6y (~5bps/yr). The 12 phases share one
price history, so 12/12 is a consistent DIRECTION, not 12 independent trials. Tiebreaker, not a
return driver.

The live `immediate` deposit mode already implements the winner (equal rupee across existing
positions only, no slot filling). Owed: make `immediate` the DEFAULT when the gate is risk-ON.
