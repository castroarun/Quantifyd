# AlgoTest re-run spec — V2 iron fly, front weekly vs 2nd weekly

STATUS: **AWAITING USER RUN.** Two configurations to enter in the AlgoTest builder,
exported trade CSVs to come back for analysis.
research/141 · 2026-09-01

---

## Why this run exists

research/60 documented the AlgoTest results but did NOT retain the trade export
("Source: AlgoTest export (Trades CSV + PDF)"), so streaks, per-year detail and any
re-filtering cannot be recomputed. Separately, research/141 found the live engine
trades the **2nd-nearest weekly** while everything AlgoTest tested was the **front
weekly** - a lever the original sweep listed (#5) but never ran, because AlgoTest
caps entry at 4 trading days before expiry.

These two runs restore the missing export AND settle the expiry question with stops
priced at 1-minute resolution, which our end-of-day engine cannot do.

## SHARED SETTINGS - identical in both runs

| Field | Value |
|---|---|
| Instrument | NIFTY |
| Underlying from | **Cash** (so the SL reads SPOT - matches the live engine) |
| Strategy Type | Positional |
| Positional expire on | Weekly Expiry |
| Entry Time | **09:20** |
| Exit Time | **15:15** |
| Exit | **1 trading day before expiry** |
| Leg 1 | SELL · CALL · ATM · 10 lots |
| Leg 2 | SELL · PUT · ATM · 10 lots |
| Leg 3 | BUY · CALL · OTM **2.0% of ATM** · 10 lots |
| Leg 4 | BUY · PUT · OTM **2.0% of ATM** · 10 lots |
| Per-leg Stop Loss | **OFF** on all four legs |
| Per-leg Target | **OFF** on all four legs |
| Re-entry on SL | ON |
| Re-entry on Target | ON |
| Overall SL | **Underlying movement 2.0%** (NOT premium %) |
| Overall Target | **40% of premium received**, combined |
| Brokerage | Rs20 per order |
| Slippage | 0.25% |
| Period | 2019-02-01 -> today (or max available) |

## THE TWO RUNS - only these differ

| | RUN 1 (priority) | RUN 2 |
|---|---|---|
| Expiry | **FRONT weekly** (nearest) | **2nd weekly** - leg expiry "Next Weekly 2" if offered |
| Entry | **4 trading days before expiry** | 4 TD, or the maximum AlgoTest allows |
| Settles | reproduces the study; the one config both engines can compare | prices what is ACTUALLY RUNNING LIVE, with 1-min stops |

If AlgoTest offers no 2nd-weekly leg selection, RUN 1 alone is still worth having -
and confirming the 4-TD cap is itself the finding.

## DO NOT SET A VIX FILTER

AlgoTest has no VIX entry gate. research/60 applied **VIX >= 13 POST-HOC** from the
export's entry-VIX column. So:

- run **unfiltered** (all trades)
- ensure the export carries the **entry-VIX column**
- VIX >= 13 and the CPR < 0.10% skip are applied afterwards, on our side, exactly as
  the original study did

## WHAT TO SEND BACK

1. **Trades CSV** - the important one. Ideally: entry date, exit date, entry/exit
   time, strike per leg, entry/exit price per leg, P&L per leg and per trade, exit
   reason, **entry VIX**.
2. The PDF / summary page.
3. If quantity can only be 1 lot, that is fine - say so and the figures scale x10.

## WHAT WILL BE DONE WITH IT

| | |
|---|---|
| restore | per-trade streaks and the per-year heatmap for the AlgoTest side |
| settle | front weekly vs 2nd weekly with properly-priced intraday stops |
| check | our EOD engine against a 1-minute engine on the SAME config (RUN 1) |
| quantify | how wrong the EOD stop approximation is - our live-arm-D result leans on 225 approximated stop exits out of 286 trades (79%) |

## THE OPEN QUESTION THIS ANSWERS

Our engine says, VIX>=13, 10 lots, 2019-2026, net of costs:

| construction | net | maxDD | capital needed (2x DD or margin) | return on it |
|---|---:|---:|---:|---:|
| FRONT weekly, no stop | +Rs15,16,346 | -Rs3,44,672 | Rs8,24,580 | 24.6%/yr |
| **LIVE 2nd weekly, + stop** | **+Rs15,69,127** | **-Rs5,72,967** | **Rs11,45,934** | **18.3%/yr** |

Similar rupees, 39% more capital. But the live figure rests on 225 approximated stop
exits, so it is the least reliable number in the set. AlgoTest prices those stops
properly, which is exactly what is missing.
