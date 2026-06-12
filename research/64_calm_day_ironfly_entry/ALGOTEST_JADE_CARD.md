# AlgoTest Test Card — Jade Lizard / Defined-Risk Bullish Skew (NIFTY)

Goal: confirm on AlgoTest's REAL premiums what research/64 P5c-P5d found — that a bullish-skewed
structure beats the symmetric fly on NIFTY's upward drift — and settle the two things the proxy can't:
(1) the exact credit / whether the jade truly has "no upside risk" at a given VIX, (2) the real crash tail.

## Structures to run (weekly NIFTY, 10 lots / qty 650)
All entries 09:20, 2nd-nearest weekly (~4 TD-before-expiry on AlgoTest), exit 1-TD-before-expiry / roll.
Costs: Rs20/order, taxes on, 0.25% slippage. Report NET, ex-COVID. VIX entry band 13-22 (from P3).

| # | Structure | Legs (strikes by % OTM of spot) |
|---|---|---|
| 0 | **Baseline iron fly** | SELL ATM CE+PE, BUY +2% CE, BUY -2% PE |
| 1 | **Jade lizard (naked)** | SELL -2% PE, SELL +1% CE, BUY +2.5% CE |
| 2 | **Jade + 5% put (defined)** | #1 + BUY -5% PE |
| 3 | **Jade + 4% put (defined, recommended)** | #1 + BUY -4% PE |
| 4 | **Jade + 3.5% put (tight)** | #1 + BUY -3.5% PE |

Key checks on each run:
- **Net credit** received (and is credit >= the call-spread width 1.5%? if yes -> truly no upside risk).
- **Worst single week** (the crash tail) — compare naked (#1) vs defined (#2-4).
- Per-year P&L (positive-every-year is the bar), Calmar, MaxDD, win-rate.

## Read-out template
| Run | net P&L | win% | Calmar | MaxDD | worst week | +years | verdict |
|---|---|---|---|---|---|---|---|
| 0 iron fly | | | | | | | |
| 1 jade naked | | | | | | | |
| 2 jade +5% put | | | | | | | |
| 3 jade +4% put | | | | | | | |
| 4 jade +3.5% put | | | | | | | |

## What AlgoTest likely CANNOT express (do forward / manual)
- **Day-1 confirmation** (enter the skew only after a day-1 up-move > 0.5%): P5b/P5d showed this lifts EV
  +Rs23k and win +10pp while keeping the tail capped. AlgoTest has no "enter next day if up" rule -> test
  this **forward in paper** (or as a manual overlay): on the morning after a green day, put the jade on.
- VIX 13-22 band is expressible (entry VIX filter); the day-1 rule is not.

## Decision
Pick the structure with the best Calmar that's positive every year AND whose worst-week tail you can
stomach. P5d proxy expectation: **jade + 4% put** is the sweet spot (defined ~-200k tail vs naked -795k,
keeps +EV and ~71-81% win). Confirm the real credit covers the call spread (no upside risk) at VIX 13-22.
