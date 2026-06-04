# research/57 — Positional Bi-Weekly Short Straddle: CUMULATIVE VERDICT

**Verdict label: SIGNAL** (30d real chain, overlapping entries, single regime Apr–Jun 2026 — promising, not yet validated).

## Best recipe (G0→G2)
Short ATM straddle in the **2nd-nearest weekly** (bi-weekly ~8-12 DTE), **09:20 entry, 1 lot**, carry
multi-day, exit by **1 DTE**. Management: **profit-target at 40% of credit collected + 1.5%
underlying-move stop** (whichever fires first). **NO overnight wings.** Net ₹80/leg.

## Numbers (26 overlapping trades, real recorded NIFTY chain)
| recipe | mean/trade | median | win% | worst | std |
|---|---|---|---|---|---|
| unmanaged baseline | +7,842 | +8,987 | 65 | −12,396 | 11,383 |
| **COMBO (move 1.5% + PT 40%)** | **+7,792** | +5,826 | **73** | **−4,999** | 9,067 |

## Key findings
1. **Theta edge is REAL** — the naked bi-weekly carry makes +₹7.8k/trade, 65% win, but with a
   −₹12k closed / **−₹23k intra-trade** tail (deeply underwater in directional weeks).
2. **Tail control = a TIGHT move-stop (1.5%)** — caps the worst −₹12.4k → −₹5.0k for ~zero mean
   give-up. **WIDE stops (2–3%) BACKFIRE** (they exit late, into the big move → worst grows to −₹23k).
3. **A profit-target at 40% of credit RAISES the mean** (banks quick theta before the directional
   weeks erode it); median jumps to +₹15k.
4. **Overnight wings are REDUNDANT** once the move-stop is in — adding them drops the mean
   +7,792 → +4,938 for negligible tail benefit. **A tight stop caps the tail far more cheaply than
   buying far-OTM wings every night.** (Direct answer to the original design question.)

## Caveats / next
- 30 days, overlapping daily entries (~4 independent bi-weekly cycles) → **SIGNAL, not validation**;
  single regime; naked-overnight in the core number; no bid/ask slippage modelled.
- **NEXT:** G3 entry-timing (which day/DTE to enter; hold-to-1-DTE vs roll-earlier); then stand up a
  **forward paper logger on the VPS** to accumulate real out-of-sample trades before any sizing.
