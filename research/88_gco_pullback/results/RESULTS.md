# research/88 — GCO-Pullback-Stoch: RESULTS

## Verdict: **NO EDGE — the entry adds nothing beyond long exposure + trailing exit**

24 cells (fresh-window {10,20} x stop {trigger-low, 2-bar-low} x exit
{2R target, sma20-trail, time-15} x L/S), daily, 74 F&O names with usable
history, IS 2005-2017, 10bps RT, vs pre-registered RANDOM-ENTRY baselines
running identical exit mechanics.

| Finding | Evidence |
|---|---|
| Best long cell: F20/SL2/sma20-trail | +43.9bps/trade, t=2.07 (fails t>=2.5 gate) |
| ...loses to RANDOM entries + same exit | baseline +46.7bps, t=7.85 |
| All target/time-stop long cells negative | -37..+5bps |
| ALL short cells negative | -22..-70bps (random shorts -60..-96 — less bad, still losing; shorts' 5th burial in this program) |
| Exit >> entry (replicates r/71) | sma20-trail positive everywhere for longs; 2R target and time stops negative |

The golden-cross + first-pullback + stochastic-confirmation entry — a
classic of trading literature — selects entries that perform NO BETTER than
entering every 10th bar blind, once the same exits and costs are applied.
What looked like 'the system works' in any casual backtest would have been
(a) long drift, (b) the trailing exit, (c) survivorship.

Ledger: +30 cells (24 + 6 baselines). Program total r/87+88: 134. OOS untouched.
