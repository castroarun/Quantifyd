# RESULTS — S1/R1-Break Straddle Adjustment & Breakout-Flip

**Verdict: NO EDGE — the adjustment HURTS. Just holding the 09:16 straddle to 15:15 beats every
S1/R1 adjustment by a wide margin, and the reverse/breakout trade is a disaster. NIFTY intraday
around the daily pivots MEAN-REVERTS, so cutting the loser / chasing the break loses.**

65 trading days, per-minute recorded NIFTY chain, ATM straddle 2 lots (QTY 130), ST(7,3), floor
pivots from the prior day's regular-session OHLC.

## The numbers

| Mode | Total | n | Win% | Calmar | maxDD | worst day |
|---|--:|--:|--:|--:|--:|--:|
| **STRADDLE hold (no adjust)** | **+35,500** | 65 | 66 | **0.94** | −37,727 | −32,394 |
| STRADDLE cut-only | −40,768 | 65 | 49 | −0.45 | −90,201 | −21,103 |
| STRADDLE trail-PRICE | +9,388 | 65 | 60 | 0.29 | −32,499 | −11,587 |
| STRADDLE trail-PREMIUM | +15,037 | 65 | 55 | 0.76 | −19,711 | −11,587 |
| BREAKOUT trail-PRICE | −74,705 | 65 | 38 | −1.02 | −73,292 | −14,527 |
| BREAKOUT trail-PREMIUM | −71,888 | 65 | 37 | −0.97 | −73,864 | −16,476 |

Break days (a 5-min close beyond R1/S1): **56 of 65** — so breaks are common, and on those days:
- HOLD +15,261 (win 66%) · cut-only −61,007 · trail-PRICE −10,851 · trail-PREMIUM −5,202.

## Why every adjustment loses

- **Cutting the loser is the single worst move (−40,768).** On an up-break we cut the losing CE and
  keep the PE; NIFTY then frequently mean-reverts, so the CE we cut *would have recovered* (we locked
  its loss) and the PE we kept now loses. Double hit. HOLD avoids both.
- **The breakout/reverse trade (short PE on R1, short CE on S1, flip) is a disaster (−72 to −75k).**
  It assumes pivot breaks *continue*; intraday they mostly revert, so the naked directional short is
  repeatedly whipsawed — entered right at the turn.
- **Trailing the survivor** (premium a bit better than price) recovers some of the damage but still
  loses heavily to HOLD on break days.
- Between trails, **PREMIUM > PRICE** (both here and on break-days), consistent with the existing NAS
  design using premium-ST for the naked survivor.

## Read

The short straddle's edge is theta + mean-reversion; the S1/R1 signal fights that. The pivot break is
a *late, coarse* signal that mostly marks the intraday extreme (the reversion point), so acting on it
— cutting, trailing, or flipping — sells the low / buys the high. **Do not build this system.** If
downside protection is wanted, the research/90 portfolio stop (−₹1,300/lot, no target on NIFTY) is
the right tool, not a pivot-based adjustment.

## Caveats
65 days, optimistic fills (no slippage, 1/5-min resolution), ST(7,3) only. But the effect is large and
uniform across all six variants, so the direction is robust: the adjustment subtracts value.
