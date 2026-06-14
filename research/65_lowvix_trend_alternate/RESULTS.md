# Low-VIX Trend Alternate — productive fill for the fly/jade idle (VIX<13) windows

**STATUS: G1 DONE — SIGNAL** (a plain NIFTY long in VIX<13 beats debt). research/65, complement to research/64.

## The ask
research/64 showed all 3 premium-selling systems (fly + bull/bear jade) only trade VIX 13-22, sitting IDLE
~31% of the year — concentrated in low-VIX (<13) years (2017/2023/2025). Goal: a system active EXACTLY in
those idle windows so the cash is not dead (vs a debt fund).

## G1 first pass (NIFTY daily 2015-2026, causal, VIX<13 = 22% of days)
| System (active when VIX<13) | days | total | ann (in-window) | Sharpe | maxDD |
|---|---|---|---|---|---|
| **Long every low-VIX day** | 637 | +43% | ~15%/yr | 0.74 | -7% |
| + price>50DMA | 533 | +36% | ~15% | 0.72 | -7% |
| + price>200DMA | 631 | +38% | ~15% | 0.69 | -7% |
| + 20d momentum>0 | 488 | +30% | ~14% | 0.62 | -7% |
| BuyHold NIFTY (ref) | 2835 | +185% | 10.7% | 0.66 | -38% |

**VERDICT: a plain NIFTY long while VIX<13 earns ~15%/yr in-window (Sharpe 0.74, DD -7%) — ~2x debt's
6.5% for a small drawdown. TREND FILTERS DON'T HELP (they cut days + lower return) -> the signal is just
"low-VIX = calm melt-up, be long". Regime-complementary: you exit as VIX rises through 13 = when the
fly/jade switch on.** Net-of-cost: NIFTY future/ETF long, cheap, low turnover.

## Caveats / next (G2+)
- A low-VIX regime can END in a vol spike + price drop; the VIX>=13 exit catches most but an overnight gap
  can hurt (the -7% DD captures some). Refine the exit (VIX trigger / a stop).
- Robustness owed: per-year stability, OOS/walk-forward, exact entry/exit mechanics, net-of-cost, the
  BLENDED book (fly/jade + low-VIX long on one pool) — does it lift the combined Calmar?
- Sizing: this is the underlying long; size vs the fly margin so the blended book is coherent.
