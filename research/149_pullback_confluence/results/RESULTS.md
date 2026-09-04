# Research 149 — Pullback Family Done Properly — RESULTS

**VERDICT: NO EDGE — 0 of 96 cells has positive after-tax expectancy. The pullback-reversal
sketch is now closed with a clean conscience: it was tested at every MA length {20/50/100/200}
× {SMA, EMA} × six confluence filters × two exits, and every single cell loses money net.**
The pre-registered resurrection bar (positive after-tax expectancy + neighbor plateau + both
windows) was never approached — the BEST of 96 cells is −0.195%/trade (ema20 + CCI-turn),
and the family averages −0.5 to −1.1%/trade across MA lengths.

> Same engine, candle mechanics, universe, costs and tax as r/146 (green-after-red at the MA
> touch, buy-stop above the green high, SL below min(green low, prior red low), hard stops,
> no averaging, PIT top-500, 25bps/side, FY-netted tax). 96 cells × gross/after-tax = 192
> runs, 160s. **96 cells were tested — any "best cell" must be discounted as best-of-96;
> none needed discounting because none was positive.**

## Answers to Arun's specific question ("did u study other sma/emas? rsi, rs, stochs, ccrb?")

| Axis | Answer |
|---|---|
| Other MAs | ALL of 20/50/100/200, SMA and EMA: mean expectancy −0.52 (ema20, least bad) to −1.06 (sma200, worst). Shorter MA = shallower pullback = less bad, still negative. |
| RSI(14)<40 | mean −0.82, best −0.43. Negative everywhere. |
| RSI(2)<10 | mean −0.94, best −0.51. WORST filter — deep short-term oversold makes the candle entry worse, not better. |
| RS-percentile ≥70 (pullback in relative-strength LEADERS — the classically claimed edge) | mean −0.74, best −0.58. **The "buy pullbacks in leaders" story does not survive this entry/exit mechanic either.** |
| Stoch(14)<20 turning up | mean −0.69, best −0.38. Negative. |
| CCI(20)<−100 turning up | mean −0.59, best −0.195 — the least-bad family, still negative in every cell. |

Best 3 cells (after-tax, tradeability gate): ema20+CCI 2R/15d: n=634, WR 38.3%, −0.195%/tr,
max losing streak 15; ema50+CCI 2R/15d: n=1,878, WR 38.0%, −0.319%/tr, streak 29;
ema20+CCI 2R/10d: n=640, WR 39.7%, −0.352%/tr, streak 16. Even the least-bad cells fail the
tradeability gate on losing streaks alone.

## Why it loses (structural, consistent across all 96 cells)

The construction pays a bad price twice: the buy-stop above the green candle enters AFTER
the bounce is underway (worst fill of the reversal), and the tight candle-low stop sits
inside normal noise — WR ~30-40% with 1.5-2R winners never covers it, before costs. The
confluence filters shrink n but do not change the geometry. This matches r/146's 50-MA
result, r/84 (dip-buy), and r/87-88 (structure screens): the fourth independent kill of
buy-the-dip-in-uptrend entries on this data.

## What was NOT tested, and why

- Exit re-optimization per cell (would be third-pass mining on a 96-cell dead grid).
- Weekly-timeframe pullbacks, gap-driven variants — different family; nothing here suggests
  the daily mechanics deserve the extension.
- The signal DAYS still feed research/150 (options-structure overlay) — the ~60% directional
  claim is tested there in payoff-restructured form, which is the honest way to keep any
  value from the idea.

## Reproducibility

`research/149_pullback_confluence/scripts/pull_sweep.py` on the shared r/146 engine
(`sleeve_engine.py`, additive back-compatible extension: RSI14/Stoch/CCI/RS-percentile/MA
grid). `results/pullback_grid.csv` (192 rows). Data: market_data.db snapshot 2026-09-04.
