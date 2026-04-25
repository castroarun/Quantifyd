# Bollinger Squeeze Breakout — Findings (NEGATIVE)

**Status:** Killed. All 6 filter variants negative; only HEROMOTOCO profitable
across variants (1 of 15 stocks). Pattern matches research/13 (VWAP-MR) and
research/15 (MA cross).

## Spec

- **Compression detection:** BB(20, 2σ) bands fully INSIDE KC(20, 1.5×ATR)
  channels for ≥ 6 bars (30 min on 5-min)
- **Fire signal:** squeeze releases (BB exits KC), AND breakout bar closes
  beyond BB upper (long) or lower (short)
- **Trade rules:** entry next bar open · stop 1×ATR · target 2.0R ·
  18-bar time stop · EOD 15:15 · 1 trade/stock/day
- **Universe:** 15 non-ORB liquid F&O (same as research 14, 15)
- **Period:** 2024-03-18 → 2026-03-12 (~500 trading days)
- **Sizing:** Rs 2,500 risk · Rs 3L cap · 0.15% round-trip costs

Detected ~280 squeeze fires per stock over 2 years (~17,000 squeezes
across the universe before filtering).

## Headline

| Variant | Trades | WR% | PF | Net P&L | Sharpe |
|---|---:|---:|---:|---:|---:|
| baseline | 1188 | 33.0 | 0.38 | −Rs 5,12,160 | −11.33 |
| rsi50 | 1188 | 33.0 | 0.38 | −Rs 5,12,160 | −11.33 |
| vwap | 1129 | 33.3 | 0.39 | −Rs 4,78,426 | −10.84 |
| volspike (≥1.5× avg) | 485 | 33.6 | 0.34 | −Rs 2,15,895 | −9.84 |
| htf (60m EMA21 slope) | 615 | 31.7 | 0.35 | −Rs 2,83,201 | −9.96 |
| confluence (all 4) | 256 | 32.8 | 0.36 | −Rs 1,12,305 | −8.37 |

**Best PF: 0.39.** Need >1.0.

## Filter signal — what we learned

- `rsi50` is **redundant** with breakout direction. A squeeze that fires UP
  through BB upper inherently has bullish momentum → RSI almost always >50 at
  that bar. Filter rejected ~0% of trades. Worth dropping in any future spec.
- `volspike` (≥1.5× 20-bar avg) is the **strongest single filter**: cuts
  trades 60% (1188→485), reduces loss by 58%. Volume confirmation does
  measure something real but not enough to flip PF.
- `htf` (60-min EMA21 slope agreement) cuts ~50%, reduces loss 45%.
  Trend-alignment filter helps but again not enough.
- `confluence` (volspike + htf + vwap) cuts 80%, smallest absolute loss
  (−Rs 1.1L on 256 trades). Highest filter selectivity, still not
  profitable.

## Per-stock breakdown (confluence variant)

Only HEROMOTOCO profitable: +Rs 4,685 on 10 trades.
14 of 15 stocks negative. No curated-universe rescue available — the
breakeven sample size is ~5-20 trades per stock, too noisy to draw
conclusions from any one positive name.

## Why it fails

1. **Win rate structurally stuck at 32-34%.** At 2.0R target / 1.0R stop,
   that's 0.33×2 − 0.67×1 = −0.01R/trade. Razor-thin negative *before* costs.
   0.15% round-trip slippage slips it deeply negative.
2. **Squeeze direction is a coin flip.** A volatility compression doesn't
   predict expansion direction. Roughly half of fires that close above BB
   upper get reversed within 18 bars. The "breakout direction" signal is
   weaker than common TA folklore suggests on Indian cash 5-min bars.
3. **2.0R target is ambitious.** Lowering to 1.5R might lift WR but the
   math gets worse: at 33% WR, R:R 1:1.5 = 0.33×1.5 − 0.67×1 = −0.16R.
   No R:R configuration with this WR is viable.
4. **Costs.** 0.15% × N trades is meaningful drag at 1188 trades.

## What worked, what didn't (cumulative across research 13/15/16)

| Approach | Result |
|---|---|
| VWAP fade (research 13) | Dead — WR 30%, PF 0.13-0.41 |
| MA crossover any TF (research 15) | Dead — WR 36-40%, PF 0.36-0.55 |
| **BB squeeze (research 16)** | **Dead — WR 32-34%, PF 0.34-0.39** |
| Volume breakout curated (research 14 v2) | **Viable — WR 44%, PF 1.35** |

## The cumulative lesson

Generic price-derived intraday signals on Indian cash 5-min bars do not
clear PF 1.0 — not VWAP-fade, not MA-cross, not BB-squeeze, not even with
HTF/RSI/CPR/BB filter stacks. Win rates lock in the 30-40% range and 1.0-2.0R
targets can't bridge the gap once 0.15% costs apply.

What does work:
- **ORB:** ties signal to a specific time-of-day structure (opening range)
- **Volume breakout v2:** ties signal to volume confirmation on quality
  names with 2.5R target

Both have a *specific structural anchor* — opening range, or volume +
quality universe. Generic indicator confluence across the broad day does
not produce edge.

## Decision

Stop hunting for generic intraday MA/squeeze/fade systems. Two productive
paths from here:

1. **Productionize volume breakout v2** — already viable on the curated
   universe. Walk-forward validate, then build live executor.
2. **Try a structurally different signal family entirely** — pair trading
   on F&O futures (market-neutral, different P&L driver), or post-15:00
   "last hour" momentum (specific time-window signal).

## Artifacts

- `scripts/run_bb_squeeze.py` — backtest engine
- `results/summary.csv` — 6 variants
- `results/trades.csv` — full trade log
- `results/daily_pnl.csv` — daily P&L per variant
- `logs/run_v1.log` — full run output
