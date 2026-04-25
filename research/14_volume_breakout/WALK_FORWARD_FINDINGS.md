# Walk-Forward Validation — Volume Breakout v2 (FAILED)

**Status:** Strategy as specified does NOT have a real edge. The +38% CAGR /
PF 1.35 / Sharpe 1.96 result in the original v2 was hindsight overfit.

## Procedure

1. **TRAIN window** 2024-03-18 → 2025-03-15 (~12 mo): ran v1 broad-universe
   spec on all 15 candidate stocks, computed per-stock PF, kept stocks with
   train PF >= 0.9 → train-curated universe.
2. **TEST window** 2025-03-16 → 2026-03-12 (~12 mo): applied v2 spec
   (rr_2.5_spike_3x_rsi) ONLY to train-curated set. Never-touched data.
3. **Pass criteria** OOS PF >= 1.15 AND OOS Sharpe >= 0.8.

## Result — fails both gates

| Metric | In-sample (train) | Out-of-sample (test) | Degradation |
|---|---:|---:|---:|
| Trades | 53 | 17 | −68% |
| Win rate | 54.7% | **29.4%** | −25 pp |
| Profit factor | 2.13 | **0.79** | −1.34 |
| Sharpe | 3.72 | **−1.56** | −5.28 |
| Net P&L (Rs 3L cap) | +Rs 41,385 | −Rs 3,614 | sign flip |

**Train period selection** would have picked 7 stocks (vs the v2 hindsight 9):
MARUTI, HEROMOTOCO, EICHERMOT, TITAN, ASIANPAINT, HINDALCO, LT.
JSWSTEEL and JINDALSTEL would have been correctly excluded — train period saw
their weakness — but the remaining 7 still collapsed in OOS.

## Per-stock IS → OOS

| Stock | IS PF | OOS PF | OOS net |
|---|---:|---:|---:|
| MARUTI | 2.21 | 0.00 | −Rs 1,904 |
| **HEROMOTOCO** | 1.18 | **2.39** | **+Rs 2,788** |
| EICHERMOT | 1.29 | 0.00 | −Rs 92 |
| TITAN | 2.10 | 0.00 | −Rs 6,518 |
| ASIANPAINT | 14.07 | 0.00 | +Rs 5,912 (1 trade only) |
| HINDALCO | 2.27 | 0.48 | −Rs 2,640 |
| LT | 2.51 | 0.09 | −Rs 1,160 |

Only **HEROMOTOCO** maintained its edge OOS — and on just 3 trades, low
statistical weight. The "winners" universally collapsed.

## Diagnosis

Textbook overfit pattern:
1. Per-stock sample sizes were too small (5-11 IS trades each) for stable
   selection. Random luck of the 2024-25 period showed up as "edge."
2. The original 121-trade portfolio result was meaningfully driven by
   ASIANPAINT's 6 trades returning +Rs 19K (54% of total profit). When that
   one stock didn't repeat, the portfolio collapsed.
3. The train-period universe selection rule (PF >= 0.9 cutoff on small
   samples) doesn't generalize.

## Conclusion

**Do not advance v14 volume breakout to live trading.** The headline 38%
CAGR was an artifact of in-sample selection.

Combined with research/13 (VWAP fade dead), research/15 (MA cross dead),
research/16 (BB squeeze dead), this is the **fourth clean negative result**
on intraday directional signals on Indian cash 5-min bars. Pattern: generic
intraday price-action signals don't survive OOS unless anchored to a
specific structural setup (ORB's opening-range concentration is the only
exception in this codebase).

## Pivot

Moving to **EOD swing momentum scanner** in research/17 — different
timeframe (daily bars), different hold style (multi-day), wider universe
(Nifty 500). Fresh test, not a fix to this one.

## Artifacts

- `scripts/walk_forward.py` — train+OOS runner
- `results/walk_forward/verdict.txt` — pass/fail summary
- `results/walk_forward/per_stock.csv` — IS vs OOS per-stock breakdown
- `logs/walk_forward.log` — full run output
